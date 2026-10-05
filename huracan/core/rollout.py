"""The single source of truth for the two-timestep unroll convention.

``unroll``       -- training-time rollout (teacher-forced blocks, ensemble dim)
``spinup_and_forecast`` -- EDA inference (spin-up cycling then forecast)

Both are pure functions over explicit tensors; the training LightningModule
and the EDA runner compose them instead of re-implementing the slicing
convention (obs at ``obs[:, 2j]``/``obs[:, 2j+1]``, forcing likewise).
"""

from __future__ import annotations

import torch


def member_noise(model, B: int, npix: int, device, generator=None) -> torch.Tensor:
    """Per-pixel Gaussian noise sized for ``model``'s noise embedder."""
    return torch.randn(
        B, npix, model.noise.mlp[0].in_features, device=device, generator=generator
    )


def unroll(
    models: dict,
    state: torch.Tensor,
    forcing: torch.Tensor,
    obs: torch.Tensor | None,
    *,
    n_calls: int,
    split_assim: int,
    ensemble: int = 1,
    use_noise: bool = False,
    generator=None,
):
    """Autoregressive training unroll.

    models: {"assim": nn.Module, "fcst": nn.Module}; calls with index
      < split_assim use the assimilation model (with observations), later
      calls use the forecast model.
    state:   (B, 2*n_calls+2, npix, C) normalized states.
    forcing: (B, 2*n_calls+2, npix, F).
    obs:     (B, 2*n_calls+2, npix, n_obs) or None.

    Returns:
      preds   (B, M, 2*n_calls, npix, C)   M = ensemble if use_noise else 1
      targets (B, 2*n_calls, npix, C)      = state[:, 2:]
    """
    B, _, npix, _ = state.shape
    M = ensemble if use_noise else 1
    preds = []
    for _m in range(M):
        x_prev, x_cur = state[:, 0], state[:, 1]
        outs = []
        for j in range(n_calls):
            assim_call = j < split_assim
            model = models["assim" if assim_call else "fcst"]
            z = member_noise(model, B, npix, state.device, generator) if use_noise else None
            kw = {}
            if assim_call and obs is not None:
                kw = dict(obs_tm1=obs[:, 2 * j], obs_t=obs[:, 2 * j + 1])
            x1, x2 = model.step(
                x_prev, x_cur, forcing[:, 2 * j], forcing[:, 2 * j + 1], z=z, **kw
            )
            outs += [x1, x2]
            x_prev, x_cur = x1, x2
        preds.append(torch.stack(outs, dim=1))
    return torch.stack(preds, dim=1), state[:, 2:]


@torch.no_grad()
def spinup_and_forecast(
    assim,
    fcst,
    state: torch.Tensor,
    forcing: torch.Tensor,
    obs: torch.Tensor | None,
    *,
    members: int,
    spinup: int,
    lead: int,
    seed_base: int = 1000,
):
    """EDA inference trajectory (paper Sec. 3.1).

    state: (2*(spinup+lead)+2, npix, C) normalized, starting at t0-1;
    forcing likewise; obs covers the spin-up window only (may be None).

    Returns (spinup_end (M, npix, C), preds (M, 2*lead, npix, C)).
    Each member seeds the global RNG (seed_base + m) exactly once, matching
    the original inference behaviour.
    """
    npix = state.shape[1]
    spinup_end, member_preds = [], []
    for m in range(members):
        torch.manual_seed(seed_base + m)
        x_prev, x_cur = state[0:1].clone(), state[1:2].clone()
        for j in range(spinup):
            z = member_noise(assim, 1, npix, state.device)
            kw = {}
            if obs is not None:
                kw = dict(obs_tm1=obs[2 * j : 2 * j + 1], obs_t=obs[2 * j + 1 : 2 * j + 2])
            x1, x2 = assim.step(
                x_prev, x_cur, forcing[2 * j : 2 * j + 1],
                forcing[2 * j + 1 : 2 * j + 2], z=z, **kw,
            )
            x_prev, x_cur = x1, x2
        spinup_end.append(x_cur.cpu())
        preds = []
        for j in range(spinup, spinup + lead):
            z = member_noise(fcst, 1, npix, state.device)
            x1, x2 = fcst.step(
                x_prev, x_cur, forcing[2 * j : 2 * j + 1],
                forcing[2 * j + 1 : 2 * j + 2], z=z,
            )
            preds += [x1, x2]
            x_prev, x_cur = x1, x2
        member_preds.append(torch.cat(preds, dim=0).cpu())
    return torch.cat(spinup_end, dim=0), torch.stack(member_preds, dim=0)
