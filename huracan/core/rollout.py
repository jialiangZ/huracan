"""The single source of truth for the autoregressive unroll convention.

Temporal semantics (how many frames a call emits, how calls map to hours)
live in ``huracan.time.CallSemantics`` -- never hard-coded here.  Per the
paper (Sec. 2.2/2.3, guide Sec. 1.2) a call ingests ``(T-6h, T)`` and
emits ``outputs_per_call`` future fields (default two: T+6h, T+12h); the
single-output setting (``model.outputs_per_call: 1``) gives the
GenCast-style one-field-per-call convention.

``unroll``       -- training-time rollout (teacher-forced blocks, ensemble dim)
``spinup_and_forecast`` -- EDA inference (spin-up cycling then forecast)

Both are pure functions over explicit tensors; the training LightningModule
and the EDA runner compose them instead of re-implementing the slicing
convention (call ``j`` consumes frames ``opc*j, opc*j+1`` of obs/forcing,
likewise; the sliding ``(x_prev, x_cur)`` pair advances by ``opc`` frames).
"""

from __future__ import annotations

import torch

from ..time import CallSemantics


def _opc_of(model, outputs_per_call: int | None) -> int:
    """Resolve the per-call output count: explicit argument wins, else the
    model's own attribute (models are the authority on their head width)."""
    if outputs_per_call is not None:
        return outputs_per_call
    return model.outputs_per_call


def member_noise(model, B: int, npix: int, device, generator=None) -> torch.Tensor:
    """Per-pixel Gaussian noise sized for ``model``'s noise embedder."""
    return torch.randn(
        B, npix, model.noise.mlp[0].in_features, device=device, generator=generator
    )


def smooth_noise(
    B: int,
    C: int,
    grid_hw: tuple[int, int],
    device,
    generator=None,
) -> torch.Tensor:
    """Spectrally-consistent (spatially smooth) unit-std noise.

    Background errors in real DA live at large scales, unlike iid pixel
    noise which a smooth spectral model can filter out on its own.  This
    draws white noise and box-filters it (circular in longitude, zero-pad
    in latitude), renormalizing to unit std per sample -- an adequate
    large-scale-error model for OSSE perturbations.

    Returns (B, nlat*nlon, C) in the row-major (lat, lon) pixel order."""
    import torch.nn.functional as F

    nlat, nlon = grid_hw
    n = torch.randn(B, C, nlat, nlon, device=device, generator=generator)
    k = max(1, nlat // 4)
    if k > 1:
        kl, kr = (k - 1) // 2, k // 2
        n = F.pad(n, (kl, kr, 0, 0), mode="circular")
        n = F.avg_pool2d(n, (1, k), stride=1)
        n = F.pad(n, (0, 0, kl, kr))
        n = F.avg_pool2d(n, (k, 1), stride=1)
    n = n / (n.flatten(1).std(dim=1).view(B, 1, 1, 1) + 1e-6)
    return n.permute(0, 2, 3, 1).reshape(B, nlat * nlon, C)


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
    ic_noise: float = 0.0,
    grid_hw: tuple[int, int] | None = None,
    outputs_per_call: int | None = None,
):
    """Autoregressive training unroll.

    models: {"assim": nn.Module, "fcst": nn.Module}; calls with index
      < split_assim use the assimilation model (with observations), later
      calls use the forecast model.
    state:   (B, window_frames(n_calls), npix, C) normalized states, where
      window_frames = 2 + outputs_per_call*n_calls (see CallSemantics).
    forcing: same frame count as state.
    obs:     same frame count as state, or None.
    outputs_per_call: fields per call; None -> read from the models.
    ic_noise: max amplitude (normalized units) of background-error
      injection on the unroll's starting state -- each member draws a
      per-sample scale from U(0, ic_noise) and perturbs x_{T-6h}, x_T
      with *spectrally-consistent* smooth noise (``grid_hw`` required).
      This puts the assimilation model in the regime where the background
      is *worse* than the observations, so fusing obs carries real
      training signal (the OSSE spin-up regime); targets stay clean.

    Returns:
      preds   (B, M, outputs_per_call*n_calls, npix, C); M = ensemble if
              use_noise else 1
      targets (B, outputs_per_call*n_calls, npix, C)  = state[:, 2:]
    """
    B, _, npix, C = state.shape
    opc = _opc_of(models["fcst"], outputs_per_call)
    M = ensemble if use_noise else 1
    preds = []
    for _m in range(M):
        x_prev, x_cur = state[:, 0], state[:, 1]
        if ic_noise > 0:
            if grid_hw is None:
                raise ValueError("ic_noise > 0 requires grid_hw=(nlat, nlon)")
            scale = ic_noise * torch.rand(
                B, 1, 1, device=state.device, generator=generator
            )
            p1 = smooth_noise(B, C, grid_hw, state.device, generator)
            p2 = smooth_noise(B, C, grid_hw, state.device, generator)
            x_prev = x_prev + scale * p1
            x_cur = x_cur + scale * p2
        outs = []
        for j in range(n_calls):
            assim_call = j < split_assim
            model = models["assim" if assim_call else "fcst"]
            lo = opc * j
            z = (
                member_noise(model, B, npix, state.device, generator)
                if use_noise
                else None
            )
            kw = {}
            if assim_call and obs is not None:
                kw = dict(obs_tm1=obs[:, lo], obs_t=obs[:, lo + 1])
            step_out = model.step(
                x_prev, x_cur, forcing[:, lo], forcing[:, lo + 1], z=z, **kw
            )
            outs += list(step_out)
            chain = [x_prev, x_cur, *step_out]
            x_prev, x_cur = chain[-2], chain[-1]
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
    collect_spinup: bool = False,
    ic_noise: float = 0.0,
    grid_hw: tuple[int, int] | None = None,
    outputs_per_call: int | None = None,
):
    """EDA inference trajectory (paper Sec. 3.1).

    state: (2 + outputs_per_call*(spinup+lead), npix, C) normalized,
      starting at t0-1; forcing likewise; obs covers the spin-up window
      only (may be None).
    outputs_per_call: fields per call; None -> read from ``assim``.

    Returns (spinup_end (M, npix, C), preds (M, outputs_per_call*lead,
    npix, C)), plus the ensemble-mean spin-up trajectory
    (outputs_per_call*spinup, npix, C) when ``collect_spinup`` is set
    (OSSE analysis-error curves).

    Each member seeds the global RNG (seed_base + m) exactly once, matching
    the original inference behaviour.  Passing the *forecast* model as
    ``assim`` (with obs=None) yields the no-assimilation control.

    ``ic_noise`` > 0 perturbs each member's starting state with iid
    Gaussian noise of that scale (normalized units) -- the OSSE setting
    where the spin-up must *recover* the trajectory from a bad initial
    condition instead of merely maintaining a perfect one.
    """
    npix = state.shape[1]
    opc = _opc_of(assim, outputs_per_call)
    spinup_end, member_preds = [], []
    member_spin = []
    for m in range(members):
        torch.manual_seed(seed_base + m)
        x_prev, x_cur = state[0:1].clone(), state[1:2].clone()  # (t0-6h, t0)
        if ic_noise > 0:
            if grid_hw is None:
                raise ValueError("ic_noise > 0 requires grid_hw=(nlat, nlon)")
            p1 = smooth_noise(1, x_prev.shape[-1], grid_hw, state.device)
            p2 = smooth_noise(1, x_cur.shape[-1], grid_hw, state.device)
            x_prev = x_prev + ic_noise * p1
            x_cur = x_cur + ic_noise * p2
        spin_out = []
        for j in range(spinup):  # ═══ spin-up 段：assim ═══
            lo = opc * j
            z = member_noise(assim, 1, npix, state.device)
            kw = {}
            if obs is not None:
                kw = dict(obs_tm1=obs[lo : lo + 1], obs_t=obs[lo + 1 : lo + 2])
            step_out = assim.step(
                x_prev,
                x_cur,
                forcing[lo : lo + 1],
                forcing[lo + 1 : lo + 2],
                z=z,
                **kw,
            )
            if collect_spinup:
                spin_out += list(step_out)
            chain = [x_prev, x_cur, *step_out]
            x_prev, x_cur = chain[-2], chain[-1]
        spinup_end.append(x_cur.cpu())
        if collect_spinup:
            member_spin.append(torch.cat(spin_out, dim=0).cpu())
        preds = []
        for j in range(spinup, spinup + lead):  # ═══ 预报段：fcst ═══
            lo = opc * j
            z = member_noise(fcst, 1, npix, state.device)
            step_out = fcst.step(
                x_prev,
                x_cur,
                forcing[lo : lo + 1],
                forcing[lo + 1 : lo + 2],
                z=z,
            )
            preds += list(step_out)
            chain = [x_prev, x_cur, *step_out]
            x_prev, x_cur = chain[-2], chain[-1]
        member_preds.append(torch.cat(preds, dim=0).cpu())
    if collect_spinup:
        spin_mean = torch.stack(member_spin, dim=0).mean(dim=0)
        return (
            torch.cat(spinup_end, dim=0),
            torch.stack(member_preds, dim=0),
            spin_mean,
        )
    return torch.cat(spinup_end, dim=0), torch.stack(member_preds, dim=0)
