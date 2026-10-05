"""Huracan model: modified SFNO with two-timestep input/output convention.

Both the assimilation and the forecasting model share this architecture and
differ only in the number of encoder input channels (paper Sec. 2.2):
  input  = [ x_{T-6h}, x_T, forcing_{T-6h}, forcing_T, (obs_{T-6h}, obs_T) ]
  output = deltas for x_{T+6h} and x_{T+12h}, applied residually on x_T.
The time dimension is flattened into channels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import grid as G
from .modules import HuracanSFNOBlock, NoiseEmbedder, _run_block


class HuracanSFNO(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_forcing: int,
        n_obs: int,
        cfg: dict,
        use_obs: bool = False,
        cache_dir: str = "data/cache",
    ):
        super().__init__()
        m = cfg["model"]
        self.n_state = n_state
        self.use_obs = use_obs
        self.n_in = 2 * (n_state + n_forcing) + (2 * n_obs if use_obs else 0)

        nside = cfg["grid"]["nside"]
        colat, lon = G.pixel_centers(nside)
        colat, lon = np.asarray(colat), np.asarray(lon)

        self.noise = NoiseEmbedder(m["noise_dim"], m["noise_embed_dim"], G.npix(nside))
        self.embed = nn.Linear(self.n_in, m["hidden"])
        self.grad_checkpoint = bool(m.get("grad_checkpoint", True))
        self.blocks = nn.ModuleList(
            [
                HuracanSFNOBlock(
                    m["hidden"],
                    nside,
                    cfg["grid"]["lmax"],
                    colat,
                    lon,
                    m["k_filters"],
                    m["swin_window"],
                    m["heads"],
                    m["noise_embed_dim"],
                    m["mlp_ratio"],
                    cache_dir,
                )
                for _ in range(m["blocks"])
            ]
        )
        self.out_norm = nn.LayerNorm(m["hidden"])
        self.out = nn.Linear(m["hidden"], 2 * n_state)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, npix, n_in); z: noise (B, npix, noise_dim) or None.

        Returns (B, npix, 2*n_state): [delta_{T+6h}, delta_{T+12h}]."""
        B, N, _ = x.shape
        if z is None:
            z = x.new_zeros(B, N, self.noise.mlp[0].in_features)
        e, g = self.noise(z)
        h = self.embed(x)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training and h.requires_grad:
                h = torch.utils.checkpoint.checkpoint(
                    _run_block, blk, h, e, g, use_reentrant=False
                )
            else:
                h = blk(h, e, g)
        return self.out(self.out_norm(h))

    # ------------------------------------------------------------------
    def step(self, x_tm1, x_t, f_tm1, f_t, obs_tm1=None, obs_t=None, z=None):
        """One model call: (T-6h, T) -> (T+6h, T+12h) with residual on x_t."""
        parts = [x_tm1, x_t, f_tm1, f_t]
        if self.use_obs:
            assert obs_tm1 is not None and obs_t is not None, (
                "assimilation model requires observations"
            )
            parts += [obs_tm1, obs_t]
        inp = torch.cat(parts, dim=-1)
        deltas = self.forward(inp, z)
        d1, d2 = deltas.chunk(2, dim=-1)
        return x_t + d1, x_t + d2


def build_models(cfg: dict, cache_dir: str = "data/cache"):
    """Returns (forecast_model, assim_model) with correctly sized encoders."""
    from .config import n_prognostic, n_obs_channels

    n_state = n_prognostic(cfg)
    n_forc = len(cfg["data"]["forcing"])
    n_obs = n_obs_channels(cfg)
    fcst = HuracanSFNO(n_state, n_forc, n_obs, cfg, use_obs=False, cache_dir=cache_dir)
    assim = HuracanSFNO(n_state, n_forc, n_obs, cfg, use_obs=True, cache_dir=cache_dir)
    return fcst, assim
