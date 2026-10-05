"""Huracan model: modified SFNO with two-timestep input/output convention.

Framework-free (pure ``torch.nn``): no Lightning, no YAML, no filesystem
side effects beyond the SHT basis cache.  The model takes explicit tensor
dimensions plus its own config sections, and the spherical-harmonic backend
is injected through ``sht_factory`` so torch-harmonics can replace the
matrix backend at nside=64 without touching this file.

Both the assimilation and the forecasting model share this architecture and
differ only in the number of encoder input channels (paper Sec. 2.2):
  input  = [ x_{T-6h}, x_T, forcing_{T-6h}, forcing_T, (obs_{T-6h}, obs_T) ]
  output = deltas for x_{T+6h} and x_{T+12h}, applied residually on x_T.
The time dimension is flattened into channels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ..config.schema import GridCfg, HuracanConfig, ModelCfg
from . import grid as G
from .modules import HuracanSFNOBlock, NoiseEmbedder, _run_block
from .sht import MatrixSHT, SHTBackend

SHTFactory = Callable[[], SHTBackend]


class HuracanSFNO(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_forcing: int,
        n_obs: int,
        mcfg: ModelCfg,
        gcfg: GridCfg,
        use_obs: bool = False,
        sht_factory: SHTFactory | None = None,
    ):
        super().__init__()
        self.n_state = n_state
        self.use_obs = use_obs
        self.n_in = 2 * (n_state + n_forcing) + (2 * n_obs if use_obs else 0)

        nside, lmax = gcfg.nside, gcfg.lmax
        colat, lon = G.pixel_centers(nside)
        colat, lon = np.asarray(colat), np.asarray(lon)

        if sht_factory is None:
            def sht_factory() -> SHTBackend:
                return MatrixSHT(nside, lmax, colat, lon, cache_dir="data/cache")

        # single shared transform, registered exactly once; every block
        # receives it through forward()
        self.sht: SHTBackend = sht_factory()

        self.noise = NoiseEmbedder(mcfg.noise_dim, mcfg.noise_embed_dim, G.npix(nside))
        self.embed = nn.Linear(self.n_in, mcfg.hidden)
        self.grad_checkpoint = bool(mcfg.grad_checkpoint)
        self.blocks = nn.ModuleList(
            [
                HuracanSFNOBlock(
                    mcfg.hidden,
                    nside,
                    lmax,
                    mcfg.k_filters,
                    mcfg.swin_window,
                    mcfg.heads,
                    mcfg.noise_embed_dim,
                    mcfg.mlp_ratio,
                )
                for _ in range(mcfg.blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(mcfg.hidden)
        self.out = nn.Linear(mcfg.hidden, 2 * n_state)
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
                    _run_block, blk, h, e, g, self.sht, use_reentrant=False
                )
            else:
                h = blk(h, e, g, self.sht)
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


def build_models(
    cfg: HuracanConfig, sht_factory: SHTFactory | None = None
) -> tuple[HuracanSFNO, HuracanSFNO]:
    """Returns (forecast_model, assim_model) with correctly sized encoders."""
    n_state = cfg.n_prognostic()
    n_forc = cfg.n_forcing()
    n_obs = cfg.n_obs_channels()
    fcst = HuracanSFNO(
        n_state, n_forc, n_obs, cfg.model, cfg.grid, use_obs=False,
        sht_factory=sht_factory,
    )
    assim = HuracanSFNO(
        n_state, n_forc, n_obs, cfg.model, cfg.grid, use_obs=True,
        sht_factory=sht_factory,
    )
    return fcst, assim


def load_with_new_columns(model: nn.Module, state: dict) -> None:
    """Copy weights from a narrower encoder into a wider one; weight tensors
    whose shapes differ keep their fresh random initialization, except the
    first Linear of the encoder where only the new input *columns* are new
    (paper Sec. 2.3: 'except for the new columns in the encoder')."""
    own = model.state_dict()
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif (
            own[k].dim() == 2 and own[k].shape[0] == v.shape[0] and "embed.weight" in k
        ):
            own[k][:, : v.shape[1]] = v
    model.load_state_dict(own)
