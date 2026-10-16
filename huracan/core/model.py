"""Huracan model: modified SFNO with the two-frame input convention.

Framework-free (pure ``torch.nn``): no Lightning, no YAML.  The model takes
explicit tensor dimensions, its own config section and a ``SphericalGrid``
geometry object; the spherical-harmonic backend is injected through
``sht_factory`` so torch-harmonics can replace the matrix backend at paper
scale without touching this file.

Both the assimilation and the forecasting model share this architecture and
differ only in the number of encoder input channels (paper Sec. 2.2):
  input  = [ x_{T-6h}, x_T, forcing_{T-6h}, forcing_T, (obs_{T-6h}, obs_T) ]
  output = ``model.outputs_per_call`` future states starting at T+6h
  (paper: two fields, T+6h and T+12h; time flattened into channels).
The temporal convention itself lives in ``huracan.time.CallSemantics``.
The output parameterization (residual on x_T vs direct state) is a [REPO]
choice -- the paper does not specify it (guide Sec. 8.6); the default keeps
the residual form with a near-persistence output layer, which trains stably.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ..config.schema import HuracanConfig, ModelCfg
from .grid import SphericalGrid, grid_from_cfg
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
        grid: SphericalGrid,
        use_obs: bool = False,
        sht_factory: SHTFactory | None = None,
    ):
        super().__init__()
        self.n_state = n_state
        self.use_obs = use_obs
        self.residual = mcfg.prediction == "residual"
        self.outputs_per_call = mcfg.outputs_per_call
        self.n_in = 2 * (n_state + n_forcing) + (2 * n_obs if use_obs else 0)

        if sht_factory is None:

            def sht_factory() -> SHTBackend:
                return MatrixSHT(grid, cache_dir="data/cache")

        # single shared transform, registered exactly once; every block
        # receives it through forward()
        self.sht: SHTBackend = sht_factory()

        self.noise = NoiseEmbedder(mcfg.noise_dim, mcfg.noise_embed_dim, grid.npix)
        self.embed = nn.Linear(self.n_in, mcfg.hidden)
        self.grad_checkpoint = bool(mcfg.grad_checkpoint)
        window = (mcfg.swin_window_lat, mcfg.swin_window_lon)
        self.blocks = nn.ModuleList(
            [
                HuracanSFNOBlock(
                    mcfg.hidden,
                    grid,
                    mcfg.k_filters,
                    window,
                    mcfg.heads,
                    mcfg.noise_embed_dim,
                    mcfg.mlp_ratio,
                )
                for _ in range(mcfg.blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(mcfg.hidden)
        self.out = nn.Linear(mcfg.hidden, self.outputs_per_call * n_state)
        if self.residual:
            # near-persistence initialization: tiny (but non-zero) output
            # weights keep the untrained model close to the persistence
            # baseline, while a *strictly* zero W would zero every gradient
            # flowing into the encoder/blocks (dL/dh = W^T dL/dy = 0),
            # freezing the whole network until W itself grows -- fine at the
            # paper's 347k steps, fatal for short runs.
            nn.init.normal_(self.out.weight, std=1e-3)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, npix, n_in); z: noise (B, npix, noise_dim) or None.

        Returns (B, npix, outputs_per_call*n_state): future states starting
        at T+6h (deltas when residual mode is on)."""
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
        """One model call: (T-6h, T) -> the next ``outputs_per_call`` states.

        One call advances the trajectory by ``outputs_per_call`` frames (12 h
        at the paper's two 6-hourly outputs; see ``huracan.time``)."""
        parts = [x_tm1, x_t, f_tm1, f_t]
        if self.use_obs:
            assert obs_tm1 is not None and obs_t is not None, (
                "assimilation model requires observations"
            )
            parts += [obs_tm1, obs_t]
        inp = torch.cat(parts, dim=-1)
        out = self.forward(inp, z)
        deltas = out.chunk(self.outputs_per_call, dim=-1)
        if self.residual:
            return tuple(x_t + d for d in deltas)
        return tuple(deltas)


def build_models(
    cfg: HuracanConfig, sht_factory: SHTFactory | None = None
) -> tuple[nn.Module, nn.Module]:
    """Returns (forecast_model, assim_model) with correctly sized encoders.

    Builds HuracanSFNO on the configured grid (gauss sphere or planar
    square)."""
    n_state = cfg.n_prognostic()
    n_forc = cfg.n_forcing()
    n_obs = cfg.n_obs_channels()

    grid = grid_from_cfg(cfg.grid)
    if sht_factory is None and cfg.grid.type == "planar":
        from .planar_fft import PlanarSHT

        def sht_factory() -> SHTBackend:
            return PlanarSHT(grid)

    fcst = HuracanSFNO(
        n_state,
        n_forc,
        n_obs,
        cfg.model,
        grid,
        use_obs=False,
        sht_factory=sht_factory,
    )
    assim = HuracanSFNO(
        n_state,
        n_forc,
        n_obs,
        cfg.model,
        grid,
        use_obs=True,
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
