"""1D backbone for cyclic-chain datasets (Lorenz-96 line, DATA_PLAN Sec. 3).

Geometry-free twin of :class:`HuracanSFNO`: identical two-frame ``step``
convention (emitting ``model.outputs_per_call`` future states), identical
encoder input layout ``[x_{k-1}, x_k, f_{k-1}, f_k, (obs_{k-1}, obs_k)]``,
identical NoiseEmbedder -> conditional LayerNorm machinery and residual
near-persistence output init.  Everything downstream -- phased training,
checkpoint column growing (:func:`load_with_new_columns` matches on
``embed.weight``), EDA cycling -- runs unchanged; only the trunk differs:
the spectral Swin blocks become cyclic Conv1d + MLP blocks, the right
inductive bias for Lorenz-96 whose dynamics couple only k+-1, k+-2
(kernel 9 over 4 blocks spans the whole 40-site ring).

Selected by ``model.backbone: mlp1d`` (dispatch in core.model.build_models;
mirrors the ``grid.type`` planar dispatch).  ``npix = nlat * nlon`` of the
config grid is the only geometry this model ever sees.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import ModelCfg
from .modules import ConditionalLayerNorm, NoiseEmbedder


class Huracan1DBlock(nn.Module):
    """x + Conv1d(cyclic)(SpecNorm(x)) then x + MLP(MixerNorm(x)), both
    conditioned on the per-pixel noise embedding (AdaSpeech style)."""

    def __init__(self, dim: int, embed_dim: int, kernel: int, mlp_ratio: float):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"cyclic conv kernel must be odd, got {kernel}")
        self.spec_norm = ConditionalLayerNorm(dim, embed_dim)
        self.spec = nn.Conv1d(
            dim,
            dim,
            kernel,
            padding=kernel // 2,
            padding_mode="circular",
        )
        self.mixer_norm = ConditionalLayerNorm(dim, embed_dim)
        hidden = int(dim * mlp_ratio)
        self.mixer = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # x: (B, npix, dim); e: per-pixel noise embedding (B, npix, embed)
        h = self.spec_norm(x, e)
        x = x + self.spec(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.mixer(self.mixer_norm(x, e))
        return x


class Huracan1D(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_forcing: int,
        n_obs: int,
        mcfg: ModelCfg,
        npix: int,
        use_obs: bool = False,
    ):
        super().__init__()
        self.n_state = n_state
        self.use_obs = use_obs
        self.residual = mcfg.prediction == "residual"
        self.outputs_per_call = mcfg.outputs_per_call
        self.n_in = 2 * (n_state + n_forcing) + (2 * n_obs if use_obs else 0)

        self.noise = NoiseEmbedder(mcfg.noise_dim, mcfg.noise_embed_dim, npix)
        self.embed = nn.Linear(self.n_in, mcfg.hidden)
        self.blocks = nn.ModuleList(
            [
                Huracan1DBlock(
                    mcfg.hidden,
                    mcfg.noise_embed_dim,
                    mcfg.conv_kernel,
                    mcfg.mlp_ratio,
                )
                for _ in range(mcfg.blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(mcfg.hidden)
        self.out = nn.Linear(mcfg.hidden, self.outputs_per_call * n_state)
        if self.residual:
            # near-persistence init, same rationale as HuracanSFNO:
            # tiny-but-nonzero W keeps gradients alive for the encoder
            # (a strictly zero W would freeze the trunk early in training)
            nn.init.normal_(self.out.weight, std=1e-3)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """x: (B, npix, n_in); z: noise (B, npix, noise_dim) or None.

        Returns (B, npix, outputs_per_call*n_state): future states starting
        at k+1 (deltas when residual mode is on)."""
        B, N, _ = x.shape
        if z is None:
            z = x.new_zeros(B, N, self.noise.mlp[0].in_features)
        e, _g = self.noise(z)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, e)
        return self.out(self.out_norm(h))

    def step(self, x_tm1, x_t, f_tm1, f_t, obs_tm1=None, obs_t=None, z=None):
        """One model call: (k-1, k) -> the next ``outputs_per_call`` states
        -- the dataset-agnostic form of the paper's (T-6h, T) ->
        (T+6h, T+12h) convention."""
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
