"""Core neural modules of Huracan: modified SFNO building blocks.

Implements the three architecture modifications of the paper (Sec. 2.2):
  1. Swin Transformer block replacing an MLP inside each SFNO block.
  2. Compressed spectral filters (Sec. A.2): k shared filters with
     degree-dependent convex weights h(l) from a sinusoidal degree
     encoding concatenated with the noise embedding.
  3. Noise embeddings -> conditional layer normalization (AdaSpeech-style).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .grid import SphericalGrid
from .sht import SHTBackend
from .swin import SwinLatLonBlock


# ---------------------------------------------------------------------------
# Noise embedding + conditional layer norm (modification 3)
# ---------------------------------------------------------------------------


class NoiseEmbedder(nn.Module):
    """Per-pixel Gaussian noise -> MLP -> context embedding (paper Sec. 2.2).

    Returns the per-pixel embedding e (B, npix, e_dim) and the globally
    pooled embedding g (B, e_dim); g conditions the spectral filters.
    """

    def __init__(self, noise_dim: int, embed_dim: int, npix: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, z: torch.Tensor):
        e = self.mlp(z)
        g = e.mean(dim=1)
        return e, g


class ConditionalLayerNorm(nn.Module):
    """LayerNorm whose scale/shift are functions of a per-pixel context."""

    def __init__(self, dim: int, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(embed_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.proj(e).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Compressed spectral filter (modification 2, Sec. A.2)
# ---------------------------------------------------------------------------


class CompressedSpectralFilter(nn.Module):
    """out[l, m, :] = sum_i h_i(l) * (kappa_i @ c[l, m, :])

    kappa_i are k shared learnable channel-mixing matrices; h(l) comes from
    an MLP over [sinusoidal_degree_encoding(l) | global noise embedding g],
    so the filter depends on the stochastic member through g.  This gives a
    ~L/k parameter reduction (paper: 12x at 1 degree with k=15, L=179).
    """

    def __init__(
        self,
        cin: int,
        cout: int,
        lmax: int,
        k: int,
        embed_dim: int,
        degree_freqs: int = 16,
    ):
        super().__init__()
        self.k, self.lmax = k, lmax
        self.kappa = nn.Parameter(torch.randn(k, cout, cin) * (0.02 / math.sqrt(cin)))
        freqs = torch.pow(10000.0, -torch.arange(degree_freqs) / degree_freqs)
        self.degree_freqs = degree_freqs
        self.register_buffer("freqs", freqs)  # (degree_freqs,)
        self.deg_mlp = nn.Sequential(
            nn.Linear(2 * degree_freqs + embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, k),
        )
        # degrees 0..lmax, normalized to [0, 1]
        self.register_buffer("degrees", torch.arange(lmax + 1, dtype=torch.float32))

    def degree_weights(self, g: torch.Tensor) -> torch.Tensor:
        """(B, k, L) weights over the k shared filters."""
        B = g.shape[0]
        l = self.degrees.view(1, -1, 1) * self.freqs.view(1, 1, -1)  # (1,L,F)
        enc = torch.cat([torch.sin(l), torch.cos(l)], dim=-1)  # (1,L,2F)
        enc = enc.expand(B, -1, -1)
        inp = torch.cat([enc, g.unsqueeze(1).expand(B, self.lmax + 1, -1)], dim=-1)
        return self.deg_mlp(inp).transpose(1, 2)  # (B,k,L)

    def forward(self, c: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # c: (B, L, M, C_in) complex
        re = torch.einsum("koi,blmi->kblmo", self.kappa, c.real)
        im = torch.einsum("koi,blmi->kblmo", self.kappa, c.imag)
        w = self.degree_weights(g)  # (B,k,L)
        out_re = torch.einsum("bkl,kblmo->blmo", w, re)
        out_im = torch.einsum("bkl,kblmo->blmo", w, im)
        return torch.complex(out_re, out_im)


# ---------------------------------------------------------------------------
# Full SFNO block with the Huracan modifications
# ---------------------------------------------------------------------------


class HuracanSFNOBlock(nn.Module):
    """x + ISHT(Filter(SHT(x))) followed by x + Swin(x), both pre-conditioned
    with conditional layer norm on the noise embedding.

    The spherical-harmonic transform is *not* owned by the block: it is a
    model-level resource passed through ``forward``, so all blocks share one
    instance (single copy of the basis matrices in memory and in the
    checkpoint) and the backend is swappable via injection.
    """

    def __init__(
        self,
        dim: int,
        grid: SphericalGrid,
        k_filters: int,
        window: tuple[int, int],
        heads: int,
        embed_dim: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.filter = CompressedSpectralFilter(dim, dim, grid.lmax, k_filters, embed_dim)
        self.spec_norm = ConditionalLayerNorm(dim, embed_dim)
        self.mixer_norm = ConditionalLayerNorm(dim, embed_dim)
        self.mixer = SwinLatLonBlock(
            dim, grid.nlat, grid.nlon, window, heads, mlp_ratio
        )

    def forward(
        self, x: torch.Tensor, e: torch.Tensor, g: torch.Tensor, sht: SHTBackend
    ):
        h = self.spec_norm(x, e)
        c = sht(h)
        c = self.filter(c, g)
        x = x + sht.inverse(c)
        x = x + self.mixer(self.mixer_norm(x, e))
        return x


def _run_block(block, x, e, g, sht):
    return block(x, e, g, sht)
