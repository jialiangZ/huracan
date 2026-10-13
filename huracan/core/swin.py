"""Swin Transformer block on the regular lat-lon grid (paper modification 1).

The paper replaces one MLP inside each SFNO block with a Swin Transformer
block but does not specify window sizes, shifts or padding (all [REPO],
guide Sec. 8.2).  This implementation works on (nlat, nlon) maps viewed from
the flat (npix,) layout:

  * W-MSA: non-overlapping (wh x ww) windows;
  * SW-MSA: shifted by (wh//2, ww//2) -- longitude is *periodic*, so the
    shift is a true wrap-around (torch.roll, no attention mask needed and
    no artificial seam); latitude is not periodic, so it is zero-padded by
    the shift amount and cropped afterwards (padded rows attend like a
    learned "null" token -- a documented simplification of masked Swin);
  * one shared attention module serves both stages (as in the previous
    HEALPix-face implementation); a single MLP follows, pre-norm residuals.

Windows must divide the grid: nlat % wh == 0, nlon % ww == 0, and both
window sides must be even so that 2*(window//2) keeps the shifted map
window-divisible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention2D(nn.Module):
    """Multi-head attention over (wh*ww) tokens of one 2D window, with the
    standard 2D relative-position bias table."""

    def __init__(self, dim: int, wh: int, ww: int, heads: int):
        super().__init__()
        self.dim, self.wh, self.ww, self.heads = dim, wh, ww, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * wh - 1, 2 * ww - 1))
        nn.init.trunc_normal_(self.rel, std=0.02)
        ys = torch.arange(wh)
        xs = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=0).view(2, -1)
        rel_idx = torch.stack(
            [
                coords[0, :, None] - coords[0, None, :] + wh - 1,
                coords[1, :, None] - coords[1, None, :] + ww - 1,
            ]
        )  # (2, n, n)
        self.register_buffer("rel_idx", rel_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (nW, wh*ww, C)
        nw, n, C = x.shape
        H = self.heads
        qkv = self.qkv(x).reshape(nw, n, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (nw,H,n,n)
        bias = self.rel[:, self.rel_idx[0], self.rel_idx[1]]  # (H,n,n)
        attn = attn + bias.unsqueeze(0)
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(nw, n, C)
        return self.proj(x)


class SwinLatLonBlock(nn.Module):
    """One Swin block on the (nlat, nlon) map; input/output (B, npix, C)."""

    def __init__(
        self,
        dim: int,
        nlat: int,
        nlon: int,
        window: tuple[int, int],
        heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        wh, ww = window
        if wh % 2 or ww % 2:
            raise ValueError(f"window sides must be even, got {window}")
        if nlat % wh or nlon % ww:
            raise ValueError(f"window {window} must divide the grid ({nlat} x {nlon})")
        self.nlat, self.nlon = nlat, nlon
        self.wh, self.ww = wh, ww
        self.shift = (wh // 2, ww // 2)

        self.norm1 = nn.LayerNorm(dim)
        self.norm1_shift = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(dim, wh, ww, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim),
        )

    # -- windowing ----------------------------------------------------------
    def _windows(self, x2d: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B*nH*nW, wh*ww, C)
        B, H, W, C = x2d.shape
        wh, ww = self.wh, self.ww
        x = x2d.view(B, H // wh, wh, W // ww, ww, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh * ww, C)

    def _unwindows(
        self, xw: torch.Tensor, B: int, H: int, W: int, C: int
    ) -> torch.Tensor:
        wh, ww = self.wh, self.ww
        x = xw.view(B, H // wh, W // ww, wh, ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x2d = x.view(B, self.nlat, self.nlon, C)

        # W-MSA
        h = self.norm1(x2d)
        xw = self.attn(self._windows(h))
        x2d = x2d + self._unwindows(xw, B, self.nlat, self.nlon, C)

        # SW-MSA: zero-pad latitude, wrap longitude
        sh, sw = self.shift
        Hp = self.nlat + 2 * sh
        # pad spec is last-dim-first: C (0,0), lon (0,0), lat (sh, sh)
        xs = F.pad(x2d, (0, 0, 0, 0, sh, sh))
        xs = torch.roll(xs, shifts=(sw,), dims=(2,))
        h = self.norm1_shift(xs)
        xw = self.attn(self._windows(h))
        xs = xs + self._unwindows(xw, B, Hp, self.nlon, C)
        xs = torch.roll(xs, shifts=(-sw,), dims=(2,))
        xs = xs[:, sh : sh + self.nlat]

        # MLP
        out = xs + self.mlp(self.norm2(xs))
        return out.reshape(B, N, C)
