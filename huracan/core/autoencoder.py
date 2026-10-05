"""MLP autoencoders compressing high-dimensional observation spectra.

Paper Sec. A.1: hyperspectral IR (16,921 channels -> 32) and GNSS-RO
bending-angle profiles (5,994 features -> 64) are each compressed by a
4-layer-encoder / 4-layer-decoder MLP autoencoder trained with L1
reconstruction loss; missing channels are set to -5 after normalization
and excluded from the loss.

Satellite radiances are not part of the pilot reproduction (registration-
gated data), but this module is complete and smoke-tested on synthetic
spectra; plug real data in via the same .npz interface when available.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObsAutoencoder(nn.Module):
    def __init__(
        self,
        n_in: int,
        bottleneck: int,
        hidden: int = 1024,
        depth: int = 4,
        missing_value: float = -5.0,
    ):
        super().__init__()
        self.missing_value = missing_value
        dims = [n_in] + [hidden] * (depth - 1) + [bottleneck]
        enc = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.GELU()]
        self.encoder = nn.Sequential(*enc[:-1])
        dec_dims = [bottleneck] + [hidden] * (depth - 1) + [n_in]
        dec = []
        for a, b in zip(dec_dims[:-1], dec_dims[1:]):
            dec += [nn.Linear(a, b), nn.GELU()]
        self.decoder = nn.Sequential(*dec[:-1])

    def forward(self, x: torch.Tensor):
        mask = (x != self.missing_value).float()
        z = self.encoder(x * mask)
        rec = self.decoder(z)
        loss = (torch.abs(rec - x) * mask).sum() / mask.sum().clamp(min=1.0)
        return z, rec, loss
