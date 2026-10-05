"""Learning-rate schedules (paper Table 3: cosine for most phases, constant
for the low-LR rollout finetunes; linear warmup in both cases)."""

from __future__ import annotations

import math


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def lr_factor(schedule: str, step: int, total: int, warmup: int) -> float:
    """'constant' still ramps during warmup, then holds 1.0 (original
    trainer behaviour, preserved bit-for-bit)."""
    if schedule == "constant" and step >= warmup:
        return 1.0
    return cosine_with_warmup(step, total, warmup)
