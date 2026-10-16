"""Temporal semantics of one model call -- the single source of truth.

Every autoregressive call ingests two stored frames ``(x_{t-1}, x_t)``
together with their concurrent forcing/obs frames and emits
``outputs_per_call`` future frames.  All call <-> frame <-> hour
arithmetic lives in :class:`CallSemantics`; no other module may
hard-code these numbers.

Two conventions:

* **paper (default)** ``frame_hours=6.0, outputs_per_call=2``: one call
  advances the trajectory by 12 h and emits two 6-hourly fields.  This
  is fixed by the paper's "6-step rollout = 3 days" and "8-step
  assimilation rollout = 4 days" (guide Sec. 1.2).
* **single-output variant** ``outputs_per_call=1``: the mainstream
  GenCast-style convention -- one call advances one frame (6 h for the
  atmosphere) and the sliding pair moves by one frame per call.

Non-calendar datasets (kolmogorov / jhtdb) have no hours:
they express experiment lengths directly in AR calls via
``eval.spinup_calls_override`` / ``lead_calls_override`` and only use
the frame-counting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallSemantics:
    """How autoregressive calls map to stored frames and calendar hours."""

    frame_hours: float = 6.0  # spacing between stored frames          [PAPER]
    outputs_per_call: int = 2  # fields emitted per call               [PAPER]

    def __post_init__(self):
        if self.frame_hours <= 0:
            raise ValueError(f"frame_hours must be positive, got {self.frame_hours}")
        if not isinstance(self.outputs_per_call, int) or self.outputs_per_call < 1:
            raise ValueError(
                f"outputs_per_call must be a positive int, got {self.outputs_per_call!r}"
            )

    # -- hours ---------------------------------------------------------------
    @property
    def advance_hours(self) -> float:
        """Trajectory advance per model call."""
        return self.frame_hours * self.outputs_per_call

    def calls_for_hours(self, hours: float) -> int:
        """Number of autoregressive calls covering ``hours``."""
        n = hours / self.advance_hours
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"{hours}h is not a multiple of the {self.advance_hours}h AR advance"
            )
        return int(round(n))

    def outputs_for_hours(self, hours: float) -> int:
        """Number of output fields covering ``hours``."""
        return self.calls_for_hours(hours) * self.outputs_per_call

    # -- frames (dataset-agnostic) --------------------------------------------
    def frames_for_calls(self, calls: int) -> int:
        """Stored output frames covered by ``calls`` calls."""
        return self.outputs_per_call * int(calls)

    def calls_for_frames(self, frames: int) -> int:
        """AR calls needed to cover ``frames`` output frames."""
        n, r = divmod(int(frames), self.outputs_per_call)
        if r:
            raise ValueError(
                f"{frames} frames is not a multiple of {self.outputs_per_call}"
            )
        return n

    def window_frames(self, calls: int) -> int:
        """Input frames an unroll of ``calls`` calls consumes: the leading
        ``(x_{t-1}, x_t)`` pair plus one frame per emitted output."""
        return 2 + self.outputs_per_call * int(calls)

    def obs_frames(self, calls: int) -> int:
        """Observation frames needed by the first ``calls`` calls (the
        frames concurrent with their input pairs).  Call ``j`` reads frames
        ``opc*j, opc*j+1``, so the deepest read is ``opc*(calls-1)+1``:
        2*calls for the paper's opc=2, calls+1 for opc=1.  Zero calls
        (forecast-only runs) need no observations."""
        if calls < 0:
            raise ValueError(f"calls must be >= 0, got {calls}")
        if calls == 0:
            return 0
        return 2 + self.outputs_per_call * (int(calls) - 1)

    def input_frames(self, j: int) -> tuple[int, int]:
        """Frame indices entering call ``j``: the sliding ``(x_prev, x_cur)``
        pair and their concurrent forcing/obs frames."""
        lo = self.outputs_per_call * int(j)
        return lo, lo + 1
