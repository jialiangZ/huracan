"""Channel accounting (guide P0: never silently "fix" the 269/334 mismatch).

The paper's main text says the assimilation encoder receives 269 observation
channels per time step, while Table 2 entries sum to 334 = 269 + 65
(upper-air in-situ) -- an unresolved discrepancy.  This module prints the
*configured* channel layout next to the paper reference numbers so any gap
is visible at startup instead of being papered over.
"""

from __future__ import annotations

from ..config.schema import HuracanConfig

PAPER_STATE_CHANNELS = 72
PAPER_FORCING_CHANNELS = 3
PAPER_OBS_TEXT = 269
PAPER_OBS_TABLE2_SUM = 334


def channel_report(cfg: HuracanConfig) -> str:
    v = cfg.data.variables
    n_upper = len(v.upper) * len(v.levels)
    n_state = cfg.n_prognostic()
    n_forc = cfg.n_forcing()
    n_obs = cfg.n_obs_channels()
    fcst_in = 2 * (n_state + n_forc)
    assim_in = fcst_in + 2 * n_obs
    lines = [
        "channel accounting (configured vs paper)",
        "  state      "
        f"{n_state:>4}  (upper-air {n_upper} = {len(v.upper)} vars x {len(v.levels)} levels"
        f" + surface {len(v.surface)})   paper: {PAPER_STATE_CHANNELS}",
        "  forcing    "
        f"{n_forc:>4}  ({', '.join(cfg.data.forcing)})   paper: "
        f"{PAPER_FORCING_CHANNELS} (insol, zsfc, lsm)",
        "  obs        "
        f"{n_obs:>4}  (IGRA/ISD in-situ only)"
        f"   paper text: {PAPER_OBS_TEXT}, Table 2 sum: {PAPER_OBS_TABLE2_SUM}"
        " (unresolved)",
        "  encoder in-features per time step:",
        f"    forecast model : {n_state + n_forc}",
        f"    assim  model   : {n_state + n_forc + n_obs} (+{n_obs} obs)",
        "  flattened two-timestep input:",
        f"    forecast model : {fcst_in}",
        f"    assim  model   : {assim_in}",
    ]
    if cfg.data.obs.enabled:
        lines.append(
            "  note: satellite radiance channels (SEVIRI/ATMS/IR-AE/RO-AE)"
            " are NOT implemented yet -- guide Level-2 work"
        )
    return "\n".join(lines)
