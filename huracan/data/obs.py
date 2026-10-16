"""Parse IGRA radiosonde + ISD-lite surface observations and grid them into
the assimilation model's observation channels (paper Sec. 2.1, Table 2):

  upper-air  z,t,u,v,q at the 13 standard pressure levels  (from IGRA)
  surface    2t,2d,10u,10v,msl                             (from ISD-lite)

P0 conventions (RESEARCH_PLAN v9 / 4090 plan Sec. 2):

* Frame assignment is a **forward window**: an observation at time tau is
  assigned to the unique 6-hourly frame t with tau in (t-6h, t]; the raw
  time offset (tau - t) in (-6h, 0] is preserved per (frame, pixel,
  channel) at 0.25 h resolution in ``obs_offsets.npy``.  22:00/23:00
  reports therefore land on the *next* day's 00:00 frame instead of being
  dropped (the old nearest-slot logic discarded them).
* Missing data is decided from the **raw integer sentinel (-9999) before
  any unit conversion**.  The old code divided wind speed by 10 first, so
  a missing -9999 became -999.9 m/s and slipped past the sentinel check.
* Validity is tracked by an explicit per-(frame, pixel, channel) count --
  never by ``value != 0``.  A genuine physical zero (e.g. a calm-station
  u=0 m/s report) is normalized like any other observation and marked
  valid; ``obs_mask.npy`` (bit-packed over channels) and ``obs_counts.npy``
  carry the validity structure that the value tensor cannot express.
* Values are gridded to the nearest grid pixel (averaged when several
  reports land on the same pixel) and normalized with the matching state
  statistics; missing entries stay exactly 0 in ``obs.npy``.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import numpy as np

from ..config.schema import HuracanConfig
from ..core.grid import grid_from_cfg, unit_vectors
from .norm import obs_norm_mapping


def _saturation_vapor_pressure_pa(t_c):
    return 611.2 * np.exp(17.62 * t_c / (243.12 + t_c))


def rh_to_specific_humidity(rh_pct, t_c, p_pa):
    e = np.clip(rh_pct, 0, 100) / 100.0 * _saturation_vapor_pressure_pa(t_c)
    eps = 0.622
    return eps * e / np.maximum(p_pa - (1 - eps) * e, 1.0)


def frame_index(times: list[str]) -> dict:
    import datetime as dt

    out = {}
    for i, s in enumerate(times):
        d = dt.datetime.fromisoformat(s)
        key = (d.year, d.month, d.day)
        out.setdefault(key, {})[d.hour] = i
    return out


def snap_to_frame(t_obs, time_idx):
    """Forward-window assignment: return (frame_index, offset_hours) for the
    unique 6-hourly frame t with t_obs in (t-6h, t], or (None, None).

    offset_hours = t_obs - t lies in (-6, 0]: an 00:00 report belongs to the
    00:00 frame (offset 0), 03:00 to the 06:00 frame (offset -3), and a
    23:00 report to the *next* day's 00:00 frame (offset -1) -- nothing is
    dropped for being "too close to midnight".  Frames missing from
    ``time_idx`` (dataset boundaries) still yield None.
    """
    import datetime as dt

    hour = t_obs.hour + t_obs.minute / 60.0 + t_obs.second / 3600.0
    slot = int(np.ceil(hour / 6.0 - 1e-9)) * 6  # 0, 6, 12, 18, 24
    day = t_obs + dt.timedelta(days=1) if slot == 24 else t_obs
    key = day.replace(hour=slot % 24, minute=0, second=0, microsecond=0)
    idx = time_idx.get(key)
    if idx is None:
        return None, None
    return idx, hour - slot  # in (-6, 0]


def parse_igra_zip(path, levels):
    """Yield (frame_time, lat, lon, {level: dict of z,t,u,v,q})."""
    import datetime as dt

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return
    with zf.open(zf.namelist()[0]) as f:
        text = io_text(f)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        head = lines[i]
        if not head.startswith("#"):
            i += 1
            continue
        try:
            year = int(head[13:17])
            month = int(head[18:20])
            day = int(head[21:23])
            hour = int(head[24:26])
            numlev = int(head[31:36])
            lat = int(head[55:62]) / 1e4
            lon = int(head[63:71]) / 1e4
        except ValueError:
            i += 1
            continue
        # IGRA header sentinels: unknown launch hour (99) and missing
        # position (-9999 -> lat/lon far outside physical range)
        if hour > 23 or abs(lat) > 90.0 or abs(lon) > 180.0:
            i += 1
            continue
        i += 1
        prof = {}
        for _ in range(numlev):
            if i >= len(lines):
                break
            row = lines[i]
            i += 1
            try:
                p_raw = int(row[9:15])  # Pa
                gph = int(row[16:21])  # m
                temp_raw = int(row[22:27])  # 0.1 C
                rh_raw = int(row[28:33])  # %
                wdir_raw = int(row[40:45])  # deg
                wspd_raw = int(row[46:51])  # 0.1 m/s
            except ValueError:
                continue
            # every sentinel check happens on the RAW integer, before any
            # unit conversion (P0 fix; the old code divided wind speed by
            # 10 first, turning a missing -9999 into a valid-looking
            # -999.9 m/s)
            if p_raw == -9999 or p_raw <= 0:
                continue
            p = p_raw / 100.0  # hPa
            lev = min(levels, key=lambda L: abs(L - p))
            if abs(lev - p) > 2.0 or lev in prof:
                continue
            v = {}
            if gph != -9999:
                v["z"] = gph * 9.80665
            if temp_raw != -9999:
                v["t"] = temp_raw / 10.0 + 273.15
            if rh_raw != -9999 and temp_raw != -9999:
                v["q"] = rh_to_specific_humidity(rh_raw, temp_raw / 10.0, p * 100.0)
            if wdir_raw != -9999 and wspd_raw != -9999:
                wspd = wspd_raw / 10.0
                d = np.deg2rad(wdir_raw)
                v["u"] = -wspd * np.sin(d)
                v["v"] = -wspd * np.cos(d)
            if v:
                prof[lev] = v
        if prof:
            yield dt.datetime(year, month, day, hour), lat, lon, prof


def io_text(fileobj):
    return fileobj.read().decode("ascii", "ignore")


def parse_isd_gz(path):
    """Yield (frame_time, station values) from an ISD-lite file.

    Fixed-width columns (0-based; 2-digit fields separated by one space):
      [0:4] YYYY, [5:7] MM, [8:10] DD, [11:13] HH,
      [14:20] temp x0.1C, [20:26] dew x0.1C, [26:32] slp x0.1hPa,
      [32:38] wind dir deg, [38:44] wind speed x0.1 m/s
    """
    import datetime as dt

    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            try:
                year = int(line[0:4])
                month = int(line[5:7])
                day = int(line[8:10])
                hour = int(line[11:13])
                temp_raw = int(line[14:20])  # 0.1 C
                dew_raw = int(line[20:26])  # 0.1 C
                slp_raw = int(line[26:32])  # 0.1 hPa
                wdir_raw = int(line[32:38])  # deg
                wspd_raw = int(line[38:44])  # 0.1 m/s
                tstamp = dt.datetime(year, month, day, hour)
            except ValueError:
                continue
            # sentinel checks on RAW integers before any unit conversion
            # (P0 fix: the old code divided wind speed by 10 first, so a
            # missing -9999 became -999.9 m/s and produced huge u/10v
            # "observations" -- ~5% of ISD rows in a real-data sample)
            v = {}
            if temp_raw != -9999:
                v["2t"] = temp_raw / 10.0 + 273.15
            if dew_raw != -9999:
                v["2d"] = dew_raw / 10.0 + 273.15
            if slp_raw != -9999:
                v["msl"] = slp_raw * 10.0  # tenths of hPa -> Pa
            if wdir_raw != -9999 and wspd_raw != -9999:
                wspd = wspd_raw / 10.0
                d = np.deg2rad(wdir_raw)
                v["10u"] = -wspd * np.sin(d)
                v["10v"] = -wspd * np.cos(d)
            if v:
                yield tstamp, v


def _parse_igra_worker(args):
    """Parse one IGRA zip -> rows (frame, pixel, chan, value, offset_h)."""
    zpath, colat, lon, times_path, levels, upper, surface_obs = args
    import datetime as dt
    import json
    import zipfile
    from scipy.spatial import cKDTree

    tree = cKDTree(unit_vectors(colat, lon))
    time_idx = {}
    with open(times_path) as f:
        for i, s in enumerate(json.load(f)):
            d = dt.datetime.fromisoformat(s)
            time_idx[d.replace(minute=0, second=0, microsecond=0)] = i
    ci = {}
    k = 0
    for lev in levels:
        for u in upper:
            ci[(u, lev)] = k
            k += 1
    for s in surface_obs:
        ci[("s", s)] = k
        k += 1
    rows = []
    for ftime, lat, lon_, prof in parse_igra_zip(zpath, levels):
        key, off = snap_to_frame(ftime, time_idx)
        if key is None:
            continue
        _, pix = tree.query(
            unit_vectors(
                np.array([np.pi / 2 - np.deg2rad(lat)]), np.array([np.deg2rad(lon_)])
            )[0],
            workers=1,
        )
        for lev, vals in prof.items():
            for var, val in vals.items():
                if (var, lev) in ci:
                    rows.append((key, int(pix), ci[(var, lev)], float(val), float(off)))
    return rows


def _parse_isd_worker(args):
    """Parse one ISD gz -> rows (frame, pixel, chan, value, offset_h).

    Surface channels sit *after* the upper-air block; chan is offset by
    len(levels) * len(upper)."""
    gz, pix, times_path, surface_obs, surf_offset = args
    import datetime as dt
    import json

    time_idx = {}
    with open(times_path) as f:
        for i, s in enumerate(json.load(f)):
            d = dt.datetime.fromisoformat(s)
            time_idx[d.replace(minute=0, second=0, microsecond=0)] = i
    ci = {("s", s): surf_offset + i for i, s in enumerate(surface_obs)}
    rows = []
    for ftime, vals in parse_isd_gz(gz):
        key, off = snap_to_frame(ftime, time_idx)
        if key is None:
            continue
        for var, val in vals.items():
            if ("s", var) in ci:
                rows.append((key, int(pix), ci[("s", var)], float(val), float(off)))
    return rows


def build_obs_tensor(
    cfg: HuracanConfig,
    obs_dir: str | Path,
    times_path: str | Path,
    state_norm,
    out_path: str | Path,
    workers: int = 16,
    exclude_igra: set | None = None,
    exclude_isd: set | None = None,
):
    """Grid parsed observations and write the value/validity bundle.

    Outputs (all next to ``out_path``):
      obs.npy          (T, npix, C) f16  normalized values, missing == 0
      obs_mask.npy     (T, npix, C/8) u8 bit-packed validity (bit c of byte
                       c//8, low bit first -- see np.packbits(axis=-1))
      obs_counts.npy   (T, npix) u8     number of valid channels (0..C)
      obs_offsets.npy  (T, npix, C) i8  mean report-time offset from the
                       frame, 0.25 h units, in [-24, 0]; 127 = no data
      obs_audit.json   per-channel valid counts + physical-zero counts

    ``exclude_igra`` / ``exclude_isd``: station-id sets to leave out entirely
    (used for the station-holdout diagnostics; defaults: none excluded).
    """
    import datetime as dt
    import multiprocessing as mp

    obs_dir = Path(obs_dir)
    grid = grid_from_cfg(cfg.grid)
    colat, lon = grid.colat, grid.lon_flat
    v = cfg.data.variables
    upper, levels = v.upper, v.levels
    surface_obs = cfg.data.obs.surface_vars
    exclude_igra = exclude_igra or set()
    exclude_isd = exclude_isd or set()

    with open(times_path) as f:
        times = json.load(f)
    T = len(times)
    C = cfg.n_obs_channels()

    # station -> pixel map for ISD
    import pandas as pd
    from scipy.spatial import cKDTree

    tree = cKDTree(unit_vectors(colat, lon))
    sdf = pd.read_csv(obs_dir / "isd_stations.csv")
    stations = {}
    for r in sdf.itertuples():
        _, pix = tree.query(
            unit_vectors(
                np.array([np.pi / 2 - np.deg2rad(r.lat)]), np.array([np.deg2rad(r.lon)])
            )[0],
            workers=1,
        )
        stations[r.sid] = int(pix)

    igra_jobs = [
        (
            str(z),
            colat,
            lon,
            str(times_path),
            levels,
            upper,
            surface_obs,
        )
        for z in sorted((obs_dir / "igra").glob("*.zip"))
        if z.stem not in exclude_igra
    ]
    surf_offset = len(levels) * len(upper)
    isd_jobs = [
        (
            str(g),
            stations[g.name.rsplit("-", 1)[0]],
            str(times_path),
            surface_obs,
            surf_offset,
        )
        for g in sorted((obs_dir / "isd").glob("*.gz"))
        if g.name.rsplit("-", 1)[0] in stations
        and g.name.rsplit("-", 1)[0] not in exclude_isd
    ]
    n_excluded = (
        len(list((obs_dir / "igra").glob("*.zip"))) - len(igra_jobs),
        len(list((obs_dir / "isd").glob("*.gz"))) - len(isd_jobs),
    )

    # acc[(frame, pixel, chan)] = [value_sum, count, offset_sum, phys_zero]
    acc = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, rows in enumerate(
            pool.imap_unordered(_parse_igra_worker, igra_jobs, chunksize=4)
        ):
            for key, pix, c, val, off in rows:
                slot = acc.setdefault((key, pix, c), [0.0, 0, 0.0, 0])
                slot[0] += val
                slot[1] += 1
                slot[2] += off
                slot[3] += val == 0.0
            if (i + 1) % 50 == 0:
                print(f"  igra {i + 1}/{len(igra_jobs)}", flush=True)
        for i, rows in enumerate(
            pool.imap_unordered(_parse_isd_worker, isd_jobs, chunksize=32)
        ):
            for key, pix, c, val, off in rows:
                slot = acc.setdefault((key, pix, c), [0.0, 0, 0.0, 0])
                slot[0] += val
                slot[1] += 1
                slot[2] += off
                slot[3] += val == 0.0
            if (i + 1) % 1000 == 0:
                print(f"  isd {i + 1}/{len(isd_jobs)}", flush=True)

    obs = np.zeros((T, grid.npix, C), dtype=np.float32)
    mask = np.zeros((T, grid.npix, C), dtype=bool)
    offsets = np.full((T, grid.npix, C), 127, dtype=np.int8)
    chan_reports = np.zeros(C, dtype=np.int64)
    chan_phys_zero = np.zeros(C, dtype=np.int64)
    norm = obs_norm_mapping(cfg, state_norm)
    for (t, p, c), (s, n, so, nz) in acc.items():
        val = s / n
        mask[t, p, c] = True
        # normalize EVERY valid report, including physical zeros (P0 fix:
        # validity comes from the report count, never from value != 0)
        obs[t, p, c] = np.clip((val - norm.mean[c]) / norm.std[c], -8.0, 8.0)
        offsets[t, p, c] = int(np.clip(round(so / n / 0.25), -24, 0))
        chan_reports[c] += n
        chan_phys_zero[c] += nz

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, obs.astype(np.float16))
    np.save(out_path.with_name("obs_mask.npy"), np.packbits(mask, axis=-1))
    np.save(
        out_path.with_name("obs_counts.npy"),
        mask.sum(axis=-1).astype(np.uint8),
    )
    np.save(out_path.with_name("obs_offsets.npy"), offsets)

    # ---- audit bundle ------------------------------------------------
    obs_names = norm.names
    per_year = {}
    years = sorted({dt.datetime.fromisoformat(times[t]).year for t, _, _ in acc})
    for y in years:
        per_year[y] = int(sum(n for (t, _, _), (s, n, _, _) in acc.items()
                              if dt.datetime.fromisoformat(times[t]).year == y))
    audit = {
        "n_reports_total": int(sum(n for _, (_, n, _, _) in acc.items())),
        "n_grid_slots": len(acc),
        "excluded": {"igra": n_excluded[0], "isd": n_excluded[1]},
        "per_channel": {
            obs_names[c]: {
                "reports": int(chan_reports[c]),
                "physical_zeros": int(chan_phys_zero[c]),
            }
            for c in range(C)
        },
        "per_year_reports": per_year,
    }
    with open(out_path.with_name("obs_audit.json"), "w") as f:
        json.dump(audit, f, indent=1)

    cov = mask.any(axis=-1).mean(axis=0)
    print(
        f"obs tensor {obs.shape} saved; reports {audit['n_reports_total']}; "
        f"mean pixel coverage per frame {cov.mean():.3f}; "
        f"physical-zero reports kept: {int(chan_phys_zero.sum())}"
    )
    return out_path
