"""Parse IGRA radiosonde + ISD-lite surface observations and grid them into
the assimilation model's observation channels (paper Sec. 2.1, Table 2):

  upper-air  z,t,u,v,q at the 13 standard pressure levels  (from IGRA)
  surface    2t,2d,10u,10v,msl                             (from ISD-lite)

Observations are assigned to the nearest HEALPix pixel, averaged when
multiple reports land on the same pixel, normalized with the state-variable
statistics and left as exact zeros where nothing is observed.
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
    """Snap an observation datetime to the closest 6-hourly frame within 3h."""
    import datetime as dt

    hour = t_obs.hour + t_obs.minute / 60.0
    slot = int(round(hour / 6.0)) % 4 * 6
    delta = abs(hour - slot)
    day = t_obs + dt.timedelta(days=1) if hour >= 22.5 else t_obs
    key = day.replace(hour=slot % 24, minute=0, second=0, microsecond=0)
    if delta > 3.0 or key not in time_idx:
        return None
    return time_idx[key]


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
        i += 1
        prof = {}
        for _ in range(numlev):
            if i >= len(lines):
                break
            row = lines[i]
            i += 1
            try:
                p = int(row[9:15]) / 100.0  # hPa
                gph = int(row[16:21])  # m
                temp = int(row[22:27])  # 0.1 C
                rh = int(row[28:33])  # %
                wdir = int(row[40:45])  # deg
                wspd = int(row[46:51]) / 10.0  # m/s
            except ValueError:
                continue
            if p <= 0:
                continue
            lev = min(levels, key=lambda L: abs(L - p))
            if abs(lev - p) > 2.0 or lev in prof:
                continue
            v = {}
            if gph != -9999:
                v["z"] = gph * 9.80665
            if temp != -9999:
                v["t"] = temp / 10.0 + 273.15
            if rh != -9999 and temp != -9999:
                v["q"] = rh_to_specific_humidity(rh, temp / 10.0, p * 100.0)
            if wdir != -9999 and wspd != -9999:
                d = np.deg2rad(wdir)
                v["u"] = -wspd * np.sin(d)
                v["v"] = -wspd * np.cos(d)
            if v:
                prof[lev] = v
        if prof:
            yield dt.datetime(year, month, day, hour % 24), lat, lon, prof


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
                temp = int(line[14:20])
                dew = int(line[20:26])
                slp = int(line[26:32])
                wdir = int(line[32:38])
                wspd = int(line[38:44]) / 10.0
                tstamp = dt.datetime(year, month, day, hour)
            except ValueError:
                continue
            v = {}
            if temp != -9999:
                v["2t"] = temp / 10.0 + 273.15
            if dew != -9999:
                v["2d"] = dew / 10.0 + 273.15
            if slp != -9999:
                v["msl"] = slp * 10.0  # tenths of hPa -> Pa
            if wdir != -9999 and wspd != -9999:
                d = np.deg2rad(wdir)
                v["10u"] = -wspd * np.sin(d)
                v["10v"] = -wspd * np.cos(d)
            if v:
                yield tstamp, v


def _parse_igra_worker(args):
    """Parse one IGRA zip -> arrays (frame, pixel, chan, value)."""
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
        key = snap_to_frame(ftime, time_idx)
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
                    rows.append((key, int(pix), ci[(var, lev)], float(val)))
    return rows


def _parse_isd_worker(args):
    """Parse one ISD gz -> rows (frame, pixel, chan, value).

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
        key = snap_to_frame(ftime, time_idx)
        if key is None:
            continue
        for var, val in vals.items():
            if ("s", var) in ci:
                rows.append((key, int(pix), ci[("s", var)], float(val)))
    return rows


def build_obs_tensor(
    cfg: HuracanConfig,
    obs_dir: str | Path,
    times_path: str | Path,
    state_norm,
    out_path: str | Path,
    workers: int = 16,
):
    import datetime as dt
    import multiprocessing as mp

    obs_dir = Path(obs_dir)
    grid = grid_from_cfg(cfg.grid)
    colat, lon = grid.colat, grid.lon_flat
    v = cfg.data.variables
    upper, levels = v.upper, v.levels
    surface_obs = cfg.data.obs.surface_vars

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
    ]

    acc = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, rows in enumerate(
            pool.imap_unordered(_parse_igra_worker, igra_jobs, chunksize=4)
        ):
            for key, pix, c, val in rows:
                slot = acc.setdefault((key, pix, c), [0.0, 0])
                slot[0] += val
                slot[1] += 1
            if (i + 1) % 50 == 0:
                print(f"  igra {i + 1}/{len(igra_jobs)}", flush=True)
        for i, rows in enumerate(
            pool.imap_unordered(_parse_isd_worker, isd_jobs, chunksize=32)
        ):
            for key, pix, c, val in rows:
                slot = acc.setdefault((key, pix, c), [0.0, 0])
                slot[0] += val
                slot[1] += 1
            if (i + 1) % 1000 == 0:
                print(f"  isd {i + 1}/{len(isd_jobs)}", flush=True)

    obs = np.zeros((T, grid.npix, C), dtype=np.float32)
    for (t, p, c), (s, n) in acc.items():
        obs[t, p, c] = s / n
    # normalize ONLY where an observation exists; missing stays exactly 0
    # (paper Sec. 2.1 item 5); clip gross outliers at +/-8 std
    norm = obs_norm_mapping(cfg, state_norm)
    mask = obs != 0.0
    ci = np.nonzero(mask)[2]
    vals = obs[mask]
    obs[mask] = np.clip((vals - norm.mean[ci]) / norm.std[ci], -8.0, 8.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, obs.astype(np.float16))
    n_reports = sum(n for _, n in acc.values())
    cov = (np.abs(obs).sum(axis=2) > 0).mean(axis=0)
    print(
        f"obs tensor {obs.shape} saved; reports {n_reports}; "
        f"mean pixel coverage per frame {cov.mean():.3f}"
    )
    return out_path
