#!/usr/bin/env python3
"""Download freely-available in-situ observations used by the DA component:

  * IGRA v2 radiosonde soundings (upper-air z,t,u,v,q at std pressure levels)
        https://www.ncei.noaa.gov/pub/data/igra/data/data-por/
  * NOAA ISD-lite land stations (surface 2t,2d,10u,10v,msl)
        https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/

Usage:
  python scripts/download_obs.py --years 2010 2011 2012 2013 \
      --n-igra 400 --n-isd 1500 --outdir data/raw_obs
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

IGRA_LIST = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
IGRA_POR = "https://www.ncei.noaa.gov/pub/data/igra/data/data-por/{sid}-data.txt.zip"
ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_FILE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{sid}-{year}.gz"


def fetch(url: str, tries: int = 3) -> bytes | None:
    for _ in range(tries):
        try:
            with urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            continue
    return None


def igra_stations(years):
    raw = fetch(IGRA_LIST).decode("ascii", "ignore").splitlines()
    rows = []
    for line in raw:
        sid = line[0:11].strip()
        try:
            lat = float(line[12:20])
            lon = float(line[21:30])
            first, last = int(line[72:76]), int(line[77:81])
        except ValueError:
            continue
        if (
            -90 <= lat <= 90
            and -180 <= lon < 180
            and last >= min(years)
            and first <= max(years)
        ):
            rows.append((sid, lat, lon))
    return pd.DataFrame(rows, columns=["sid", "lat", "lon"])


def isd_stations(years):
    df = pd.read_csv(ISD_HISTORY, dtype={"USAF": str, "WBAN": str})
    df = df.dropna(subset=["LAT", "LON", "USAF", "WBAN"])
    # some USAF codes are alphanumeric; keep numeric ones (fixed stations)
    df["USAF"] = pd.to_numeric(df["USAF"], errors="coerce")
    df["WBAN"] = pd.to_numeric(df["WBAN"], errors="coerce")
    df = df.dropna(subset=["USAF", "WBAN"])
    df["USAF"] = df["USAF"].astype(int).astype(str).str.zfill(6)
    df["WBAN"] = df["WBAN"].astype(int).astype(str).str.zfill(5)
    df["BEGIN"] = pd.to_datetime(df["BEGIN"], format="%Y%m%d")
    df["END"] = pd.to_datetime(df["END"], format="%Y%m%d", errors="coerce")
    df["END"] = df["END"].fillna(pd.Timestamp.max)
    ok = (df["BEGIN"] <= f"{max(years)}-12-31") & (df["END"] >= f"{min(years)}-01-01")
    df = df[ok]
    df["sid"] = df["USAF"] + "-" + df["WBAN"]
    return (
        df[["sid", "LAT", "LON"]]
        .rename(columns={"LAT": "lat", "LON": "lon"})
        .reset_index(drop=True)
    )


def download_all(urls_and_paths, n_threads=32):
    def job(url, path):
        if path.exists():
            return True
        data = fetch(url)
        if data is None:
            return False
        path.write_bytes(data)
        return True

    ok = fail = 0
    with ThreadPoolExecutor(n_threads) as ex:
        futs = [ex.submit(job, u, p) for u, p in urls_and_paths]
        for i, f in enumerate(as_completed(futs)):
            ok += f.result()
            fail += not f.result()
            if (i + 1) % 100 == 0:
                print(
                    f"  {i + 1}/{len(urls_and_paths)} (ok {ok}, failed {fail})",
                    flush=True,
                )
    print(f"downloaded {ok}, failed {fail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2010, 2011, 2012, 2013])
    ap.add_argument("--outdir", default="data/raw_obs")
    ap.add_argument("--n-igra", type=int, default=400)
    ap.add_argument("--n-isd", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "igra").mkdir(parents=True, exist_ok=True)
    (out / "isd").mkdir(parents=True, exist_ok=True)

    igra = igra_stations(args.years).sample(
        min(args.n_igra, 10**9), random_state=args.seed
    )
    print(f"IGRA stations: {len(igra)}")
    urls = [
        (IGRA_POR.format(sid=r.sid), out / "igra" / f"{r.sid}.zip")
        for r in igra.itertuples()
    ]
    download_all(urls)

    isd = isd_stations(args.years).sample(
        min(args.n_isd, 10**9), random_state=args.seed
    )
    print(f"ISD stations: {len(isd)}")
    urls = []
    for y in args.years:
        urls += [
            (ISD_FILE.format(year=y, sid=r.sid), out / "isd" / f"{r.sid}-{y}.gz")
            for r in isd.itertuples()
        ]
    download_all(urls)

    igra.to_csv(out / "igra_stations.csv", index=False)
    isd.to_csv(out / "isd_stations.csv", index=False)


if __name__ == "__main__":
    main()
