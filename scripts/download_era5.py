#!/usr/bin/env python3
"""Download ERA5 subsets from the Copernicus CDS (new 2024+ API).

Requests 6-hourly fields at 1-degree grid (regridded server-side) to keep
download volumes tractable; the pilot needs ~1.2 GB per year.

Usage:
  python scripts/download_era5.py --years 2010 2011 2012 2013
  python scripts/download_era5.py --test           # one month, verifies setup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cdsapi

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = ["00:00", "06:00", "12:00", "18:00"]

UPPER = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

SINGLE = {
    "2m_temperature": "2t",
    "2m_dewpoint_temperature": "2d",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "total_cloud_cover": "tcc",
    "total_column_water_vapour": "tcwv",
    "mean_sea_level_pressure": "msl",
}


def retrieve_year(client, year, outdir):
    req_common = dict(
        product_type=["reanalysis"],
        year=[str(year)],
        month=MONTHS,
        day=DAYS,
        time=TIMES,
        data_format="grib",
        download_format="unarchived",
        grid=[1.0, 1.0],
    )
    f_pl = outdir / f"era5_pl_{year}.grib"
    if not f_pl.exists():
        print(f"requesting pressure levels {year} ...", flush=True)
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                **req_common,
                "variable": list(UPPER),
                "pressure_level": [str(l) for l in LEVELS],
            },
            str(f_pl),
        )
    f_sl = outdir / f"era5_sl_{year}.grib"
    if not f_sl.exists():
        print(f"requesting single levels {year} ...", flush=True)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {**req_common, "variable": list(SINGLE)},
            str(f_sl),
        )
    return f_pl, f_sl


def retrieve_static(client, outdir):
    f_st = outdir / "era5_static.grib"
    if not f_st.exists():
        print("requesting static fields ...", flush=True)
        client.retrieve(
            "reanalysis-era5-single-levels",
            dict(
                variable=["geopotential", "land_sea_mask"],
                date="2010-01-01",
                time="00:00",
                data_format="grib",
                download_format="unarchived",
            ),
            str(f_st),
        )
    return f_st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2010, 2011, 2012, 2013])
    ap.add_argument("--outdir", default="data/raw")
    ap.add_argument(
        "--test",
        action="store_true",
        help="single month of 2010 to verify the API setup",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=False)

    if args.test:
        f = outdir / "era5_test_201001.grib"
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            dict(
                product_type=["reanalysis"],
                variable=["temperature"],
                pressure_level=["500"],
                year=["2010"],
                month=["01"],
                day=DAYS,
                time=TIMES,
                data_format="grib",
                download_format="unarchived",
                grid=[1.0, 1.0],
            ),
            str(f),
        )
        print(f"test file written: {f}")
        return

    retrieve_static(client, outdir)
    for y in args.years:
        retrieve_year(client, y, outdir)
    print("all downloads complete")


if __name__ == "__main__":
    main()
