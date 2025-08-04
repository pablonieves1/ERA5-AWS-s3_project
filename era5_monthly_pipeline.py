from __future__ import annotations

import sys, subprocess, shutil, os, calendar, argparse, json
from datetime import datetime
from pathlib import Path
from typing import Sequence, Optional, List, Dict, Tuple

import numpy as np  # needed for forecast-time collapsing

# ----------------- General configuration -----------------

SRC_BUCKET = "nsf-ncar-era5"
DST_BUCKET = "pablo-era5-results-usw2"

# Ohio bounding box (decimal degrees)
OHIO = dict(lat_min=38.4, lat_max=41.9, lon_min=-84.8, lon_max=-80.5)

# Variable specifications:
#  - cls/pid   : used to build the expected filename pattern (when applicable)
#  - canon     : canonical variable name in the final dataset (C#-friendly)
#  - prefix    : source S3 collection
VAR_SPECS: Dict[str, Dict[str, object]] = {
    "10u":  {"cls": 128, "pid": 165, "canon": "u10",  "prefix": "e5.oper.an.sfc"},
    "10v":  {"cls": 128, "pid": 166, "canon": "v10",  "prefix": "e5.oper.an.sfc"},
    "100u": {"cls": 228, "pid": 246, "canon": "u100", "prefix": "e5.oper.an.sfc"},
    "100v": {"cls": 228, "pid": 247, "canon": "v100", "prefix": "e5.oper.an.sfc"},
    "2t":   {"cls": 128, "pid": 167, "canon": "t2m",  "prefix": "e5.oper.an.sfc"},
    "sp":   {"cls": 128, "pid": 134, "canon": "sp",   "prefix": "e5.oper.an.sfc"},
    # 10 m gust (forecast min/max set; split in two chunks per month)
    "10fg": {"cls": 128, "pid": 49,  "canon": "fg10", "prefix": "e5.oper.fc.sfc.minmax"},
}

# ----------------- Date & S3 helpers -----------------

def last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]

def prev_year_month(year: int, month: int) -> tuple[int, int]:
    """Return previous month (year, month)."""
    if month == 1:
        return year - 1, 12
    return year, month - 1

def month_range(start: str, end: Optional[str]=None) -> List[Tuple[int,int]]:
    """Yield (year, month) from 'YYYYMM' start to 'YYYYMM' end inclusive.
       If end is None, use current UTC month."""
    s_y, s_m = int(start[:4]), int(start[4:])
    if end:
        e_y, e_m = int(end[:4]), int(end[4:])
    else:
        now = datetime.utcnow()
        e_y, e_m = now.year, now.month
    out: List[Tuple[int,int]] = []
    y, m = s_y, s_m
    while (y < e_y) or (y == e_y and m <= e_m):
        out.append((y, m))
        m += 1
        if m == 13:
            y += 1; m = 1
    return out

def run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def s3_exists(bucket: str, key: str) -> bool:
    try:
        run(["aws", "s3", "ls", f"s3://{bucket}/{key}"])
        return True
    except subprocess.CalledProcessError:
        return False

def s3_cp(src: str, dst: str) -> None:
    """Copy S3<->local without progress bar and with reduced output."""
    run(["aws", "s3", "cp", src, dst, "--no-progress", "--only-show-errors"])

def list_keys_for_var_month(year: int, month: int, var: str) -> List[str]:
    """
    Return the list of S3 keys for (variable, month):
      - e5.oper.an.sfc → a single monthly file.
      - e5.oper.fc.sfc.minmax (10fg) → files for the month PLUS the previous-month
        chunk that ends at {YYYYMM}0106.nc (to include 00–05Z on day 1).
    """
    spec = VAR_SPECS[var]
    prefix = spec["prefix"]  # type: ignore[assignment]
    pid = int(spec["pid"])   # type: ignore[assignment]
    yyyymm = f"{year:04d}{month:02d}"

    if prefix == "e5.oper.an.sfc":
        # monthly file (0100 → last day 23)
        cls = int(spec["cls"])  # type: ignore[assignment]
        fname = (
            f"{prefix}.{cls:03d}_{pid:03d}_{var}.ll025sc."
            f"{yyyymm}0100_{yyyymm}{last_day_of_month(year,month):02d}23.nc"
        )
        return [f"{prefix}/{yyyymm}/{fname}"]

    # For fc.sfc.minmax: list objects by prefix and filter by _{pid}_{var}
    def list_prefix_keys(mm: str) -> List[str]:
        api_prefix = f"{prefix}/{mm}/"
        proc = run(["aws","s3api","list-objects-v2","--bucket", SRC_BUCKET,"--prefix", api_prefix])
        data = json.loads(proc.stdout) if proc.stdout else {}
        contents = data.get("Contents", [])
        return [c["Key"] for c in contents if f"_{pid:03d}_{var}" in c.get("Key","")]

    keys_now = sorted(list_prefix_keys(yyyymm))

    # Include previous-month boundary file ending at {YYYYMM}0106.nc
    py, pm = prev_year_month(year, month)
    yyyymm_prev = f"{py:04d}{pm:02d}"
    keys_prev = list_prefix_keys(yyyymm_prev)
    keys_prev_boundary = [k for k in keys_prev if k.endswith(f"{yyyymm}0106.nc")]

    return sorted(keys_prev_boundary + keys_now)

# ----------------- NetCDF helpers -----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def pick_main_var(ds) -> str:
    """Choose the largest data variable (ignore scalars)."""
    candidates = []
    for name, da in ds.data_vars.items():
        size = 1
        try:
            for s in da.shape:
                size *= int(s)
        except Exception:
            pass
        candidates.append((size, name))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else list(ds.data_vars)[0]

def open_and_subset(nc_path: Path, lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    """Open a NetCDF and subset to bbox; return (ds, sub, lat_name, lon_name, time_name)."""
    import xarray as xr
    try:
        ds = xr.open_dataset(nc_path.as_posix(), engine="netcdf4")
    except Exception:
        ds = xr.open_dataset(nc_path.as_posix())

    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    time_name = "time" if "time" in ds.coords else None
    if not lat_name or not lon_name:
        raise RuntimeError("Could not find latitude/longitude coords")

    # convert 0..360 → -180..180 if needed
    try:
        if float(ds[lon_name].max()) > 180.0:
            ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
    except Exception:
        pass

    # handle latitude descending
    lat0 = float(ds[lat_name].isel({lat_name: 0}))
    lat1 = float(ds[lat_name].isel({lat_name: -1}))
    lat_slice = slice(lat_max, lat_min) if lat0 > lat1 else slice(lat_min, lat_max)

    sub = ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})
    return ds, sub, lat_name, lon_name, time_name

def collapse_fc_to_time(sub):
    """
    For forecast min/max files (10fg), collapse forecast_initial_time + forecast_hour
    into a single 'time' coordinate (datetime64[ns]). Removes forecast dims/coords.
    """
    # find coordinate names
    fi = None
    for cand in ("forecast_initial_time", "forecast_reference_time"):
        if cand in sub.coords:
            fi = cand
            break
    fh = None
    for cand in ("forecast_hour", "step", "lead_time"):
        if cand in sub.coords:
            fh = cand
            break

    if fi and fh and fi in sub.dims and fh in sub.dims:
        # 1) Get 1-D numpy arrays
        fi_vals = sub[fi].astype("datetime64[h]").values  # shape: (Nfi,)
        fh_vals = sub[fh].astype("timedelta64[h]").values # shape: (Nfh,)

        # 2) Broadcast to 2-D grid and convert to ns precision
        #    (Nfi, Nfh) datetime64[h] + timedelta64[h] -> datetime64[h] -> [ns]
        time2d_h = fi_vals[:, None] + fh_vals[None, :]
        time2d = time2d_h.astype("datetime64[ns]")

        # 3) Attach as a 2-D coord with dims (fi, fh)
        sub = sub.assign_coords({"time": ((fi, fh), time2d)})

        # 4) Stack the two forecast dims into a single 'time' dim
        sub = sub.stack(time=(fi, fh))

        # 5) Sort by time and drop duplicated timestamps (keep first)
        sub = sub.sortby("time")
        uniq_vals, uniq_idx = np.unique(sub["time"].values, return_index=True)
        sub = sub.isel(time=np.sort(uniq_idx))

        # 6) Remove leftover forecast dims/coords
        sub = sub.drop_vars([fi, fh], errors="ignore").drop_dims([fi, fh], errors="ignore")

    return sub


# ----------------- Per-month processing -----------------

def process_month(year: int, month: int, vars_list: List[str], work_root: Path, out_s3_prefix: str, verbose: bool=False) -> None:
    import xarray as xr

    yyyymm = f"{year:04d}{month:02d}"
    start_iso = f"{year:04d}-{month:02d}-01T00:00:00"
    end_iso   = f"{year:04d}-{month:02d}-{last_day_of_month(year,month):02d}T23:59:59"

    month_dir = work_root / yyyymm
    dl_dir = month_dir / "download"
    ss_dir = month_dir / "subset"
    ensure_dir(dl_dir); ensure_dir(ss_dir)

    subset_paths: Dict[str, Path] = {}

    for var in vars_list:
        spec = VAR_SPECS[var]
        canon = spec["canon"]  # type: ignore[index]
        keys = list_keys_for_var_month(year, month, var)
        if not keys:
            raise FileNotFoundError(f"No source files for {var} in {yyyymm}")

        if verbose:
            print(f"[SRC] {var} → {len(keys)} file(s) in {yyyymm}")

        sub_pieces = []
        time_name_common: Optional[str] = None

        for key in keys:
            src = f"s3://{SRC_BUCKET}/{key}"
            local_nc = dl_dir / Path(key).name
            if verbose: print(f"[DL] {src}")
            s3_cp(src, local_nc.as_posix())

            ds, sub, lat_name, lon_name, time_name = open_and_subset(
                local_nc, OHIO["lon_min"], OHIO["lon_max"], OHIO["lat_min"], OHIO["lat_max"]
            )

            # Pick main var and rename to canonical
            main = pick_main_var(sub)
            sub = sub[[main]]
            if main != canon:
                sub = sub.rename({main: canon})

            # If this is gust (forecast min/max), collapse forecast dims into 'time'
            if var == "10fg":
                sub = collapse_fc_to_time(sub)
                time_name = "time" if "time" in sub.coords else time_name

            # Strictly clip to the month time window
            if time_name and time_name in sub.coords:
                sub = sub.sel({time_name: slice(start_iso, end_iso)})

            sub_pieces.append(sub)
            if time_name and not time_name_common:
                time_name_common = time_name

            # close & free disk
            try: ds.close()
            except Exception: pass
            try: local_nc.unlink()
            except Exception: pass

        # Concatenate chunks along time (or merge if no time)
        if len(sub_pieces) == 1:
            sub_all = sub_pieces[0]
        else:
            if time_name_common and (time_name_common in sub_pieces[0].coords):
                sub_all = xr.concat(sub_pieces, dim=time_name_common).sortby(time_name_common)
            else:
                sub_all = xr.merge(sub_pieces)

        # Write per-variable subset (temporary). Keep simple intermediate.
        out_subset = ss_dir / f"{canon}_{yyyymm}.nc"
        if verbose: print(f"[WR] {out_subset.name} ({canon})")
        sub_all.to_netcdf(out_subset.as_posix(), engine="netcdf4", format="NETCDF4_CLASSIC", encoding={canon: {}})

        try: sub_all.close()
        except Exception: pass

        subset_paths[canon] = out_subset

    # --- MERGE canonical variables ---
    if verbose: print("[MERGE] combining variables")
    dsets = [xr.open_dataset(p.as_posix()) for p in subset_paths.values()]
    merged = xr.merge(dsets, compat="no_conflicts", join="exact")

    # --- Normalize coordinates: names, dtype (float32), attrs ---
    coord_map = {}
    if "lat" in merged.coords and "latitude" not in merged.coords:
        coord_map["lat"] = "latitude"
    if "lon" in merged.coords and "longitude" not in merged.coords:
        coord_map["lon"] = "longitude"
    if coord_map:
        merged = merged.rename(coord_map)

    if "latitude" in merged.coords and merged["latitude"].dtype != "float32":
        merged["latitude"] = merged["latitude"].astype("float32")
    if "longitude" in merged.coords and merged["longitude"].dtype != "float32":
        merged["longitude"] = merged["longitude"].astype("float32")

    if "latitude" in merged.coords:
        merged["latitude"].attrs.update({
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        })
    if "longitude" in merged.coords:
        merged["longitude"].attrs.update({
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        })

    # --- Drop any leftover forecast dims/coords (safety) ---
    for name in ("forecast_hour","forecast_initial_time","forecast_reference_time","step","lead_time"):
        if name in merged.dims:
            merged = merged.drop_dims(name, errors="ignore")
        if name in merged.coords:
            merged = merged.drop_vars(name, errors="ignore")

    # --- Time encoding exactly like your 'good' file ---
    if "time" in merged.coords:
        merged["time"].encoding.update({
            "units": "hours since 1900-01-01 00:00:00.0",
            "calendar": "gregorian",
        })

    # --- Ensure dim order (time, latitude, longitude) when present ---
    for v in list(merged.data_vars):
        dims = merged[v].dims
        desired = tuple([d for d in ("time","latitude","longitude") if d in dims])
        if set(dims) == set(desired) and dims != desired:
            merged[v] = merged[v].transpose(*desired)

    # --- (Optional) match some variable dtypes to float64 like your 'good' ---
    # If you want: uncomment next lines to force float64 for these variables
    # for v in ("sp","t2m","u100","v100"):
    #     if v in merged and merged[v].dtype != "float64":
    #         merged[v] = merged[v].astype("float64")

    # --- Write final monthly file as NetCDF3 (max compatibility) ---
    out_local = month_dir / f"ERA5_OH_{yyyymm}.nc"
    merged.to_netcdf(out_local.as_posix(), engine="scipy")

    # close temps
    for ds_tmp in dsets:
        try: ds_tmp.close()
        except Exception: pass
    try: merged.close()
    except Exception: pass

    # Upload to S3 and clean local
    dst_key = f"{out_s3_prefix}/{year:04d}/ERA5_OH_{yyyymm}.nc"
    if verbose: print(f"[UP] s3://{DST_BUCKET}/{dst_key}")
    s3_cp(out_local.as_posix(), f"s3://{DST_BUCKET}/{dst_key}")

    try:
        out_local.unlink()
        shutil.rmtree(ss_dir, ignore_errors=True)
        shutil.rmtree(dl_dir, ignore_errors=True)
    except Exception:
        pass

# ----------------- CLI -----------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download, slice (Ohio), merge variables (incl. 10fg collapsing), and upload monthly ERA5 NetCDFs."
    )
    ap.add_argument("--start", required=True, help="Start month YYYYMM, e.g. 199501")
    ap.add_argument("--end", help="End month YYYYMM (inclusive). If omitted, runs to current month.")
    ap.add_argument("--vars", nargs="+",
                    default=["10u","10v","100u","100v","2t","sp","10fg"],
                    help="Variables to process (default includes 10fg).")
    ap.add_argument("--workdir", default=str(Path.home() / "era5" / "work"), help="Local work dir")
    ap.add_argument("--s3-prefix", default="ohio-monthly", help="Dest S3 prefix inside your bucket")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Show plan without doing work")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    work_root = Path(args.workdir)
    ensure_dir(work_root)

    months = month_range(args.start, args.end)
    for (y, m) in months:
        yyyymm = f"{y:04d}{m:02d}"
        dst_key = f"{args.s3_prefix}/{y:04d}/ERA5_OH_{yyyymm}.nc"

        if s3_exists(DST_BUCKET, dst_key):
            if args.verbose: print(f"[SKIP] exists in S3: {dst_key}")
            continue

        if args.dry_run:
            print(f"[PLAN] {yyyymm} -> s3://{DST_BUCKET}/{dst_key}")
            for v in args.vars:
                keys = list_keys_for_var_month(y, m, v)
                for k in keys:
                    print(f"       will fetch s3://{SRC_BUCKET}/{k}")
            continue

        if args.verbose: print(f"[DO] {yyyymm}")
        try:
            process_month(y, m, args.vars, work_root, args.s3_prefix, verbose=args.verbose)
        except subprocess.CalledProcessError as cpe:
            print(f"[ERR] AWS CLI failed for {yyyymm}: {cpe.stderr.strip()}", file=sys.stderr)
        except FileNotFoundError as fnf:
            print(f"[MISS] {yyyymm}: {fnf}", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] {yyyymm}: {e}", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())
