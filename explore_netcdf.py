#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import argparse

def summarize_nc(path_str: str, engine: str | None = None) -> int:
    """
    Print a concise summary of a NetCDF file without loading everything in RAM.
    Returns 0 on success, >0 on error.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    if not p.exists():
        print(f"[ERROR] File does not exist: {p}")
        return 2

    # Lazy import so the script still runs even if dependencies are missing
    import xarray as xr
    import os

    # Try engines in order; allow forcing one via --engine
    tried = []
    ds = None
    engines_to_try = [engine] if engine else ["netcdf4", "h5netcdf", None]
    for eng in engines_to_try:
        try:
            ds = xr.open_dataset(p.as_posix(), engine=eng, chunks={})
            used = eng if eng else "auto"
            break
        except Exception as e:
            tried.append((eng, str(e)))
            ds = None

    if ds is None:
        print("[ERROR] Could not open the file with any engine.")
        for eng, err in tried:
            print(f"  - engine={eng}: {err}")
        print("Hints: `pip install netcdf4` or `pip install h5netcdf`")
        return 3

    # Basic summary
    size_mb = os.path.getsize(p) / (1024 * 1024)
    print("=== NetCDF Summary ===")
    print(f"File        : {p}")
    print(f"Size        : {size_mb:.2f} MiB")
    print(f"Engine used : {used}")
    print("--- Dataset structure ---")
    print(ds)  # prints dims, coords, and variables at a high level

    # Helpful coordinate ranges if present
    def _minmax(coord: str):
        if coord in ds.coords:
            try:
                vmin = float(ds[coord].min())
                vmax = float(ds[coord].max())
                print(f"{coord:10s}: min={vmin:.4f}  max={vmax:.4f}")
            except Exception:
                pass

    print("--- Coordinate ranges ---")
    _minmax("time")
    _minmax("latitude")
    _minmax("longitude")

    # Variables with shape and dtype
    print("--- Variables ---")
    for name, da in ds.data_vars.items():
        try:
            shape = "x".join(str(s) for s in da.shape)
            dtype = str(da.dtype)
            print(f"{name:20s}  shape={shape:>15s}  dtype={dtype}")
        except Exception:
            print(f"{name:20s}")

    ds.close()
    return 0

def parse_args(argv):
    ap = argparse.ArgumentParser(description="Print a quick summary of a NetCDF file (xarray).")
    ap.add_argument("path", help="Path to the file (relative paths are fine).")
    ap.add_argument("--engine", choices=["netcdf4", "h5netcdf"], help="Force a specific xarray engine.")
    return ap.parse_args(argv)

def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    return summarize_nc(args.path, engine=args.engine)

if __name__ == "__main__":
    sys.exit(main())
