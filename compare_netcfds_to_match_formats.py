#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NetCDF comparator for PyCharm (no CLI args):
- Set GOOD_PATH and TEST_PATH below, or leave them as None to get file-pickers.
- Compares file model, dimensions, coordinates, variables, and time axis.
- Optionally writes a .txt report next to the 'test' file.

Requirements:
  pip install xarray netCDF4 numpy
"""

from pathlib import Path
import numpy as np

# ========== CONFIGURE HERE (option 1: direct file paths) ==========
# Examples:
# GOOD_PATH = r"C:\Users\you\Documents\netcdf\good_daily.nc"
# TEST_PATH = r"C:\Users\you\Documents\netcdf\my_monthly.nc"
GOOD_PATH = r"C:\OEE Dropbox\Pablo Nieves\Due Diligence - Raw Data\ERA5\OH - 38 to 42.5N -86.25 to -80W\1994-2025\ERA5_N38.0_to_42.5_W-86.25_to_-80.0_1995_01_01.nc"   # set to a string path or leave as None
TEST_PATH = r"C:\Users\OEE2024_05\Documents\GitHub\ERA5-AWS-s3_project\ohio_ERA5_dataset testing\ERA5_OH_199501.nc"   # set to a string path or leave as None
SAVE_REPORT = True  # write a .txt report next to the test file

# ========== (option 2) Use file-pickers if paths are None ==========
def select_files_if_needed():
    """Show file dialogs if GOOD_PATH/TEST_PATH were not set."""
    global GOOD_PATH, TEST_PATH
    if GOOD_PATH and TEST_PATH:
        return
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        if not GOOD_PATH:
            GOOD_PATH = filedialog.askopenfilename(
                title="Select reference NetCDF (the one that works in your app)",
                filetypes=[("NetCDF", "*.nc"), ("All files", "*.*")]
            )
        if not TEST_PATH:
            TEST_PATH = filedialog.askopenfilename(
                title="Select generated NetCDF to compare",
                filetypes=[("NetCDF", "*.nc"), ("All files", "*.*")]
            )
    except Exception:
        # If no GUI available, the script will fail later with a clear message
        pass

# ========= Helpers to open and summarize =========
def open_xr(path: str):
    """Open a dataset trying multiple engines for compatibility."""
    import xarray as xr
    for eng in ("scipy", "netcdf4", None):
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            continue
    raise RuntimeError(f"Could not open {path} with any engine (xarray).")

def detect_file_model(path: str) -> str:
    """Return data model via netCDF4 if available (e.g., NETCDF3_CLASSIC, NETCDF4_CLASSIC)."""
    try:
        from netCDF4 import Dataset
        with Dataset(path, "r") as ds:
            return ds.data_model
    except Exception:
        return "unknown"

def summarize_nc(path: str) -> dict:
    """Extract a concise structural summary of a NetCDF file."""
    ds = open_xr(path)
    info = {}
    info["path"] = str(path)
    info["file_model"] = detect_file_model(path)

    # Dimensions
    info["dims"] = {k: int(v) for k, v in ds.dims.items()}

    # Coordinates (dtype, dims, a few key attrs)
    coord_summary = {}
    for c in ds.coords:
        da = ds[c]
        coord_summary[c] = {
            "dtype": str(da.dtype),
            "dims": tuple(da.dims),
            "attrs": {k: str(v) for k, v in da.attrs.items()
                      if k in ("units", "standard_name", "long_name", "axis")}
        }
    info["coords"] = coord_summary

    # Data variables (dtype, dims, common attrs)
    var_summary = {}
    for v in ds.data_vars:
        da = ds[v]
        attrs_keep = ("units", "_FillValue", "missing_value",
                      "scale_factor", "add_offset", "standard_name", "long_name")
        var_summary[v] = {
            "dtype": str(da.dtype),
            "dims": tuple(da.dims),
            "attrs": {k: str(da.attrs[k]) for k in attrs_keep if k in da.attrs}
        }
    info["vars"] = var_summary

    # Time encoding & quick stats
    if "time" in ds.coords:
        enc = ds["time"].encoding
        info["time_encoding"] = {k: str(enc[k]) for k in ("units", "calendar") if k in enc}
        try:
            tvals = ds["time"].values
            if tvals.size > 0:
                info["time_first"] = str(tvals[0])
                info["time_last"] = str(tvals[-1])
                if tvals.size >= 2:
                    # numpy datetime64 in ns; convert to hours
                    step_ns = (tvals[1] - tvals[0]).astype("timedelta64[ns]").astype(np.int64)
                    info["time_step_hours"] = round(step_ns / 3_600_000_000_000, 6)
        except Exception:
            pass

    # Global attributes
    info["global_attrs"] = {k: str(v) for k, v in ds.attrs.items()}

    ds.close()
    return info

def diff_dicts(a: dict, b: dict):
    """Return keys only in a, only in b, and in both (sorted)."""
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    both = sorted(set(a) & set(b))
    return only_a, only_b, both

def compare_reports(good: dict, test: dict) -> dict:
    """Compare two summaries and build a difference report."""
    out = {}

    # File model
    out["file_model_good"] = good["file_model"]
    out["file_model_test"] = test["file_model"]

    # Dimensions
    oa, ob, both = diff_dicts(good["dims"], test["dims"])
    out["dims_only_in_good"] = oa
    out["dims_only_in_test"] = ob
    out["dims_size_diff"] = {k: (good["dims"][k], test["dims"][k])
                             for k in both if good["dims"][k] != test["dims"][k]}

    # Coordinates (presence)
    oa, ob, bothc = diff_dicts(good["coords"], test["coords"])
    out["coords_only_in_good"] = oa
    out["coords_only_in_test"] = ob

    # Coordinates (details)
    coord_diffs = {}
    for c in bothc:
        g = good["coords"][c]; t = test["coords"][c]
        diff = {}
        if g["dtype"] != t["dtype"]:
            diff["dtype"] = (g["dtype"], t["dtype"])
        if tuple(g["dims"]) != tuple(t["dims"]):
            diff["dims"] = (g["dims"], t["dims"])
        keys = set(g["attrs"]) | set(t["attrs"])
        attrdiff = {}
        for k in keys:
            gv = g["attrs"].get(k); tv = t["attrs"].get(k)
            if gv != tv:
                attrdiff[k] = (gv, tv)
        if attrdiff:
            diff["attrs"] = attrdiff
        if diff:
            coord_diffs[c] = diff
    out["coords_diffs"] = coord_diffs

    # Variables (presence)
    oa, ob, bothv = diff_dicts(good["vars"], test["vars"])
    out["vars_only_in_good"] = oa
    out["vars_only_in_test"] = ob

    # Variables (details)
    var_diffs = {}
    for v in bothv:
        g = good["vars"][v]; t = test["vars"][v]
        diff = {}
        if g["dtype"] != t["dtype"]:
            diff["dtype"] = (g["dtype"], t["dtype"])
        if tuple(g["dims"]) != tuple(t["dims"]):
            diff["dims"] = (g["dims"], t["dims"])
        keys = set(g["attrs"]) | set(t["attrs"])
        attrdiff = {}
        for k in keys:
            gv = g["attrs"].get(k); tv = t["attrs"].get(k)
            if gv != tv:
                attrdiff[k] = (gv, tv)
        if attrdiff:
            diff["attrs"] = attrdiff
        if diff:
            var_diffs[v] = diff
    out["vars_diffs"] = var_diffs

    # Time axis
    out["time_good"] = good.get("time_encoding", {})
    out["time_test"] = test.get("time_encoding", {})
    out["time_first_good"] = good.get("time_first")
    out["time_first_test"] = test.get("time_first")
    out["time_last_good"] = good.get("time_last")
    out["time_last_test"] = test.get("time_last")
    out["time_step_hours_good"] = good.get("time_step_hours")
    out["time_step_hours_test"] = test.get("time_step_hours")

    return out

def build_text_report(diff: dict, good_path: str, test_path: str) -> str:
    """Format a human-readable report with the comparison results and hints."""
    lines = []
    lines.append(f"GOOD: {good_path}")
    lines.append(f"TEST: {test_path}")
    lines.append("")

    lines.append("=== File model ===")
    lines.append(f"good: {diff['file_model_good']}")
    lines.append(f"test: {diff['file_model_test']}")
    lines.append("")

    lines.append("=== Dimensions ===")
    if diff["dims_only_in_good"]:
        lines.append(f"Only in good: {diff['dims_only_in_good']}")
    if diff["dims_only_in_test"]:
        lines.append(f"Only in test: {diff['dims_only_in_test']}")
    if diff["dims_size_diff"]:
        lines.append("Size differences:")
        for k, (ga, ta) in diff["dims_size_diff"].items():
            lines.append(f"  {k}: good={ga}, test={ta}")
    lines.append("")

    lines.append("=== Coordinates (presence) ===")
    if diff["coords_only_in_good"]:
        lines.append(f"Only in good: {diff['coords_only_in_good']}")
    if diff["coords_only_in_test"]:
        lines.append(f"Only in test: {diff['coords_only_in_test']}")
    lines.append("")

    lines.append("=== Coordinates (detail differences) ===")
    if not diff["coords_diffs"]:
        lines.append("(none)")
    else:
        for c, d in diff["coords_diffs"].items():
            lines.append(f"- {c}:")
            for k, v in d.items():
                lines.append(f"   {k}: {v}")
    lines.append("")

    lines.append("=== Variables (presence) ===")
    if diff["vars_only_in_good"]:
        lines.append(f"Only in good: {diff['vars_only_in_good']}")
    if diff["vars_only_in_test"]:
        lines.append(f"Only in test: {diff['vars_only_in_test']}")
    lines.append("")

    lines.append("=== Variables (detail differences) ===")
    if not diff["vars_diffs"]:
        lines.append("(none)")
    else:
        for v, d in diff["vars_diffs"].items():
            lines.append(f"- {v}:")
            for k, val in d.items():
                lines.append(f"   {k}: {val}")
    lines.append("")

    lines.append("=== Time axis ===")
    lines.append(f"good encoding: {diff['time_good']}")
    lines.append(f"test encoding: {diff['time_test']}")
    lines.append(f"good first/last: {diff.get('time_first_good')}  {diff.get('time_last_good')}")
    lines.append(f"test first/last: {diff.get('time_first_test')}  {diff.get('time_last_test')}")
    lines.append(f"good step (h): {diff.get('time_step_hours_good')}")
    lines.append(f"test step (h): {diff.get('time_step_hours_test')}")
    lines.append("")

    lines.append("=== Suggestions ===")
    lines.append("- Ensure coordinates are named exactly 'latitude' and 'longitude'.")
    lines.append("- Ensure 'latitude' and 'longitude' are float32 (Single).")
    lines.append("- Set coord attrs: latitude.units='degrees_north', longitude.units='degrees_east'; add axis=Y/X.")
    lines.append("- Ensure time units are 'hours since 1900-01-01 00:00:00' and calendar='standard'.")
    lines.append("- If the reader dislikes HDF5, write final files as NetCDF3 (xarray engine='scipy').")
    lines.append("- Keep data variable dims ordered as (time, latitude, longitude).")
    lines.append("- If the good file uses _FillValue/scale_factor/add_offset, mirror them if needed.")

    return "\n".join(lines)

def main():
    # Pick files if paths were not configured
    select_files_if_needed()

    if not GOOD_PATH or not TEST_PATH:
        raise SystemExit("No paths provided/selected for GOOD_PATH and TEST_PATH.")

    good_path = str(GOOD_PATH)
    test_path = str(TEST_PATH)

    # Summaries
    good = summarize_nc(good_path)
    test = summarize_nc(test_path)

    # Diff
    diff = compare_reports(good, test)

    # Print report (PyCharm console)
    report = build_text_report(diff, good_path, test_path)
    print(report)

    # Save .txt report
    if SAVE_REPORT:
        out_txt = Path(test_path).with_suffix(".compare_report.txt")
        out_txt.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {out_txt}")

if __name__ == "__main__":
    main()
