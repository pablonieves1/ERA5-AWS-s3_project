import xarray as xr

sample_file = r'C:\OEE Dropbox\Pablo Nieves\Due Diligence - Raw Data\ERA5\OH - 38 to 42.5N -86.25 to -80W\1994-2025\ERA5_N38.0_to_42.5_W-86.25_to_-80.0_2025_07_01.nc'

ds = xr.open_dataset(sample_file)


pass