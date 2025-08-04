import xarray as xr

sample_file = r"C:\Users\OEE2024_05\Documents\GitHub\ERA5-AWS-s3_project\ohio_ERA5_dataset testing\ERA5_OH_199501.nc"

ds = xr.open_dataset(sample_file)


pass