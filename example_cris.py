from process_cris import (find_cris_files, find_moon_intrusions)

for crisfile in find_cris_files(basedir='./SDR_data_npp', dirfilter="*"):
    moon_intrusions = find_moon_intrusions(crisfile, wavelen_id=99, threshold=20)
    moon_intrusions.to_netcdf(f"Rad_{crisfile.name}")
