import numpy as np
from pprint import PrettyPrinter
from process_cris import find_cris_files, find_moon_intrusions, find_max_mean_radiances
from multiprocessing import Pool


def process_file(crisfile):
    moon_intrusions = find_moon_intrusions(crisfile, wavelen_id=99, threshold=20)

    if moon_intrusions:
        max_radiances = find_max_mean_radiances(moon_intrusions)
        print(f"Found {len(moon_intrusions)} moon intrusions in {crisfile.name}:")
        PrettyPrinter(width=120).pprint(
            list(
                zip(
                    np.round(moon_intrusions.max_wavenumbers.values, decimals=1),
                    moon_intrusions.max_fors.values,
                    moon_intrusions.max_fovs.values,
                    moon_intrusions.max_scanids.values,
                    np.round(moon_intrusions.max_angular_diameters.values, decimals=2),
                    np.round(moon_intrusions.max_phases.values, decimals=2),
                    np.round(moon_intrusions.max_radiances.values, decimals=1),
                    np.round(moon_intrusions.max_brightness_temperatures.values, decimals=1),
                )
            )
        )
        moon_intrusions.to_netcdf(f"out/Rad_{crisfile.name}")
    else:
        print(f"No moon intrusions found in {crisfile.name}")

# for crisfile in find_cris_files(basedir="./SDR_data_npp", dirfilter="20220312_*"):
#     process_file(crisfile)


with Pool(processes=8) as pool:
    pool.map(process_file, find_cris_files(basedir="./SDR_data_j01", dirfilter="*"))
