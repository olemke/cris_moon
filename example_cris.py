import os
from process_cris import (
    find_cris_files,
    find_moon_intrusions,
    find_max_mean_radiances,
    collect_results_in_csv,
)
from multiprocessing import Pool


def process_file(crisfile, output_directory="out"):
    moon_intrusions = find_moon_intrusions(crisfile, wavelen_id=99, threshold=20)

    if moon_intrusions:
        find_max_mean_radiances(moon_intrusions)
        print(
            f"Found {len(moon_intrusions.n_intrusions)} moon intrusions in {crisfile.name}"
        )
        os.makedirs(output_directory, exist_ok=True)
        moon_intrusions.to_netcdf(f"{output_directory}/Rad_{crisfile.name}")
    else:
        print(f"No moon intrusions found in {crisfile.name}")


if __name__ == "__main__":
    with Pool(processes=8) as pool:
        pool.map(process_file, find_cris_files(basedir="./SDR_data_npp", dirfilter="*"))
    collect_results_in_csv(path="./out")
