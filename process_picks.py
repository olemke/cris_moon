#%%
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from process_cris import find_moon_intrusions, cris_wavenumbers

N_CPUS = 8

filelist = [
    "20150625_1",
    "20150625_2",
    "20150625_3",
    "20150625_4",
    "20150625_5",
    "20150625_6",
    "20150625_7",
    "20150625_8",
    "20150626_1",
    "20150626_2",
    "20150626_3",
    "20150626_4",
    "20150626_5",
    "20150626_6",
    "20150627_1",
    "20150627_2",
    "20160615_1",
    "20160615_2",
    "20160615_3",
    "20160615_4",
    "20160713_1",
    "20160713_2",
    "20160713_3",
    "20160713_4",
    "20160713_5",
    "20160713_6",
    "20170601_1",
    "20170601_2",
    "20170601_3",
    "20170604_1",
    "20170604_2",
    "20170604_3",
    "20170701_1",
    "20170701_2",
    "20170701_3",
    "20170701_4",
    "20170701_5",
    "20170701_6",
    "20170703_1",
    "20170703_2",
    "20170703_3",
    "20170703_4",
    "20170703_5",
    "20170703_6",
    "20180620_1",
    "20180622_1",
    "20180622_2",
    "20180623_1",
    "20180623_2",
    "20190610_1",
    "20190610_2",
    "20190610_3",
    "20190612_1",
    "20190612_2",
    "20190612_3",
    "20200601_1",
    "20200601_2",
    "20200601_3",
    "20200628_1",
    "20200628_2",
    "20200628_3",
    "20200628_4",
    "20200628_5",
    "20200630_1",
    "20200630_2",
    "20200630_3",
    "20200630_4",
    "20210618_1",
    "20210618_2",
    "20210618_3",
    "20210618_4",
    "20210620_1",
    "20210620_2",
    "20210620_3",
    "20210620_4",
    "20220607_1",
    "20220607_2",
    "20220607_3",
    "20220610_1",
    "20220610_2",
    "20220610_3",
]


def get_scans_with_maximum(df, fieldofregard=0):
    """Return list of spectra with maximum in field of regard"""
    return [
        df[f"Rad_{fieldofregard}_{maxind.values[0]}_{maxind.values[1]}"]
        for maxind in df[f"rad_for{fieldofregard}_maxind"]
    ]


def main():
    #%%
    find_moon_intrusions_partial = partial(find_moon_intrusions,
                                           wavelen_id=99,
                                           threshold=20)
    files = [
        f"../SDR_data_npp/{f}/Lunar_fsr_spectra_{f}_npp.nc" for f in filelist
    ]
    with Pool(N_CPUS) as pool:
        results = pool.map(find_moon_intrusions_partial, files)

    #%%
    valid_results = [
        r for r in results
        if r is not None and "rad_for0_maxind" in r and "rad_for1_maxind" in r
    ]

    no_results = [filelist[i] for i, r in enumerate(results) if r is None]
    print(f"Files without results: {len(no_results)} out of {len(filelist)}")

    #%%
    max_scans = [
        get_scans_with_maximum(df, fieldofregard=0) for df in valid_results
    ]
    max_scans += [
        get_scans_with_maximum(df, fieldofregard=1) for df in valid_results
    ]
    max_scans = sum(max_scans, [])
    combined_scans = xr.concat(max_scans, dim="scans").rename("combined_scans")
    combined_scans.attrs[
        "description"] = "Combination of all scans that fulfill selection criteria"
    wn = 10000 / cris_wavenumbers()
    combined_scans = combined_scans.assign_coords(n_Channels=wn)
    combined_scans.n_Channels.attrs["units"] = "µm"

    # %%
    fig, ax = plt.subplots()
    combined_scans.mean(dim="scans").sel(n_Channels=slice(9.0, 7.0)).plot(
        ax=ax)

    # Point plot to visualize gaps in the frequency grid
    # combined_scans.mean(dim="scans").sel(n_Channels=slice(10.0, 9.0)).plot(
    #     ax=ax, marker='.', markersize=0.75, ls='None')

    fig.savefig("mean_scans.pdf")
    #combines_scans.to_netcdf("combined_scans.nc")


if __name__ == "__main__":
    main()
