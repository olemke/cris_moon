import xarray
import matplotlib.pyplot as plt


def plot_cris_maxima(filename, f_o_r=0):
    ds = xarray.open_dataset(filename)
    for varname in (v for v in ds.variables if v.startswith(f"Rad_{f_o_r}")):
        plt.plot(ds[varname], color='gray', alpha=0.5)
    for maxima in ds[f"rad_for{f_o_r}_maxind"]:
        f_o_v, maxind = maxima
        varname = f"Rad_{f_o_r}_{f_o_v.item()}_{maxind.item()}"
        plt.vlines(maxind,
                   0,
                   ds[varname].max(),
                   color="red",
                   linestyles="solid")
        plt.plot(ds[varname], label=varname)
    plt.legend()


def main():
    plot_cris_maxima("Rad_Lunar_fsr_spectra_20141229_2_npp.nc", f_o_r=0)
    plt.show()


if __name__ == "__main__":
    main()
