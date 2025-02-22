from pathlib import Path
from glob import glob
import xarray
import numpy as np
from scipy.constants import speed_of_light
import logging
import datetime as dt
from astropy.time import Time as aTime
from time import mktime
import xarray as xr

logging.basicConfig(level=logging.INFO)

# define constants
mjy_sterad = 1e17
speed_of_light_cgs = speed_of_light * 1e2
cris_fov_diameter = 0.963

number_of_extra_scans = 3
rad_ave_ext_SW = 120
rad_ave_ext_MW = 50
rad_ave_ext_LW = 65

wavenumber_ranges = (
    (30, 36),
    (42, 59),
    (56, 77),
    (74, 101),
    (82, 134),
    (220, 272),
    (374, 431),
    (569, 700),
    (590, 631),
    (969, 1070),
    (1193, 1282),
    (1599, 1840),
    (1622, 1660),
    (1658, 1695),
    (1701, 1739),
    (1717, 1755),
    (2112, 2169),
)


def cris_wavenumbers():
    # for i=1:717
    # wn(i)=648.75+(i-1)*(1096.25-648.75)/716
    # end
    # for i=718:717+869
    # wn(i)=1208.75+(i-718)*(1751.25-1208.75)/868
    # end
    # for i=717+870:717+869+637
    # wn(i)=2153.75+(i-717-870)*(2551.25-2153.75)/637
    # end
    wn = np.zeros((717 + 869 + 637))
    for i in np.arange(0, 717):
        wn[i] = 648.75 + i * (1096.25 - 648.75) / 716
    for i in np.arange(717, 717 + 869):
        wn[i] = 1208.75 + (i - 717) * (1751.25 - 1208.75) / 868
    for i in np.arange(717 + 869, 717 + 869 + 637):
        wn[i] = 2153.75 + (i - 717 - 869) * (2551.25 - 2153.75) / 637

    return wn


def find_cris_files(basedir, dirfilter="*"):
    """Generate input file list of CrIS SDR spectra.

    Args:
        basedir (str): Data directory containing CrIS SDR data
        dirfilter (str, optional): Regular expression to filter directories to
            process. Defaults to "*".
    """
    for dir in Path(basedir).glob(dirfilter):
        if dir.is_dir():
            for path in Path(dir).rglob('Lunar_fsr_spectra*.nc'):
                yield path.absolute()


def calculate_cris_radiances(ds, dia, scanid, f_o_v):
    """Calculate radiances for a given scan, field of view and field of regard.

    The counts are averaged over the two fields of regard.

    Args:
        ds (xarray.Dataset): CrIS SDR dataset
        dia (numpy.array): Lunar angular diameter
        scanid (int): Scan index
        f_o_v (int): Field of view index

    Returns:
        numpy.array: Radiance values
    """
    # For i=1:2223
    # Rad_8_33(i)=(Robs(i,8,1,33)+Robs(i,8,2,33))/2/2.997925E10*1E17*(0.963/(dia(8,1,33)*180/pi))^2
    # End
    for_average = (ds.Robs[scanid, 0, f_o_v, :] + ds.Robs[scanid, 1, f_o_v, :]) / 2.
    return for_average / speed_of_light_cgs * mjy_sterad * (
        cris_fov_diameter / (dia[scanid, 0, f_o_v] * 180. / np.pi))**2


def calc_radiance_average(wavelength, extend, radiances):
    """Calculate average radiance over a given wavelength range."""
    wavelengths = 10000 / cris_wavenumbers()
    wl_pos = np.abs(wavelengths - wavelength).argmin()
    return radiances[wl_pos - extend:wl_pos + extend + 1].mean()


def calculate_radiance_average(moon_intrusions, f_o_v, scanid,
                               radiances, radiance_average_extend, wl):
    varname = f"Rad_{f_o_v}_{scanid}_average_{str(wl).replace('.','_')}"
    rad_avg = calc_radiance_average(wl, radiance_average_extend, radiances)
    moon_intrusions[varname] = rad_avg
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance average ({2*radiance_average_extend+1} values) around {wl}um for FOV {f_o_v} ScanID {scanid}"
    return rad_avg


def calculate_radiance_averages(moon_intrusions, f_o_v, scanid,
                                radiances):
    rad_avg_SW = calculate_radiance_average(moon_intrusions, f_o_v,
                                             scanid, radiances, rad_ave_ext_SW, 4.4705)
    rad_avg_MW = calculate_radiance_average(moon_intrusions, f_o_v,
                                             scanid, radiances, rad_ave_ext_MW, 7.1556)
    rad_avg_LW = calculate_radiance_average(moon_intrusions, f_o_v,
                                             scanid, radiances, rad_ave_ext_LW, 9.5694)

    varname = f"Rad_{f_o_v}_{scanid}_SW"
    moon_intrusions[varname] = rad_avg_SW
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance average SW for FOV {f_o_v} ScanID {scanid}"

    varname = f"Rad_{f_o_v}_{scanid}_MW"
    moon_intrusions[varname] = rad_avg_MW
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance average MW for FOV {f_o_v} ScanID {scanid}"
    
    varname = f"Rad_{f_o_v}_{scanid}_LW"
    moon_intrusions[varname] = rad_avg_LW
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance average LW for FOV {f_o_v} ScanID {scanid}"


def find_moon_intrusions(crisfile, wavelen_id=99, threshold=20):
    """Find moon intrusions in CrIS SDR spectra.

    Args:
        crisfile (str): Input file
        wavelen_id (int, optional): Reference wavelen index. Defaults to 99.
        threshold (int, optional): Desired threshold radiance for moon detection.
            Defaults to 20.
    """
    if isinstance(crisfile, str):
        crisfile = Path(crisfile)

    logging.debug(f"Loading {crisfile}")
    ds = xarray.load_dataset(crisfile)
    lunarfile = glob(str(crisfile.parents[0] / 'lunar_*.h5'))[0]
    try:
        dia = xarray.load_dataset(lunarfile).angular_diameter
    except ValueError:
        raise ValueError(f"Could not find lunar file for {crisfile}")

    # Create output dataset and metadata
    moon_intrusions = xarray.Dataset()
    moon_intrusions.attrs["crisfile"] = str(crisfile)
    moon_intrusions.attrs["lunarfile"] = str(lunarfile)
    moon_intrusions.attrs["creationtime"] = dt.datetime.now().isoformat()
    moon_intrusions.attrs["n_Scans_cris"] = ds.Robs.shape[0]
    moon_intrusions.attrs["n_Scans_lunar"] = dia.shape[0]

    mi_maxind = []
    mi_maxval = []
    for f_o_v in range(ds.Robs.shape[2]):
        selected_for_fov = ds.Robs[:, 0, f_o_v, wavelen_id]
        # Find index of maximum value
        maxind = selected_for_fov.argsort()[-1:][0].item()
        # Determine value of maximum
        maxval = selected_for_fov[maxind].to_numpy()

        if maxval > threshold:
            logging.info(
                f"{crisfile.name}:Found max value {maxval} at scanid {maxind} "
                f"for:0 fov:{f_o_v}")
            # Select additional scans around maximum
            selected_scanids = range(
                ds.Robs.shape[0])[maxind - number_of_extra_scans:maxind +
                                    number_of_extra_scans + 1]
            logging.debug(
                f"Selected scan ids: {ds.Robs[selected_scanids, 0, f_o_v, wavelen_id].to_numpy()}")
            for scanid in selected_scanids:
                radiances = calculate_cris_radiances(ds, dia, scanid, f_o_v)

                # Create variable for this spectrum
                varname = f"Rad_{f_o_v}_{scanid}"
                moon_intrusions[varname] = radiances
                moon_intrusions[varname].attrs[
                    'description'] = f"Radiances for FOV {f_o_v} ScanID {scanid}"

                calculate_radiance_averages(
                    moon_intrusions, f_o_v, scanid, radiances)

            mi_maxind.append((f_o_v, maxind))
            mi_maxval.append(maxval)
        else:
            logging.debug(
                f"{crisfile.name}:Max value {maxval} below threshold {threshold} "
                f"fov:{f_o_v}")

    if mi_maxind:
        moon_intrusions['mi_maxind'] = (('n_intrusions', 'n_values'), mi_maxind)
        moon_intrusions['mi_maxind'].attrs[
            'description'] = "Indices of max radiances for FOR 0"
        moon_intrusions["mi_maxval"] = (("n_intrusions"), mi_maxval)
        moon_intrusions["mi_maxval"].attrs[
            "description"] = f"Radiance for wavelen id {wavelen_id} for FOR 0"

    return moon_intrusions if mi_maxind else None


def tai_time_to_datetime(tai):
    """Convert TAI time to datetime object."""
    # Difference between TAI and UTC in seconds
    offset = (dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) -
              dt.datetime(1958, 1, 1, tzinfo=dt.timezone.utc)).total_seconds()

    return aTime(tai/1e6 - offset, format='unix_tai', scale='tai').utc.datetime


def find_max_mean_radiances(moon_intrusions):
    wavenumbers = np.array([np.mean(cris_wavenumbers()[r[0]:r[1]])
                            for r in wavenumber_ranges])
    max_means = []
    max_fovs = []
    max_scanids = []
    max_bts = []
    if 'mi_maxind' not in moon_intrusions:
        raise RuntimeError("No moon intrusionsfound")
    for maxind in moon_intrusions['mi_maxind'].values:
        tmax_means = []
        tmax_fovs = []
        tmax_scanids = []
        for wn, r in zip(wavenumbers, wavenumber_ranges):
            max_mean = -np.inf
            for scanid in range(maxind[1]-1, maxind[1]+2):
                if scanid > moon_intrusions.attrs['n_Scans_cris'] - 1:
                    continue
                try:
                    this_mean = np.mean(
                        moon_intrusions[f'Rad_{maxind[0]}_{scanid}'].values[r[0]:r[1]])
                except KeyError:
                    raise KeyError(
                        f"Could not find variable Rad_{maxind[0]}_{scanid} for file {moon_intrusions.attrs['crisfile']}")
                if this_mean > max_mean:
                    max_mean = this_mean
                    max_fov = maxind[0]
                    max_scanid = scanid
            if max_mean == -np.inf:
                tmax_means.append(np.nan)
                tmax_fovs.append(-1)
                tmax_scanids.append(-1)
            else:
                tmax_means.append(max_mean)
                tmax_fovs.append(max_fov)
                tmax_scanids.append(max_scanid)
        max_means.append(tmax_means)
        max_fovs.append(tmax_fovs)
        max_scanids.append(tmax_scanids)

        c1 = 1.191042e-5
        c2 = 1.4387752
        max_bts.append(c2 * wavenumbers / np.log(1 + c1 * wavenumbers**3 / tmax_means / 3e-7))


    moon_intrusions['max_wavenumbers'] = (('n_intrusions', 'n_wavenumbers'), np.tile(wavenumbers, (len(moon_intrusions['mi_maxind']), 1)))
    moon_intrusions['max_fovs'] = (('n_intrusions', 'n_wavenumbers'), max_fovs)
    moon_intrusions['max_scanids'] = (('n_intrusions', 'n_wavenumbers'), max_scanids)
    moon_intrusions['max_radiances'] = (('n_intrusions', 'n_wavenumbers'), max_means)
    moon_intrusions['max_brightness_temperatures'] = (('n_intrusions', 'n_wavenumbers'), max_bts)

    # Get moon data
    timestamp = xarray.load_dataset(moon_intrusions.lunarfile).Time
    angles = xarray.load_dataset(moon_intrusions.lunarfile).angle
    diameter = xarray.load_dataset(moon_intrusions.lunarfile).angular_diameter
    phases = xarray.load_dataset(moon_intrusions.lunarfile).phase
    max_diameters = []
    max_phases = []
    max_angles = []
    max_times = []
    for rf_o_v, rscanid in zip(max_fovs, max_scanids):
        max_diameter = []
        max_phase = []
        max_angle = []
        max_time = []
        for f_o_v, scanid in zip(rf_o_v, rscanid):
            if f_o_v == -1 or scanid == -1:
                max_diameter.append(np.nan)
                max_phase.append(np.nan)
                max_angle.append(np.nan)
                max_time.append(-1)
                continue
            # We already averaged the FORs, so we use FOR 0 here since
            # the values should be nearly identical for both FORs
            max_diameter.append(diameter[scanid, 0, f_o_v]*180./np.pi)
            max_phase.append(phases[scanid, 0, f_o_v]*180./np.pi)
            max_angle.append(angles[scanid, 0, f_o_v]*180./np.pi)
            max_time.append(float(timestamp[scanid, 0, f_o_v]))
        max_diameters.append(max_diameter)
        max_phases.append(max_phase)
        max_angles.append(max_angle)
        max_times.append(max_time)
    moon_intrusions['max_angular_diameters'] = (('n_intrusions', 'n_wavenumbers'), max_diameters)
    moon_intrusions['max_angular_diameters'].attrs['description'] = "Angular diameter of moon at maximum radiance"
    moon_intrusions['max_phases'] = (('n_intrusions', 'n_wavenumbers'), max_phases)
    moon_intrusions['max_phases'].attrs['description'] = "Phase angle of moon at maximum radiance"
    moon_intrusions['max_angle'] = (('n_intrusions', 'n_wavenumbers'), max_angles)
    moon_intrusions['max_angle'].attrs['description'] = "Angle between moon and satellite at maximum radiance"
    moon_intrusions["max_time"] = (('n_intrusions', 'n_wavenumbers'), [
        [(tai_time_to_datetime(t).strftime("%Y-%m-%d %H:%M:%S.%f")) if t > 0 else "" for t in r] for r in max_times
    ])
    moon_intrusions["max_time"].attrs["description"] = "Time of maximum radiance"
    moon_intrusions["max_unixtime"] = (('n_intrusions', 'n_wavenumbers'), [
    [(mktime(tai_time_to_datetime(t).timetuple())) if t > 0 else -1 for t in r] for r in max_times ])
    moon_intrusions["max_time"].attrs["description"] = "Unix timestamp of maximum radiance"

    return moon_intrusions


def collect_results_in_csv(path, csvfile="intrusions.csv"):
    csv_fields = [
        "max_time",
        "max_fovs",
        "max_scanids",
        "max_angular_diameters",
        "max_phases",
        "max_wavenumbers",
        "max_radiances",
        "max_brightness_temperatures",
    ]
    sep = ","
    with open(csvfile, "w", newline="") as csvfile:
        first = True
        for field in csv_fields:
            if first:
                first = False
            else:
                csvfile.write(sep)
            if field == "max_time":
                csvfile.write("date" + sep + "time")
            else:
                csvfile.write(f"{field.replace('max_', '')}")
        csvfile.write(sep + "crisfile")
        csvfile.write("\n")
        for filename in Path(path).rglob("Rad_*.nc"):
            ds = xr.open_dataset(filename)
            outputranges = [(569, 700), (969, 1070), (1599, 1840)]

            maxval = np.max(ds.mi_maxval.values)
            for r in range(len(ds.n_intrusions)):
                if ds.mi_maxval.values[r] < maxval*0.95:
                    continue
                values = np.stack([ds[f].values[r, :] for f in csv_fields]).T
                for r in outputranges:
                    i = wavenumber_ranges.index(r)
                    if not values[i][0]:
                        logging.info(f"No data for {r} in {filename}")
                        continue
                    first = True
                    for value, field in zip(values[i], csv_fields):
                        if first:
                            first = False
                        else:
                            csvfile.write(sep)
                        if field == "max_time":
                            t = value.split(" ")
                            csvfile.write(f"{t[0]}{sep}{t[1]}")
                        else:
                            csvfile.write(f"{value}")
                    csvfile.write(sep + f"{Path(ds.attrs['crisfile']).name}")
                    csvfile.write("\n")


def main():
    wl = 10000/cris_wavenumbers()
    print(cris_wavenumbers())
    print((np.abs(wl - 6.05)).argmin())
    print((np.abs(wl - 6.15)).argmin())
    print((np.abs(wl - 6.2)).argmin())
    print((np.abs(wl - 5.8)).argmin())
    for crisfile in find_cris_files(basedir='./SDR_data_npp', dirfilter="*"):
        moon_intrusions = find_moon_intrusions(crisfile, wavelen_id=99, threshold=20)
        if moon_intrusions:
            moon_intrusions.to_netcdf(f"Rad_{crisfile.name}")


# if __name__ == '__main__':
#     main()
