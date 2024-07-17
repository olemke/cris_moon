from pathlib import Path
from glob import glob
import xarray
import numpy as np
from scipy.constants import speed_of_light
import logging
import datetime

logging.basicConfig(level=logging.INFO)

# define constants
mjy_sterad = 1e17
speed_of_light_cgs = speed_of_light * 1e2
cris_fov_diameter = 0.963

number_of_extra_scans = 3
radiance_average_extend = 2

wavenumber_ranges = (
    (30, 36),
    (42, 59),
    (56, 77),
    (74, 101),
    (95, 121),
    (122, 149),
    (148, 174),
    (232, 259),
    (374, 431),
    (590, 631),
    (933, 998),
    (1193, 1282),
    (1622, 1660),
    (1658, 1695),
    (1701, 1739),
    (1717, 1755),
    (1990, 2035),
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
                               radiances, wl):
    varname = f"Rad_{f_o_v}_{scanid}_average_{str(wl).replace('.','_')}"
    rad_avg = calc_radiance_average(wl, radiance_average_extend, radiances)
    moon_intrusions[varname] = rad_avg
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance average ({2*radiance_average_extend+1} values) around {wl}um for FOV {f_o_v} ScanID {scanid}"
    return rad_avg


def calculate_radiance_averages(moon_intrusions, f_o_v, scanid,
                                radiances):
    rad_avg_5_8 = calculate_radiance_average(moon_intrusions, f_o_v,
                                             scanid, radiances, 5.8)
    rad_avg_6_1 = calculate_radiance_average(moon_intrusions, f_o_v,
                                             scanid, radiances, 6.1)
    rad_avg_6_2 = calculate_radiance_average(moon_intrusions, f_o_v,
                                             scanid, radiances, 6.2)

    varname = f"Rad_{f_o_v}_{scanid}_ratio_5_8_to_6_1"
    moon_intrusions[varname] = rad_avg_5_8 / rad_avg_6_1
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance ratio between 5.8um and 6.1um for FOV {f_o_v} ScanID {scanid}"

    varname = f"Rad_{f_o_v}_{scanid}_ratio_avg_5_8_6_2_to_6_1"
    moon_intrusions[varname] = (rad_avg_5_8 + rad_avg_6_2) / 2. / rad_avg_6_1
    moon_intrusions[varname].attrs[
        'description'] = f"Radiance ratio between average of 5.8um+6.2um and 6.1um for FOV {f_o_v} ScanID {scanid}"


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
    moon_intrusions.attrs["creationtime"] = datetime.datetime.now().isoformat()
    moon_intrusions.attrs["n_Scans_cris"] = ds.Robs.shape[0]
    moon_intrusions.attrs["n_Scans_lunar"] = dia.shape[0]

    rad_maxind = []
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

            rad_maxind.append((f_o_v, maxind))
        else:
            logging.debug(
                f"{crisfile.name}:Max value {maxval} below threshold {threshold} "
                f"fov:{f_o_v}")

    if rad_maxind:
        moon_intrusions['rad_maxind'] = (('n_fov', 'n_scanid'), rad_maxind)
        moon_intrusions['rad_maxind'].attrs[
            'description'] = "Indices of max radiances for FOR 0"

    return moon_intrusions if rad_maxind else None


def find_max_mean_radiances(moon_intrusions):
    wavenumbers = np.array([np.mean(cris_wavenumbers()[r[0]:r[1]])
                            for r in wavenumber_ranges])
    max_means = []
    max_fovs = []
    max_scanids = []
    for wn, r in zip(wavenumbers, wavenumber_ranges):
        max_mean = -np.inf
        if 'rad_maxind' not in moon_intrusions:
            continue
        for maxind in moon_intrusions['rad_maxind'].values:
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
            max_means.append(np.nan)
            max_fovs.append(np.nan)
            max_scanids.append(np.nan)
        else:
            max_means.append(max_mean)
            max_fovs.append(max_fov)
            max_scanids.append(max_scanid)

    moon_intrusions['max_wavenumbers'] = wavenumbers
    moon_intrusions['max_fovs'] = max_fovs
    moon_intrusions['max_scanids'] = max_scanids
    moon_intrusions['max_radiances'] = max_means

    c1 = 1.191042e-5
    c2 = 1.4387752
    max_bts = c2 * wavenumbers / np.log(1 + c1 * wavenumbers**3 / max_means / 3e-7)
    moon_intrusions['max_brightness_temperatures'] = max_bts

    # Get moon data
    angles = xarray.load_dataset(moon_intrusions.lunarfile).angle
    dias = xarray.load_dataset(moon_intrusions.lunarfile).angular_diameter
    phases = xarray.load_dataset(moon_intrusions.lunarfile).phase
    max_dias = []
    max_phases = []
    max_min_angle = []
    max_scanid_min_angle = []
    for f_o_v, scanid in zip(max_fovs, max_scanids):
        if np.isnan(f_o_v) or np.isnan(scanid):
            max_dias.append(np.nan)
            max_phases.append(np.nan)
            continue
        # We already averaged the FORs, so we use FOR 0 here since
        # the values should be nearly identical for both FORs
        min_angle_index = np.argmin(angles[:, 0, f_o_v].values)
        max_dias.append(dias[min_angle_index, 0, f_o_v]*180./np.pi)
        max_phases.append(phases[min_angle_index, 0, f_o_v]*180./np.pi)
        max_min_angle.append(angles[min_angle_index, 0, f_o_v]*180./np.pi)
        max_scanid_min_angle.append(min_angle_index)
    moon_intrusions['max_angular_diameters'] = max_dias
    moon_intrusions['max_phases'] = max_phases
    moon_intrusions['max_min_angle'] = max_min_angle
    moon_intrusions['max_scanid_min_angle'] = max_scanid_min_angle


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
