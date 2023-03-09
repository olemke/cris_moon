from pathlib import Path
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


def calculate_cris_radiances(ds, dia, scanid, f_o_v, f_o_r):
    """Calculate radiances for a given scan, field of view and field of regard.

    The counts are averaged over the two fields of regard.

    Args:
        ds (xarray.Dataset): CrIS SDR dataset
        dia (numpy.array): Lunar angular diameter
        scanid (int): Scan index
        f_o_v (int): Field of view index
        f_o_r (int): Field of regard index

    Returns:
        numpy.array: Radiance values
    """
    # For i=1:2223
    # Rad_8_33(i)=(Robs(i,8,1,33)+Robs(i,8,2,33))/2/2.997925E10*1E17*(0.963/(dia(8,1,33)*180/pi))^2
    # End
    for_average = (ds.Robs[scanid, 0, f_o_v, :] + ds.Robs[scanid, 1, f_o_v, :]) / 2.
    return for_average / speed_of_light_cgs * mjy_sterad * (
        cris_fov_diameter / (dia[scanid, f_o_r, f_o_v] * 180. / np.pi))**2


def find_moon_intrusions(crisfile, wavelen_id=99, threshold=20):
    """Find moon intrusions in CrIS SDR spectra.

    Args:
        crisfile (str): Input file
        wavelen_id (int, optional): Reference wavelen index. Defaults to 99.
        threshold (int, optional): Desired threshold radiance for moon detection.
            Defaults to 20.
    """
    logging.debug(f"Loading {crisfile}")
    ds = xarray.load_dataset(crisfile)
    lunarfile = crisfile.parents[0] / 'lunar_npp.h5'
    dia = xarray.load_dataset(lunarfile).angular_diameter

    # Create output dataset and metadata
    moon_intrusions = xarray.Dataset()
    moon_intrusions.attrs["crisfile"] = str(crisfile)
    moon_intrusions.attrs["lunarfile"] = str(lunarfile)
    moon_intrusions.attrs["creationtime"] = datetime.datetime.now().isoformat()

    rad_0_maxind = []
    rad_1_maxind = []
    for f_o_r in range(ds.Robs.shape[1]):
        for f_o_v in range(ds.Robs.shape[2]):
            selected_for_fov = ds.Robs[:, f_o_r, f_o_v, wavelen_id]
            # Find index of maximum value
            maxind = selected_for_fov.argsort()[-1:][0].item()
            # Determine value of maximum
            maxval = selected_for_fov[maxind].to_numpy()

            if maxval > threshold:
                logging.info(
                    f"{crisfile.name}:Found max value {maxval} at scanid {maxind} "
                    f"for:{f_o_r} fov:{f_o_v}")
                # Select additional scans around maximum
                selected_scanids = range(
                    ds.Robs.shape[0])[maxind - number_of_extra_scans:maxind +
                                      number_of_extra_scans + 1]
                logging.debug(
                    f"Selected scan ids: {ds.Robs[selected_scanids, f_o_r, f_o_v, wavelen_id].to_numpy()}")
                for scanid in selected_scanids:
                    radiances = calculate_cris_radiances(ds, dia, scanid, f_o_v, f_o_r)

                    # Create variable for this spectrum
                    varname = f"Rad_{f_o_r}_{f_o_v}_{scanid}"
                    moon_intrusions[varname] = radiances
                    moon_intrusions[varname].attrs[
                        'description'] = f"Radiances for FOR {f_o_r} FOV {f_o_v} ScanID {scanid}"

                (rad_0_maxind if f_o_r == 0 else rad_1_maxind).append((f_o_v, maxind))
            else:
                logging.debug(
                    f"{crisfile.name}:Max value {maxval} below threshold {threshold} "
                    f"for:{f_o_r} fov:{f_o_v}")

    if rad_0_maxind:
        moon_intrusions['rad_for0_maxind'] = (('n_0_fov', 'n_0_scanid'), rad_0_maxind)
        moon_intrusions['rad_for0_maxind'].attrs[
            'description'] = "Indices of max radiances for FOR 0"
    if rad_1_maxind:
        moon_intrusions['rad_for1_maxind'] = (('n_1_fov', 'n_1_scanid'), rad_1_maxind)
        moon_intrusions['rad_for1_maxind'].attrs[
            'description'] = "Indices of max radiances for FOR 1"

    return moon_intrusions if rad_0_maxind or rad_1_maxind else None


def main():
    for crisfile in find_cris_files(basedir='./SDR_data_npp', dirfilter="*"):
        moon_intrusions = find_moon_intrusions(crisfile, wavelen_id=99, threshold=20)
        if moon_intrusions:
            moon_intrusions.to_netcdf(f"Rad_{crisfile.name}")


if __name__ == '__main__':
    main()
