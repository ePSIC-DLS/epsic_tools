#!/usr/bin/env bash
import os
import shutil
import numpy as np
import h5py
import blosc
from hyperspy.signals import Signal2D
from hyperspy.misc.array_tools import rebin
from hyperspy.signal import BaseSignal

from itertools import product
from math import floor

# C extensions
try:
    from fast_binning import fast_bin
except ImportError:
    print("Cannot import fast binning function.")

from mib_prop import mib_props
import ipywidgets
import ipywidgets as widgets
import json
import sys
import glob
import logging
import subprocess
import time

formatter = logging.Formatter("%(asctime)s    %(process)5d %(processName)-12s %(threadName)-12s                   %(levelname)-8s %(pathname)s:%(lineno)d %(message)s")
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set the debug log level.
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)

# Make a logger for this module.
logger = logging.getLogger(__name__)


#####################################################################################################################
#############################################   mib_flyback_utils - start   #########################################
#####################################################################################################################

PIXEL_DEPTH_NPY_TYPE = {"U01": np.uint8,
                        "U08": np.uint8,
                        "U16": np.uint16,
                        "U32": np.uint32,
                        "U64": np.uint64,
                        }

PIXEL_DEPTH_NPY_TYPE_PROMOTED = {"U01": np.uint8,
                                 "U08": np.uint16,
                                 "U16": np.uint32,
                                 "U32": np.uint64,
                                 "U64": np.uint64,
                                 }
def _add_crosses(a):
    """
    Adds 3 pixel buffer cross to quad chip data.

    Parameters
    ----------
    a : numpy.ndarray
        Stack of raw frames or reshaped dask array object, prior to dimension reshaping, to insert
        3 pixel buffer cross into.

    Returns
    -------
    b : numpy.ndarray
        Stack of frames or reshaped 4DSTEM object including 3 pixel buffer cross in the diffraction plane.
    """
    original_shape = a.shape

    if len(original_shape) == 4:
        a = a.reshape(
            original_shape[0] * original_shape[1], original_shape[2], original_shape[3]
        )

    a_half = int(original_shape[-1] / 2), int(original_shape[-2] / 2)
    # Define 3 pixel wide cross of zeros to pad raw data
    if len(original_shape) == 4:
        z_array = np.zeros(
            (original_shape[0] * original_shape[1], original_shape[-2], 3),
            dtype=a.dtype,
        )
        z_array2 = np.zeros(
            (original_shape[0] * original_shape[1], 3, original_shape[-1] + 3),
            dtype=a.dtype,
        )
    else:
        z_array = np.zeros((original_shape[0], original_shape[-2], 3), dtype=a.dtype)
        z_array2 = np.zeros(
            (original_shape[0], 3, original_shape[-1] + 3), dtype=a.dtype
        )

    # Insert blank cross into raw data
    b = np.concatenate((a[:, :, : a_half[1]], z_array, a[:, :, a_half[1] :]), axis=-1)

    b = np.concatenate((b[:, : a_half[0], :], z_array2, b[:, a_half[0] :, :]), axis=-2)

    if len(original_shape) == 4:
        b = b.reshape(
            original_shape[0],
            original_shape[1],
            original_shape[2] + 3,
            original_shape[3] + 3,
        )

    return b

def STEM_flag_dict(exp_times_list):
    """
    Determines whether a .mib file contains STEM or TEM data and how many
    frames to skip due to triggering from a list of exposure times.

    Parameters
    ----------
    exp_times_list : list
        List of exposure times extracted from a .mib file.

    Returns
    -------
    output : dict
        Dictionary containing - STEM_flag, scan_X, exposure_time,
                                number_of_frames_to_skip, flyback_times
    Example
    -------
    {'STEM_flag': 1,
     'scan_X': 256,
     'exposure time': 0.0007,
     'number of frames_to_skip': 136,
     'flyback_times': [0.0392, 0.0413, 0.012625, 0.042]}
    """
    output = {}
    times_set = set(exp_times_list)
    # If single exposure times in header, treat as TEM data.
    if len(times_set) == 1:
        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = list(times_set)
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # In case exp times not appearing in header treat as TEM data
    elif len(times_set) == 0:

        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = None
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # Otherwise, treat as STEM data.
    else:
        STEM_flag = 1
        # Check that the smallest time is the majority of the values
        exp_time = max(times_set, key=exp_times_list.count)
        if exp_times_list.count(exp_time) < int(0.9 * len(exp_times_list)):
            logger.debug('Something wrong with the triggering!')
        peaks = [i for i, e in enumerate(exp_times_list) if e > 5 * exp_time]
        # Diff between consecutive elements of the array
        lines = np.ediff1d(peaks)
        if len(set(lines)) == 1:
            scan_X = lines[0]
            frames_to_skip = peaks[0]
            # if frames_to_skip is 1 less than scan_X we do not need to skip any frames
            # if frames_to_skip == peaks[0]:
            #     frames_to_skip = 0
        else:
            # Assuming the last element to be the line length
            scan_X = lines[-1]
            check = np.ravel(np.where(lines == scan_X, True, False))
            # Checking line lengths
            start_ind = np.where(check == False)[0][-1] + 2
            frames_to_skip = peaks[start_ind]

        flyback_times = list(times_set)
        flyback_times.remove(exp_time)
        output['STEM_flag'] = STEM_flag
        output['scan_X'] = scan_X
        output['exposure time'] = exp_time
        output['number of frames_to_skip'] = frames_to_skip
        output['flyback_times'] = flyback_times

    return output


def get_scan_y(nframe, start_frame, scan_x):
    """Return the number of rows (scan y)."""
    return floor((nframe - start_frame) / scan_x)


def bright_flyback_frame(start_frame, scan_y, scan_x):
    """Return the indices of flyback frame in a stack (the first column)."""
    return start_frame + np.arange(scan_y)*scan_x


def grid_chunk_offsets(scan_shape):
    """A generator for grid chunk offsets."""
    scan_y = scan_shape[0]
    scan_x = scan_shape[1]

    for syx in product(range(scan_y), range(scan_x)):
        yield (syx[0], syx[1], 0, 0)


def stack_chunk_offsets(scan_shape):
    """A generator for stack chunk offsets."""
    scan_yx = scan_shape[0]

    for syx in range(scan_yx):
        yield (syx, 0, 0)


def binned_nav_indices(linear_index, ncol, bw, row_shift=0, col_shift=0):
    """Get the bin indices from a linear index.

    If I have the 5x5 array below and bin it by 2 (bw):

    -------------------------------
    |  0  |  1  |  2  |  3  |  4  |
    -------------------------------
    |  5  |  6  |  7  |  8  |  9  |
    -------------------------------
    |  10 |  11 |  12 |  13 |  14 |
    -------------------------------
    |  15 |  16 |  17 |  18 |  19 |
    -------------------------------
    |  20 |  21 |  22 |  23 |  24 |
    -------------------------------

    (the indices are linear)

    the binned array will be a 2x2 array, and if row_shift and col_shift
    are both 1, the above data will be in the following bin:

    -----------------------------
    |             |             |
    |  6,7,11,12  |  8,9,13,14  |
    |             |             |
    -----------------------------
    |             |             |
    | 16,17,21,22 | 18,19,23,24 |
    |             |             |
    -----------------------------

    the row_shift and col_shift is equivalent to the number of cropping
    from top row and left column, respectively.

    The function returns:
        - (0, 0) for linear indices 6, 7, 11, 12;
        - (0, 1) for linear indices 8, 9, 13, 14;
        - (1, 0) for linear indices 16, 17, 21, 22;
        - (1, 1) for linear indices 18, 19, 23, 24;

    Parameters
    ----------
    linear_index : int
        the index (linear) of the 2D array to be binned
    ncol : int
        the number of column of the 2D array
    bw : int
        the width of the bin
    row_shift, col_shift : int
        the number of rows and columns to be shifted, equivalent to
        cropping to the same number of top rows and left columns
        respectively

    Returns
    -------
    binned_row_idx, binned_col_idx
        the row and column indices of the bin
    """

    norm_idx = linear_index - col_shift - ncol*row_shift

    binned_row_idx = norm_idx // (ncol*bw)
    binned_col_idx = (norm_idx % ncol) // bw

    return binned_row_idx, binned_col_idx

def empty_hspy_hdf5(output_path, shape, data_dict=None):
    """Create an empty hdf5 file following HyperSpy hierarchy with metadata.

    Parameters
    ----------
    output_path : str
        the output hdf5 file
    shape : tuple
        the shape of the dataset, with 4 members (scan_y, scan_x, det_y,
        det_x).
    data_dict : dict, optional
        a dictionary contains some values for the metadata

    Returns
    -------
    the dataset key where the actual data will be saved.

    """
    axes_dict = _get_axes_dict(shape)

    # construct metadata dictionary
    metadata_dict = {"Signal": {}}
    metadata_dict["Signal"]["flip"] = "True"
    # this is what pyxem would have set at the end
    # so skip setting "STEM" or "TEM"
    metadata_dict["Signal"]["signal_type"] = "electron_diffraction"

    if data_dict is not None:
        metadata_dict["Signal"]["scan_X"] = data_dict["scan_X"]
        metadata_dict["Signal"]["frames_number_skipped"] = data_dict["number of frames_to_skip"]
        # in ms
        metadata_dict["Signal"]["exposure_time"] = data_dict["exposure time"]
        metadata_dict["Signal"]["flyback_times"] = data_dict["flyback_times"]

    # fake HyperSpy signal
    # its content and dims are not important (it is not saved)
    s = BaseSignal(np.empty((1,2,3,4)),
                   axes=axes_dict,
                   metadata=metadata_dict
                   )

    # this creates a file consistent with HyerSpy signal
    # "write_dataset=False" to skip writing the fake data
    # we just want the hierarchy
    s.save(output_path,
           overwrite=True,
           file_format="HSPY",
           write_dataset=False)

    # inspect the created hdf5 file and return the dataset where "data"
    # should be saved
    with h5py.File(output_path, "r") as f:
        # this should be fixed by Hyperspy
        expg = f["/Experiments"]
        # this could depend on "title" in the metadata
        # and it has one member only
        dset_name = list(expg.keys())[0]

    return f"/Experiments/{dset_name}/data"

def _get_axes_dict(shape):
    if len(shape) == 3:
        # a stack
        syx = shape[0]
        dety = shape[1]
        detx = shape[2]

        # construct HyperSpy axes dictionary
        ax_syx = {"size": syx, "navigate": True}
        ax_dy = {"size": dety}
        ax_dx = {"size": detx}

        return [ax_syx, ax_dy, ax_dx]
    elif len(shape) == 4:
        # a grid
        sy = shape[0]
        sx = shape[1]
        dety = shape[2]
        detx = shape[3]

        # construct HyperSpy axes dictionary
        ax_sy = {"size": sy, "navigate": True}
        ax_sx = {"size": sx, "navigate": True}
        ax_dy = {"size": dety}
        ax_dx = {"size": detx}

        return [ax_sy, ax_sx, ax_dy, ax_dx]

    msg = "It only supports saving 3D or 4D data"
    raise ValueError(msg)

#####################################################################################################################
#############################################   mib_flyback_utils - start   #########################################
#####################################################################################################################





###########################################################################################################
#############################################   Functions - Start   #######################################
###########################################################################################################


def find_metadat_file(timestamp, acquisition_path):
    metadata_file_paths = []
    mib_file_paths = []

    for root, folders, files in os.walk(acquisition_path):
        for file in files:
            if file.endswith('hdf'):
                metadata_file_paths.append(os.path.join(root, file))
            elif file.endswith('mib'):
                mib_file_paths.append(os.path.join(root, file))
    for path in metadata_file_paths:
        if timestamp == path.split('/')[-1].split('.')[0]:
            return path
    logger.debug('No metadata file could be matched.')
    return

def write_vds(source_h5_path, writing_h5_path, entry_key='Experiments/__unnamed__/data', vds_key = '/data/frames', metadata_path = ''):
    if metadata_path is None:
        try:
            with h5py.File(source_h5_path,'r') as f:
                vsource = h5py.VirtualSource(f[entry_key])
                sh = vsource.shape
                logger.debug(f"4D shape: {sh}")
        except KeyError:
            logger.debug('Key provided for the input data file not correct')
            return

        layout = h5py.VirtualLayout(shape=tuple((np.prod(sh[:2]), sh[-2], sh[-1])), dtype = np.uint16)
        for i in range(sh[0]):
            for j in range(sh[1]):
                layout[i * sh[1] + j] = vsource[i, j, :, :]

        with h5py.File(writing_h5_path, 'w', libver='latest') as f:
            f.create_virtual_dataset(vds_key, layout)
    else:
        # copy over the metadata file
        src_path = metadata_path
        dest_path = os.path.dirname(writing_h5_path)
        shutil.copy(src_path, dest_path)

        # Open the metadata dest file and add links
        try:
            with h5py.File(source_h5_path,'r') as f:
                vsource = h5py.VirtualSource(f[entry_key])
                sh = vsource.shape
                logger.debug(f"4D shape {sh}")
        except KeyError:
            logger.debug('Key provided for the input data file not correct')
            return

        layout = h5py.VirtualLayout(shape=tuple((np.prod(sh[:2]), sh[-2], sh[-1])), dtype = np.uint16)
        for i in range(sh[0]):
            for j in range(sh[1]):
                layout[i * sh[1] + j] = vsource[i, j, :, :]
        logger.debug('Adding vds to: ' + os.path.join(dest_path, os.path.basename(metadata_path)))
        with h5py.File(os.path.join(dest_path, os.path.basename(metadata_path)), 'r+', libver='latest') as f:
            f.create_virtual_dataset(vds_key, layout)
            f['/data/mask'] = h5py.ExternalLink('/dls_sw/e02/medipix_mask/Merlin_12bit_mask.h5', "/data/mask")

    return


def gen_config(template_path, dest_path, config_name, meta_file_path, rotation_angle, camera_length, conv_angle):
    config_file = dest_path + '/' + config_name + '.json'

    with open(template_path, 'r') as template_file:
        pty_expt = json.load(template_file)
    data_path = meta_file_path

    pty_expt['base_dir'] = dest_path
    pty_expt['process']['save_dir'] = dest_path
    pty_expt['experiment']['data']['data_path'] = data_path
    pty_expt['experiment']['experiment_ID'] = config_name

    pty_expt['process']['common']['scan']['rotation'] = rotation_angle

    # pty_expt['process']['common']['scan']['N'] = scan_shape
    pty_expt['experiment']['detector']['position'] = [0, 0, camera_length]
    pty_expt['experiment']['optics']['lens']['alpha'] = conv_angle

    #edit the json in order to perform auto ptycho
    pty_expt['process']['common']['scan']['region'] = [0.0,1.0,0.0,1.0,1,1]
    pty_expt['process']['PIE']['MultiSlice']['slices'] = 1
    pty_expt['process']['PIE']['iterations'] = 25

    with h5py.File(meta_file_path, 'r') as microscope_meta:
        meta_values = microscope_meta['metadata']
        pty_expt['process']['common']['scan']['N'] = [int(meta_values['4D_shape'][:2][0]),
                                                      int(meta_values['4D_shape'][:2][1])]
        pty_expt['process']['common']['source']['energy'] = [float(np.array(meta_values['ht_value(V)']))]
        pty_expt['process']['common']['scan']['dR'] = [float(np.array(meta_values['step_size(m)'])),
                                                       float(np.array(meta_values['step_size(m)']))]
        # pty_expt['experiment']['optics']['lens']['alpha'] = 2 * float(np.array(meta_values['convergence_semi-angle(rad)']))
        pty_expt['experiment']['optics']['lens']['defocus'] = [float(np.array(meta_values['defocus(nm)']) * 1e-9),
                                                               float(np.array(meta_values['defocus(nm)']) * 1e-9)]
        pty_expt['process']['save_prefix'] = config_name

    with open(config_file, 'w') as f:
        json.dump(pty_expt, f, indent=4)


def Meta2Config(acc,nCL,aps):
    '''This function converts the meta data from the 4DSTEM data set into parameters to be used in a ptyREX json file'''

    '''The rotation angles noted here are from ptychographic reconstructions which have been successful. see the
    following directory for example reconstruction from which these values are derived:
     /dls/science/groups/imaging/ePSIC_ptychography/experimental_data'''
    if acc == 80e3:
        rot_angle = 238.5
        print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 41.65e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 31.74e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 24.80e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =15.44e-3
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    elif acc == 200e3:
        rot_angle = -77.585
        print('Rotation angle = ' + str(rot_angle) +' Warning: This rotation angle need further calibration')
        if aps == 1:
            conv_angle = 37.7e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 28.8e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 22.4e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle = 14.0
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 5:
            conv_angle = 6.4
            print('Condenser aperture size is 10um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
    elif acc == 300e3:
        rot_angle = -85.5
        print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 44.7e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 34.1e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 26.7e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =16.7e-3
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    else:
        print('Rotation angle for this acceleration voltage is unknown, please collect calibration data. Rotation angle being set to zero')
        rot_angle = 0

    '''this is incorrect way of calucating the actual camera length but okay for this prototype code'''
    '''TODO: add py4DSTEM workflow which automatic determines the camera length from a small amount of reference data and the known convergence angle'''
    camera_length = 1.5*nCL
    print('camera length estimated to be ' + str(camera_length))

    return rot_angle,camera_length,conv_angle


###########################################################################################################
#############################################   Functions - End   #########################################
###########################################################################################################


info_path = sys.argv[1]
index = int(sys.argv[2])
info = {}
with open(info_path, 'r') as f:
    for line in f:
        tmp = line.split(" ")
        if tmp[0] == 'to_convert_paths':
            info[tmp[0]] = line.split(" = ")[1].split('\n')[:-1]
            print(tmp[0], line.split(" = ")[1].split('\n')[:-1])
        else:
            info[tmp[0]] = tmp[-1].split("\n")[0]
            print(tmp[0], tmp[-1].split("\n")[0])

mib_path = eval(info['to_convert_paths'][0])[index]

adr_split = mib_path.split("/")
for i in range(len(adr_split)):
    if adr_split[i] == "Merlin":
        break

tmp_save = []
tmp_save.append('/')
tmp_save.extend(adr_split[:i])
tmp_save.append('processing')
tmp_save.extend(adr_split[i:-2])
save_dir = os.path.join(*tmp_save)

# Load data as stack

src_path = []
src_path.append('/')
src_path.extend(adr_split[:-2])
src_path = os.path.join(*src_path) + "/"

time_stamp = adr_split[-1].split('_data')[0]
save_path = os.path.join(save_dir, time_stamp)
if not os.path.exists(save_path):
     os.makedirs(save_path)

auto_reshape = eval(info['auto_reshape'])
no_reshaping = eval(info['no_reshaping'])
use_fly_back = eval(info['use_fly_back'])
known_shape = eval(info['known_shape'])
Scan_X = eval(info['Scan_X'])
Scan_Y = eval(info['Scan_Y'])
iBF = eval(info['iBF'])
bin_sig_flag = eval(info['bin_sig_flag'])
bin_sig_factor = eval(info['bin_sig_factor'])
bin_nav_flag = eval(info['bin_nav_flag'])
bin_nav_factor = eval(info['bin_nav_factor'])
reshaping = info['reshaping']
try:
    print('stuff')
    create_json = eval(info['create_json'])
except:
    create_json = False
if create_json:
    ptycho_config = info['ptycho_config']
    ptycho_template = info['ptycho_template']
try:
    create_virtual_image = eval(info['create_virtual_image'])
except:
    create_virtual_image = False
if create_virtual_image:
    disk_lower_thresh = eval(info['disk_lower_thresh'])
    disk_upper_thresh = eval(info['disk_upper_thresh'])
    mask_path = info['mask_path']
    DPC_check = eval(info['DPC'])
    parallax_check = eval(info['parallax'])

hdf5_path = os.path.join(save_path, f'{time_stamp}_data.hdf5')
ibf_path = os.path.join(save_path, f'{time_stamp}_iBF.jpg')
bin_nav_path = os.path.join(save_path, f'{time_stamp}_data_bin_nav_factor_{bin_nav_factor}.hspy')
bin_sig_path = os.path.join(save_path, f'{time_stamp}_data_bin_sig_factor_{bin_sig_factor}.hspy')


# check provided reshaping options
print('**********')
print(no_reshaping, use_fly_back, known_shape)
print('**********')

# check provided reshaping options
if sum([bool(auto_reshape), bool(no_reshaping), bool(use_fly_back), bool(known_shape)]) != 1:
    msg = (f"Only one of the options 'auto_reshape' ({auto_reshape}), "
           f"'no_reshaping' ({no_reshaping}) or 'known_shape' "
           f"'use_fly_back' ({use_fly_back}) or 'known_shape' "
           f"({known_shape}) should be True.")
    raise ValueError(msg)


# the Blosc filter registered ID (for h5py)
compression_id = 32001
# maximum compression level (0-9)
clevel = 9
# use "blosclz" compressor
compressor = "blosclz"
compressor_code = blosc.name_to_code(compressor)

# set the block size for Blosc compression
# the default is 32 kB and modfiied by clevel and others although
# the docs said setting to L2 (but source code use L1 as a start?)
# cache should provide some optimisation, some experimentation told
# us that 1 kB is quite good for our compression setting
try:
    blksz = int(os.environ["BLOSC_BLOCKSIZE"])
except (KeyError, ValueError):
    blksz = 1024
blosc.set_blocksize(blksz)
print(f"Blosc block size: {blksz} B")

try:
    blosc_nthreads = int(os.environ["BLOSC_NTHREADS"])
except (KeyError, ValueError):
    # leave it as default, i.e. the maximum detected number
    blosc_nthreads = blosc.detect_number_of_cores()
    blosc.set_nthreads(blosc_nthreads)
    print(f"Blosc number of threads: {blosc_nthreads}")

# fetch all useful information from the headers in the mib file
mib_properties = mib_props(mib_path,
                       sequence_number=True,
                       header_bytes=True,
                       pixel_depth=True,
                       det_x=True,
                       det_y=True,
                       exposure_time_ns=True,
                       bit_depth=True,
                       )

if auto_reshape:
    if mib_properties['sequence_number'][-1] == 262144:
        no_reshaping = False
        use_fly_back = True
        known_shape = False
        print("Going to use the 'Fly-back' option")
        print("The scan shape will be 512*512")
    elif mib_properties['sequence_number'][-1] == 261632:
        no_reshaping = False
        use_fly_back = False
        known_shape = True
        Scan_X = 512
        Scan_Y = 511
        print("Going to use the 'known_shape' option")
        print("The scan shape will be 512*511")
    elif mib_properties['sequence_number'][-1] == 65536:
        no_reshaping = False
        use_fly_back = True
        known_shape = False
        print("Going to use the 'Fly-back' option")
        print("The scan shape will be 256*256")
    elif mib_properties['sequence_number'][-1] == 65280:
        no_reshaping = False
        use_fly_back = False
        known_shape = True
        Scan_X = 256
        Scan_Y = 255
        print("Going to use the 'known_shape' option")
        print("The scan shape will be 256*255")
    elif mib_properties['sequence_number'][-1] == 16384:
        no_reshaping = False
        use_fly_back = True
        known_shape = False
        print("Going to use the 'Fly-back' option")
        print("The scan shape will be 128*128")
    elif mib_properties['sequence_number'][-1] == 16256:
        no_reshaping = False
        use_fly_back = False
        known_shape = True
        Scan_X = 128
        Scan_Y = 127
        print("Going to use the 'known_shape' option")
        print("The scan shape will be 128*127")
    else:
        no_reshaping = True
        use_fly_back = False
        known_shape = False
        print("A proper scan shape was not detected")
        print("The scan shape will be %d*1"%(mib_properties['sequence_number'][-1]))

if no_reshaping == True:
    iBF = False
    bin_nav_flag = 0

# check the size of the detector to determine whether or not to add a cross
if mib_properties['det_x'][0] == 256:
    print("Single-Medipix 4DSTEM data - No cross added")
    add_cross = False
elif mib_properties['det_x'][0] == 512:
    print("Quad-Medipix 4DSTEM data - Cross added")
    add_cross = True
else:
    print("Warning! The dimensions of diffraction pattern are unusual.")


with open(mib_path, "rb") as mib:
    # determine header size, dtype and detector size from first header
    header_size = mib_properties["header_bytes"][0]
    det_y = mib_properties["det_y"][0]
    det_x = mib_properties["det_x"][0]
    dtype = PIXEL_DEPTH_NPY_TYPE[mib_properties["pixel_depth"][0]]

    # the number of bytes of each frame, including header
    num_frames = len(mib_properties["sequence_number"])
    stride = (header_size + det_y*det_x*np.dtype(dtype).itemsize)
    exposure_time_ms = mib_properties["exposure_time_ns"] * 1e-6

    if use_fly_back:
        # parse all headers to fetch metadata from every frame
        data_dict = STEM_flag_dict(exposure_time_ms.tolist())

        start_frame = data_dict["number of frames_to_skip"]
        scan_x = data_dict["scan_X"]
        scan_y = get_scan_y(num_frames, start_frame, scan_x)
        flyback_frames = bright_flyback_frame(start_frame, scan_y, scan_x)
        end_frame = start_frame + scan_y*scan_x
    else:
        data_dict = None

        start_frame = 0
        scan_x = Scan_X
        scan_y = Scan_Y
        flyback_frames = ()
        end_frame = num_frames

    if add_cross:
        # for 2x2 chip configuration, in order to represent
        # correct angular relationship in reciprocal space,
        # a 3 pixel-width cross is added
        width_cross = 3
    else:
        width_cross = 0

    if no_reshaping:
        # a stack
        mib_data_shape = (num_frames,
                          det_y + width_cross,
                          det_x + width_cross
                          )
        chunk_sz = (1, mib_data_shape[-2], mib_data_shape[-1])

        if iBF:
            msg = ("Saving the MIB frames as a stack does not support "
                   "saving the integrated bright field image.")
            raise ValueError(msg)

        if bin_nav_flag:
            msg = ("Saving the MIB frames as a stack does not support "
                   "binning across the navigation dimension.")
            raise ValueError(msg)
    elif use_fly_back:
        # use information from flyback
        # scan_x-1 to account for the flyback column
        mib_data_shape = (scan_y,
                          scan_x - 1,
                          det_y + width_cross,
                          det_x + width_cross
                          )
        chunk_sz = (1, 1, mib_data_shape[-2], mib_data_shape[-1])

    elif known_shape:
        # use the provided shape
        mib_data_shape = (scan_y,
                          scan_x,
                          det_y + width_cross,
                          det_x + width_cross
                          )
        chunk_sz = (1, 1, mib_data_shape[-2], mib_data_shape[-1])

        if scan_y*scan_x != num_frames:
            msg = (f"The requested scan dimension ({scan_y} x {scan_x}) "
                   f"does not match the total number of frames "
                   f"({num_frames}) in the MIB file.")
            raise ValueError(msg)
    else:
        msg = "You have to select one of the actions on reshaping!!!"
        raise ValueError(msg)

    if iBF:
        # guarantee the first two dims are scanning dims
        # fix uint32
        ibf_buffer = np.zeros(mib_data_shape[:2], dtype=np.uint32)

    if bin_sig_flag:
        dtype_bin_sig = PIXEL_DEPTH_NPY_TYPE_PROMOTED[mib_properties["pixel_depth"][0]]

        # determine top row (y) and left col (x) for cropping
        sig_to_cropy = mib_data_shape[-2] % bin_sig_factor
        sig_to_cropx = mib_data_shape[-1] % bin_sig_factor

    if bin_nav_flag:
        dtype_bin_nav = PIXEL_DEPTH_NPY_TYPE_PROMOTED[mib_properties["pixel_depth"][0]]

        # assume enough memory for holding this array
        arr_nav_binned = np.zeros((mib_data_shape[0] // bin_nav_factor,
                                   mib_data_shape[1] // bin_nav_factor,
                                   mib_data_shape[2],
                                   mib_data_shape[3]),
                                  dtype=dtype_bin_nav
                                  )

        # determine top row (y) and left col (x) for cropping
        nav_to_cropy = mib_data_shape[0] % bin_nav_factor
        nav_to_cropx = mib_data_shape[1] % bin_nav_factor

    # create an hdf5 file following HyperSpy hierarchy
    # without saving actual data but with metadata
    # "dset_path" is the key where the actual data will be saved
    dset_path = empty_hspy_hdf5(hdf5_path, mib_data_shape, data_dict)

    with h5py.File(hdf5_path, "r+") as hdf:
        # the last 3 indices:
        # compression level (0-9)
        # NOSHUFFLE, SHUFFLE, BITSHUFFLE
        # compressor, "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"
        if np.dtype(dtype).itemsize == 1:
            # for 1 byte data type don't use bit shuffle
            comopts_mib = (0, 0, 0, 0,
                           clevel, blosc.SHUFFLE, compressor_code)
        else:
            comopts_mib = (0, 0, 0, 0,
                           clevel, blosc.BITSHUFFLE, compressor_code)

        # create the output dataset
        dset = hdf.require_dataset(dset_path,
                                   shape=mib_data_shape,
                                   dtype=dtype,
                                   chunks=chunk_sz,
                                   compression=compression_id,
                                   compression_opts=comopts_mib,
                                   )

        # start at the beginning of the MIB file
        ptr = 0
        # ensure the dtype is big endian (from Merlin manual)
        dtype_be = np.dtype(dtype).newbyteorder(">")
        # some counters for tracking
        count_saved = 0
        count_skipped = 0
        count_sig_binned = 0
        count_nav_binned = 0
        # set up the chunk offset generator
        if no_reshaping:
            ch_offset_gen = stack_chunk_offsets(mib_data_shape)
        else:
            ch_offset_gen = grid_chunk_offsets(mib_data_shape)

        # for each frame, save to the hdf5 dataset
        for k in range(num_frames):
            # if count_saved == 10:
                # break
            if (start_frame <= k < end_frame) and (k not in flyback_frames):
                # point to the current frame
                mib.seek(ptr)
                # read the frame (header+data), this moves the file
                # pointer by stride
                frame = mib.read(stride)

                # construct only the frame
                arr = np.frombuffer(frame[header_size:], dtype=dtype_be)

                # convert from big-endian to little-endian
                arr_le = arr.byteswap()
                arr_le = arr_le.view(arr_le.dtype.newbyteorder())

                # reshape to 2D for flipping
                resh = arr_le.reshape(det_y, det_x)

                # flip it to match those from pyxem (but why?)
                resh = np.flipud(resh)

                # reshape to the chunk frame size
                if no_reshaping:
                    resh = resh.reshape(1, det_y, det_x)
                else:
                    resh = resh.reshape(1, 1, det_y, det_x)

                # add cross if needed
                # final shape will match chunk_sz
                if add_cross:
                    resh = _add_crosses(resh)

                if bin_sig_flag:
                    # crop the signal (divisible by bin factor),
                    # return to 2D, make sure C-contiguous
                    sig_cropped = np.ascontiguousarray(np.squeeze(resh)[sig_to_cropy:, sig_to_cropx:])

                    if 2 <= bin_sig_factor <= 8:
                        try:
                            # fast bin if available
                            sig_binned = fast_bin(sig_cropped,
                                                  bin_sig_factor,
                                                  np.dtype(dtype_bin_sig).num
                                                  )
                        except NameError:
                            # fall back to the slower bin
                            sig_binned = rebin(sig_cropped,
                                               scale=bin_sig_factor,
                                               dtype=dtype_bin_sig)
                    else:
                        # fall back to the slower bin
                        sig_binned = rebin(sig_cropped,
                                           scale=bin_sig_factor,
                                           dtype=dtype_bin_sig)

                    # promote to dims of chunk frame size
                    if no_reshaping:
                        sig_binned = np.array(sig_binned, copy=False, ndmin=3)
                    else:
                        sig_binned = np.array(sig_binned, copy=False, ndmin=4)

                    # create bin sig file using the binned shape
                    # only necessary the first time
                    if count_sig_binned == 0:
                        if no_reshaping:
                            bin_sig_shape = (mib_data_shape[0],
                                             *sig_binned.shape[-2:]
                                             )
                            bin_sig_chunk_sz = (1,
                                                *sig_binned.shape[-2:]
                                                )
                        else:
                            bin_sig_shape = (*mib_data_shape[:2],
                                             *sig_binned.shape[-2:]
                                             )
                            bin_sig_chunk_sz = (1, 1,
                                                *sig_binned.shape[-2:]
                                                )

                        dset_bin_sig_key = empty_hspy_hdf5(bin_sig_path,
                                                           bin_sig_shape,
                                                           data_dict)

                        # remember to close the file handle!
                        f_sig_bin = h5py.File(bin_sig_path, "r+")

                        # set up compression options
                        if np.dtype(dtype_bin_sig).itemsize == 1:
                            # for 1 byte data type don't use bit shuffle
                            comopts_bin_sig = (0, 0, 0, 0,
                                               clevel, blosc.SHUFFLE, compressor_code)
                        else:
                            comopts_bin_sig = (0, 0, 0, 0,
                                               clevel, blosc.BITSHUFFLE, compressor_code)

                        # create the output dataset
                        dset_bin_sig = f_sig_bin.require_dataset(dset_bin_sig_key,
                                                                 shape=bin_sig_shape,
                                                                 dtype=dtype_bin_sig,
                                                                 chunks=bin_sig_chunk_sz,
                                                                 compression=compression_id,
                                                                 compression_opts=comopts_bin_sig,
                                                                 )

                        # set up the chunk offset generator
                        if no_reshaping:
                            ch_offset_bin_sig_gen = stack_chunk_offsets(bin_sig_shape)
                        else:
                            ch_offset_bin_sig_gen = grid_chunk_offsets(bin_sig_shape)

                if bin_nav_flag:
                    # the if condition is equivalent to cropping
                    # such as [nav_to_cropy:, nav_to_cropx:], but operates
                    # on linear index (i.e. count_saved here)
                    if ((count_saved // mib_data_shape[1] >= nav_to_cropy) and
                        (count_saved % mib_data_shape[1] >= nav_to_cropx)):

                        # return the indices of this frame that
                        # belong to the current bin
                        bin_idx = binned_nav_indices(count_saved,
                                                     mib_data_shape[1],
                                                     bin_nav_factor,
                                                     row_shift=nav_to_cropy,
                                                     col_shift=nav_to_cropx,
                                                     )

                        # add the frame to the bin (binning)
                        arr_nav_binned[bin_idx[0], bin_idx[1], :, :] += resh[0, 0, :, :]
                        count_nav_binned += 1

                # Blosc compression by using pointer
                resh = np.ascontiguousarray(resh)
                arr_compressed = blosc.compress_ptr(resh.__array_interface__["data"][0],
                                                    items=resh.size,
                                                    typesize=resh.itemsize,
                                                    clevel=comopts_mib[4],
                                                    shuffle=comopts_mib[5],
                                                    cname=compressor,
                                                    )


                try:
                    # get the chunk offset for the dataset for this
                    # frame
                    chunk_offset = next(ch_offset_gen)
                except StopIteration:
                    msg = ("There are more chunks than it should be! "
                           "Check the shape of the dataset.")
                    raise RuntimeError(msg)
                else:
                    # write the frame to the offset for this frame
                    dset.id.write_direct_chunk(chunk_offset, arr_compressed)
                    count_saved += 1

                    # sum the frame for integrated bright field
                    if iBF:
                        ibf_buffer[chunk_offset[:2]] += resh.sum()


                # save the bin sig to the dataset
                if bin_sig_flag:
                    # Blosc compression by using pointer
                    sig_binned = np.ascontiguousarray(sig_binned)
                    sig_bin_compressed = blosc.compress_ptr(sig_binned.__array_interface__["data"][0],
                                                            items=sig_binned.size,
                                                            typesize=sig_binned.itemsize,
                                                            clevel=comopts_bin_sig[4],
                                                            shuffle=comopts_bin_sig[5],
                                                            cname=compressor,
                                                            )
                    try:
                        # get the chunk offset for the dataset for this
                        # frame
                        chunk_offset = next(ch_offset_bin_sig_gen)
                    except StopIteration:
                        msg = ("There are more chunks than it should be! "
                               "Check the shape of the dataset.")
                        raise RuntimeError(msg)
                    else:
                        # write the binned sig to the offset
                        dset_bin_sig.id.write_direct_chunk(chunk_offset, sig_bin_compressed)
                        count_sig_binned += 1
            else:
                # print(f"Skipped: frame index {k}, sequence number "
                      # f"{mib_properties['sequence_number'][k]}, "
                      # f"exposure time {mib_properties['exposure_time_ns'][k]} ns")
                count_skipped += 1

            # point to the next frame
            ptr += stride

        print(f"Number of frames saved: {count_saved}")
        print(f"Number of frames skipped: {count_skipped}")

    # save iBF after rescale to 8-bit and change dtype
    if iBF:
        ibf = ibf_buffer - ibf_buffer.min()
        if ibf.max() != 0:
            ibf = ibf * (255 / ibf.max())
        ibf = ibf.astype(np.uint8)

        s_ibf = Signal2D(ibf)
        s_ibf.save(ibf_path, overwrite=True)

    # save binned array across the navigation axes (first 2)
    if bin_nav_flag:
        # although not streaming the chunk, using HyperSpy save
        # method still much slower than iterating the NumPy array
        # and direct chunk write with Blosc compression

        # dummy file
        dset_bin_nav_path = empty_hspy_hdf5(bin_nav_path,
                                            arr_nav_binned.shape,
                                            data_dict
                                            )

        with h5py.File(bin_nav_path, "r+") as bin_nav_hdf:
            # set up compression options
            if np.dtype(arr_nav_binned.dtype).itemsize == 1:
                # for 1 byte data type don't use bit shuffle
                comopts_bin_nav = (0, 0, 0, 0,
                                   clevel, blosc.SHUFFLE, compressor_code)
            else:
                comopts_bin_nav = (0, 0, 0, 0,
                                   clevel, blosc.BITSHUFFLE, compressor_code)

            # create the output dataset
            chunk_sz = (1, 1, arr_nav_binned.shape[-2], arr_nav_binned.shape[-1])
            dset = bin_nav_hdf.require_dataset(dset_bin_nav_path,
                                               shape=arr_nav_binned.shape,
                                               dtype=arr_nav_binned.dtype,
                                               chunks=chunk_sz,
                                               compression=compression_id,
                                               compression_opts=comopts_bin_nav,
                                               )

            # set chunk offset generator
            ch_offset_gen = grid_chunk_offsets(arr_nav_binned.shape)

            # for each frame use direct chunk with compression
            for chunk_offset in ch_offset_gen:
                row = chunk_offset[0]
                col = chunk_offset[1]

                arr = arr_nav_binned[row, col, :, :].reshape(chunk_sz)

                # Blosc compression by using pointer
                arr = np.ascontiguousarray(arr)
                arr_compressed = blosc.compress_ptr(arr.__array_interface__["data"][0],
                                                    items=arr.size,
                                                    typesize=arr.itemsize,
                                                    clevel=comopts_bin_nav[4],
                                                    shuffle=comopts_bin_nav[5],
                                                    cname=compressor,
                                                    )

                # write the frame to the offset for this frame
                dset.id.write_direct_chunk(chunk_offset, arr_compressed)

    if bin_sig_flag:
        f_sig_bin.close()

print("Conversion finished.")
time.sleep(10)

meta_path = find_metadat_file(time_stamp, src_path)
write_vds(save_path+'/'+time_stamp+'_data.hdf5', save_path +'/'+time_stamp+'_vds.h5', metadata_path=meta_path)


if create_virtual_image:
    with h5py.File(meta_path, 'r') as microscope_meta:
        meta_values = microscope_meta['metadata']
        print(meta_values['aperture_size'][()])
        print(meta_values['nominal_camera_length(m)'][()])
        print(meta_values['ht_value(V)'][()])
        acc = meta_values['ht_value(V)'][()]
        nCL = meta_values['nominal_camera_length(m)'][()]
        aps = meta_values['aperture_size'][()]
    rot_angle,camera_length,conv_angle = Meta2Config(acc, nCL, aps)

    import time
    import tifffile
    import matplotlib.pyplot as plt
    import py4DSTEM
    import hyperspy.api as hs
    print(py4DSTEM.__version__)

    data_path = save_path+'/'+time_stamp+'_data.hdf5'
    data_name = data_path.split("/")[-1].split(".")[0]
    save_dir = os.path.dirname(data_path) # directory for saving the results
    print(meta_path)
    print(data_path)
    print(data_name)
    print(mask_path)

    device = 'cpu'

    if meta_path != '':
        try:
            with h5py.File(meta_path,'r') as f:
                print("----------------------------------------------------------")
                print(f['metadata']["defocus(nm)"])
                print(f['metadata']["defocus(nm)"][()])
                defocus_exp = f['metadata']["defocus(nm)"][()]*10 # Angstrom
                print("----------------------------------------------------------")
                print(f['metadata']["ht_value(V)"])
                print(f['metadata']["ht_value(V)"][()])
                HT = f['metadata']["ht_value(V)"][()]
                print("----------------------------------------------------------")
                print(f['metadata']["step_size(m)"])
                print(f['metadata']["step_size(m)"][()])
                scan_step = f['metadata']["step_size(m)"][()] * 1E10 # Angstrom
                print("----------------------------------------------------------")
        except:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                HT = metadata['process']['common']['source']['energy'][0]
                print("HT: ", HT)
                defocus_exp = metadata['experiment']['optics']['lens']['defocus'][0]*1E10
                print("defocus: ", defocus_exp)
                scan_step = metadata['process']['common']['scan']['dR'][0]*1E10
                print("scan step: ", scan_step)

    if mask_path != '':
        try:
            with h5py.File(mask_path,'r') as f:
                mask = f['data']['mask'][()]

        except:
            with h5py.File(mask_path,'r') as f:
                mask = f['root']['np.array']['data'][()]

        mask = np.invert(mask)
        mask = mask.astype(np.float32)

        print(type(mask))
        print(mask.dtype)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(mask)
        fig.tight_layout()
        plt.savefig(save_dir+"/mask.png")

    ### VERY IMPORTANT VARIABLE ###
    semiangle = conv_angle * 1000 # mrad
    ### ####################### ###

    if data_path.split(".")[-1] == "hspy":
    # This is for the simulated 4DSTEM data using 'submit_abTEM_4DSTEM_simulation.ipynb'
    # stored in /dls/science/groups/e02/Ryu/RYU_at_ePSIC/multislice_simulation/submit_abtem/submit_abtem_4DSTEM_simulation.ipynb
        original_stack = hs.load(data_path)
        print(original_stack)
        n_dim = len(original_stack.data.shape)
        scale = []
        origin = []
        unit = []
        size = []

        for i in range(n_dim):
            print(original_stack.axes_manager[i].scale, original_stack.axes_manager[i].offset, original_stack.axes_manager[i].units, original_stack.axes_manager[i].size)
            scale.append(original_stack.axes_manager[i].scale)
            origin.append(original_stack.axes_manager[i].offset)
            unit.append(original_stack.axes_manager[i].units)
            size.append(original_stack.axes_manager[i].size)

        HT = eval(original_stack.metadata["HT"])
        defocus_exp = eval(original_stack.metadata["defocus"])
        semiangle = eval(original_stack.metadata["semiangle"])
        scan_step = scale[0]
        print("HT: ", HT)
        print("experimental defocus: ", defocus_exp)
        print("semiangle: ", semiangle)
        print("scan step: ", scan_step)
        original_stack = original_stack.data
        det_name = 'ePSIC_EDX'
        data_key = 'Experiments/__unnamed__/data'


    elif data_path.split(".")[-1] == "hdf" or data_path.split(".")[-1] == "hdf5" or data_path.split(".")[-1] == "h5":
        # try:
        #     original_stack = hs.load(data_path, reader="HSPY", lazy=True)
        #     print(original_stack)
        #     original_stack = original_stack.data
        try:
            f = h5py.File(data_path,'r')
            print(f)
            original_stack = f['Experiments']['__unnamed__']['data'][:]
            f.close()
            det_name = 'ePSIC_EDX'
            data_key = 'Experiments/__unnamed__/data'

        except:
            f = h5py.File(data_path,'r')
            print(f)
            original_stack = f['data']['frames'][:]
            f.close()
            det_name = 'pty_data'
            data_key = "data/frames"

    elif data_path.split(".")[-1] == "dm4":
        original_stack = hs.load(data_path)
        print(original_stack)
        n_dim = len(original_stack.data.shape)
        scale = []
        origin = []
        unit = []
        size = []

        for i in range(n_dim):
            print(original_stack.axes_manager[i].scale, original_stack.axes_manager[i].offset, original_stack.axes_manager[i].units, original_stack.axes_manager[i].size)
            scale.append(original_stack.axes_manager[i].scale)
            origin.append(original_stack.axes_manager[i].offset)
            unit.append(original_stack.axes_manager[i].units)
            size.append(original_stack.axes_manager[i].size)

        HT = 1000 * original_stack.metadata['Acquisition_instrument']['TEM']['beam_energy']
        scan_step = scale[0] * 10
        defocus_exp = eval(info['defocus'])
        print("HT: ", HT)
        print("experimental defocus: ", defocus_exp)
        print("semiangle: ", semiangle)
        print("scan step: ", scan_step)
        original_stack = original_stack.data
        original_stack = original_stack.astype(np.float32)
        original_stack -= np.min(original_stack)
        original_stack /= np.max(original_stack)
        original_stack *= 128.0
        # det_name = 'ePSIC_EDX'
        # data_key = 'Experiments/__unnamed__/data'

    else:
        print("Wrong data format!")

    original_stack = original_stack.astype(np.float32)
    print(original_stack.dtype)
    print(original_stack.shape)
    print(np.min(original_stack), np.max(original_stack))

    # masking
    if original_stack.shape[2] == 515:
        if mask_path != '' and type(mask) == np.ndarray:
            for i in range(original_stack.shape[0]):
                for j in range(original_stack.shape[1]):
                    original_stack[i, j] = np.multiply(original_stack[i, j], mask)
                    
    elif original_stack.shape[2] == 256:
         if mask_path != '' and type(mask) == np.ndarray:
            for i in range(original_stack.shape[0]):
                for j in range(original_stack.shape[1]):
                    original_stack[i, j] = np.multiply(original_stack[i, j], mask[:256, :256])
            
    else:
        print('No mask')

    dataset = py4DSTEM.DataCube(data=original_stack)
    print("original dataset")
    print(dataset)

    del original_stack

    dataset.get_dp_mean()
    dataset.get_dp_max()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(dataset.tree('dp_mean')[:, :], cmap='jet')
    fig.tight_layout()
    plt.savefig(save_dir+"/pacbed.png")

    probe_radius_pixels, probe_qx0, probe_qy0 = dataset.get_probe_size(thresh_lower=disk_lower_thresh, thresh_upper=disk_upper_thresh, N=100, plot=True)
    plt.savefig(save_dir+"/disk_detection.png")

    dataset.calibration._params['Q_pixel_size'] = semiangle / probe_radius_pixels
    dataset.calibration._params['Q_pixel_units'] = "mrad"
    dataset.calibration._params['R_pixel_size'] = scan_step
    dataset.calibration._params['R_pixel_units'] = "A"

    print(dataset)
    print(dataset.calibration)

    # Make a virtual bright field and dark field image
    center = (probe_qx0, probe_qy0)
    radius_BF = probe_radius_pixels
    radii_DF = (probe_radius_pixels, int(dataset.Q_Nx/2))

    dataset.get_virtual_image(
        mode = 'circle',
        geometry = (center,radius_BF),
        name = 'bright_field',
        shift_center = False,
    )
    dataset.get_virtual_image(
        mode = 'annulus',
        geometry = (center,radii_DF),
        name = 'dark_field',
        shift_center = False,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(dataset.tree('bright_field')[:, :], cmap="inferno")
    ax[0].set_title("BF image")
    ax[1].imshow(dataset.tree('dark_field')[:, :], cmap="inferno")
    ax[1].set_title("ADF image [%.1f, %.1f] mrad"%(radii_DF[0]*dataset.Q_pixel_size, radii_DF[1]*dataset.Q_pixel_size))
    fig.tight_layout()
    plt.savefig(save_dir+"/STEM_image.png")
    vBF = dataset.tree('bright_field')[:, :]
    vADF = dataset.tree('dark_field')[:, :]
    # tifffile.imwrite(save_dir+"/BF_image.tif", dataset.tree('bright_field')[:, :])
    # tifffile.imwrite(save_dir+"/ADF_image.tif", dataset.tree('dark_field')[:, :])

    if DPC_check:
        dpc = py4DSTEM.process.phase.DPC(
            datacube=dataset,
            energy = HT,
        ).preprocess(force_com_rotation=rot_angle)
        plt.savefig(save_dir+"/DPC_optimization.png")

        dpc.reconstruct(
            max_iter=8,
            store_iterations=True,
            reset=True,
            gaussian_filter_sigma=0.1,
            gaussian_filter=True,
            q_lowpass=None,
            q_highpass=None
        ).visualize(
            iterations_grid='auto',
            figsize=(16, 10)
        )
        plt.savefig(save_dir+"/iDPC_image.png")

        dpc_cor = py4DSTEM.process.phase.DPC(
            datacube=dataset,
            energy=HT,
            verbose=False,
        ).preprocess(
            force_com_rotation = np.rad2deg(dpc._rotation_best_rad),
            force_com_transpose = False,
        )
        plt.savefig(save_dir+"/corrected_DPC_optimization.png")

        dpc_cor.reconstruct(
            max_iter=8,
            store_iterations=True,
            reset=True,
            gaussian_filter_sigma=0.1,
            gaussian_filter=True,
            q_lowpass=None,
            q_highpass=None
        ).visualize(
            iterations_grid='auto',
            figsize=(16, 10)
        )
        plt.savefig(save_dir+"/corrected_iDPC_image.png")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(dpc._com_normalized_y, cmap="bwr")
        ax[0].set_title("CoMx")
        ax[1].imshow(dpc._com_normalized_x, cmap="bwr")
        ax[1].set_title("CoMy")
        ax[2].imshow(np.sqrt(dpc._com_normalized_y**2 + dpc._com_normalized_x**2), cmap="inferno")
        ax[2].set_title("Magnitude of CoM")
        fig.tight_layout()
        plt.savefig(save_dir+"/CoM_image.png")


        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(dpc_cor._com_normalized_y, cmap="bwr")
        ax[0].set_title("CoMx - rotation corrected")
        ax[1].imshow(dpc_cor._com_normalized_x, cmap="bwr")
        ax[1].set_title("CoMy - rotation corrected")
        ax[2].imshow(np.sqrt(dpc_cor._com_normalized_y**2 + dpc_cor._com_normalized_x**2), cmap="inferno")
        ax[2].set_title("Magnitude of CoM - rotation corrected")
        fig.tight_layout()
        plt.savefig(save_dir+"/corrected_CoM_image.png")


        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(dpc.object_phase, cmap="inferno")
        ax[0].set_title("iCoM")
        ax[1].imshow(dpc_cor.object_phase, cmap="inferno")
        ax[1].set_title("iCoM - rotation corrected")
        fig.tight_layout()
        plt.savefig(save_dir+"/iDPC_comparison.png")
        # tifffile.imwrite(save_dir+"/iDPC_corrected.tif", dpc_cor.object_phase)

    if parallax_check:
        print("rebin by 2, implemented for parallax to reduce the memory usage")
        dataset.bin_Q(2)
        parallax = py4DSTEM.process.phase.Parallax(
            datacube=dataset,
            energy = HT,
            device = device,
            verbose = True
        ).preprocess(
            normalize_images=True,
            plot_average_bf=False,
            edge_blend=8,
        )

        parallax = parallax.reconstruct(
            reset=True,
            regularizer_matrix_size=(1,1),
            regularize_shifts=True,
            running_average=True,
            min_alignment_bin = 16,
            num_iter_at_min_bin = 2,
        )
        plt.savefig(save_dir+"/parallax_rough.png")

        parallax.show_shifts()
        plt.savefig(save_dir+"/parallax_shift.png")

        parallax.subpixel_alignment(
            #kde_upsample_factor=2,
            kde_sigma_px=0.125,
            plot_upsampled_BF_comparison=True,
            plot_upsampled_FFT_comparison=True,
        )
        plt.savefig(save_dir+"/parallax_subpixel_alignment.png")

        parallax.aberration_fit(
            plot_CTF_comparison=True,
        )
        plt.savefig(save_dir+"/parallax_CTF.png")

        parallax.aberration_correct(figsize=(5, 5))
        plt.savefig(save_dir+"/parallax_aberration_corrected.png")

        # Get the probe convergence semiangle from the pixel size and estimated radius in pixels
        semiangle_cutoff_estimated = dataset.calibration.get_Q_pixel_size() * probe_radius_pixels
        print('semiangle cutoff estimate = ' + str(np.round(semiangle_cutoff_estimated, decimals=1)) + ' mrads')

        # Get the estimated defocus from the parallax reconstruction - note that defocus dF has the opposite sign as the C1 aberration!
        defocus_estimated = -parallax.aberration_C1
        print('estimated defocus         = ' + str(np.round(defocus_estimated)) + ' Angstroms')

        rotation_degrees_estimated = np.rad2deg(parallax.rotation_Q_to_R_rads)
        print('estimated rotation        = ' + str(np.round(rotation_degrees_estimated)) + ' deg')

        # with open(save_dir+"/parallax_estimates.txt", 'w') as fp:
        #     fp.write('semiangle cutoff estimate = ' + str(np.round(semiangle_cutoff_estimated, decimals=1)) + ' mrads\n')
        #     fp.write('estimated defocus         = ' + str(np.round(defocus_estimated)) + ' Angstroms\n')
        #     fp.write('estimated rotation        = ' + str(np.round(rotation_degrees_estimated)) + ' deg')

    with h5py.File(save_dir+"/"+time_stamp+"_py4DSTEM_processed.hdf5", 'w') as vi_save:
        vi_save.create_dataset('data_path', data=data_path)
        vi_save.create_dataset('defocus(nm)', data=defocus_exp)
        vi_save.create_dataset('ht_value(V)', data=acc)
        vi_save.create_dataset('nominal_camera_length(m)', data=nCL)
        vi_save.create_dataset('aperture_size', data=aps)
        vi_save.create_dataset('convergence_angle(mrad)', data=semiangle)
        vi_save.create_dataset('pixel_size(Å)', data=scan_step)
        vi_save.create_dataset('vBF', data=vBF)
        vi_save.create_dataset('vADF', data=vADF)
        if eval(info["DPC"]):
            vi_save.create_dataset('CoMx', data=dpc._com_normalized_y)
            vi_save.create_dataset('CoMy', data=dpc._com_normalized_x)
            vi_save.create_dataset('iDPC', data=dpc_cor.object_phase)

        if eval(info["parallax"]):
            vi_save.create_dataset('parallax', data=parallax._recon_phase_corrected)
            vi_save.create_dataset('parallax_estimated_defocus(Å)', data=defocus_estimated)
            vi_save.create_dataset('parallax_estimated_rotation(deg)', data=rotation_degrees_estimated)


if create_json:
    pty_dest = save_path + '/pty_out'
    pty_dest_2 = save_path + '/pty_out/initial_recon'

    try:
        os.makedirs(pty_dest)
    except:
        print('skipping this folder as it already has pty_out folder')
    try:
        os.makedirs(pty_dest_2)
    except:
        print('skipping this folder as it already has pty_out/initial folder')

    with h5py.File(meta_path, 'r') as microscope_meta:
        meta_values = microscope_meta['metadata']
        print(meta_values['aperture_size'][()])
        print(meta_values['nominal_camera_length(m)'][()])
        print(meta_values['ht_value(V)'][()])
        acc = meta_values['ht_value(V)'][()]
        nCL = meta_values['nominal_camera_length(m)'][()]
        aps = meta_values['aperture_size'][()]
    rot_angle,camera_length,conv_angle = Meta2Config(acc, nCL, aps)

    if ptycho_config == '':
        config_name = 'pty_recon'
    else:
        config_name = ptycho_config

    if ptycho_template == '':
        template_path = '/dls/science/groups/imaging/ePSIC_ptychography/experimental_data/User_example/UserExampleJson.json'
    else:
        template_path = ptycho_template

    gen_config(template_path, pty_dest_2, config_name, save_path +'/'+time_stamp+'.hdf', rot_angle, camera_length, 2*conv_angle)
