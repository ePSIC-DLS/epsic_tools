#!/usr/bin/env bash
import hyperspy.api as hs
print(f"hyperspy version: {hs.__version__}")
import pyxem as pxm
print(f"pyxem version: {pxm.__version__}")
import numpy as np
import py4DSTEM
print(f"py4DSTEM version: {py4DSTEM.__version__}")
import dask.array as da
from rsciio.quantumdetector import load_mib_data, parse_exposures
from rsciio.quantumdetector import file_reader
import h5py
import shutil

import ipywidgets
import ipywidgets as widgets
import sys
import os
import glob
import logging
import subprocess

formatter = logging.Formatter("%(asctime)s    %(process)5d %(processName)-12s %(threadName)-12s                   %(levelname)-8s %(pathname)s:%(lineno)d %(message)s")
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set the debug log level.
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)

# Make a logger for this module.
logger = logging.getLogger(__name__)


###########################################################################################################
#############################################   Functions - Start   #######################################
###########################################################################################################

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

def reshape_4DSTEM_FlyBack(data):
    """Reshapes the lazy-imported frame stack to navigation dimensions determined
    based on stored exposure times.

    Parameters
    ----------
    data : pyxem / hyperspy lazy Signal2D
        Lazy loaded electron diffraction data: <framenumbers | det_size, det_size>
        the data metadata contains flyback info as:
            ├── General
        │   └── title =
        └── Signal
            ├── binned = False
            ├── exposure_time = 0.001
            ├── flyback_times = [0.01826, 0.066, 0.065]
            ├── frames_number_skipped = 68
            ├── scan_X = 256
            └── signal_type = STEM

    Returns
    -------
    data_skip : pyxem.signals.LazyElectronDiffraction2D
        Reshaped electron diffraction data <scan_x, scan_y | det_size, det_size>
    """
    from math import floor
    # Get detector size in pixels
    # detector size in pixels
    det_size = data.axes_manager[1].size
    # Read metadata
    skip_ind = data.metadata.Signal.frames_number_skipped
    line_len = data.metadata.Signal.scan_X

    n_lines = floor((data.data.shape[0] - skip_ind) / line_len)

    # Remove skipped frames
    data_skip = data.inav[skip_ind : skip_ind + (n_lines * line_len)]
    # Reshape signal
    data_skip.data = data_skip.data.reshape(n_lines, line_len, det_size, det_size)
    data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
    data_skip.get_dimensions_from_data()
    # Cropping the bright fly-back pixel
    data_skip = data_skip.inav[1:]

    return data_skip

def change_dtype(d):
    """
    Changes the data type of hs object d to uint8
    Parameters:
    -----------
    d : hyperspy.signals.Signal2D
        Signal2D object with dtype float64

    Returns
    -------
    d : hyperspy.signals.Signal2D
        Signal2D object following dtype change to uint8
    """
    d = d.data.astype('uint8')
    d = hs.signals.Signal2D(d)

    return d

def bin_sig(d, bin_fact):
    """
    bins the reshaped 4DSTEMhs object by bin_fact on signal (diffraction) plane
    Parameters:
    ------------
    d: hyperspy.signals.Signal2D - can also be lazy
        reshaped to scanX, scanY | DetX, DetY
        This needs to be computed, i.e. not lazy, to work. If lazy, and binning 
        not aligned with dask chunks raises ValueError 
    Returns:
    -------------
    d_sigbin: binned d in the signal plane
    """
    # figuring out how many pixles to crop before binning
    # we assume the Detx and DetY dimensions are the same
    to_crop = d.axes_manager[-1].size % bin_fact
    d_crop = d.isig[to_crop:,to_crop:]
    try:
        d_sigbin = d_crop.rebin(scale=(1,1,bin_fact,bin_fact), dtype=np.uint16)
    except ValueError:
        logger.debug('Rebinning does not align with data dask chunks. Pass non-lazy signal before binning.')
        return
    return d_sigbin

def bin_nav(d, bin_fact):
    """
    bins the reshaped 4DSTEMhs object by bin_fact on navigation (probe scan) plane
    Parameters:
    ------------
    d: hyperspy.signals.Signal2D - can also be lazy
        reshaped to scanX, scanY | DetX, DetY
        This needs to be computed, i.e. not lazy, to work. If lazy, and binning 
        not aligned with dask chunks raises ValueError 
    Returns:
    -------------
    d_navbin: binned d in the signal plane
    """
    # figuring out how many pixles to crop before binning
    # we assume the Detx and DetY dimensions are the same
    to_cropx = d.axes_manager[0].size % bin_fact
    to_cropy = d.axes_manager[1].size % bin_fact
    d_crop = d.inav[to_cropx:,to_cropy:]
    try:
        d_navbin = d_crop.rebin(scale=(bin_fact,bin_fact,1,1), dtype=np.uint16)
    except ValueError:
        logger.debug('Rebinning does not align with data dask chunks. Pass non-lazy signal before binning.')
        return
    return d_navbin

def max_contrast8(d):
    """Rescales contrast of hyperspy Signal2D to 8-bit.
    Parameters
    ----------
    d : hyperspy.signals.Signal2D
        Signal2D object to be rescaled
    Returns
    -------
    d : hyperspy.signals.Signal2D
        Signal2D object following intensity rescaling
    """
    data = d.data
    data = data - data.min()
    if data.max() != 0:
        data = data * (255 / data.max())
    d.data = data
    return d
    
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
            f['metadata']['4D_shape'] = tuple(sh)
        
    return

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

path = eval(info['to_convert_paths'][0])[index]

adr_split = path.split('/')
tmp_save = []
tmp_save.append('/')
tmp_save.extend(adr_split[1:6])
tmp_save.append('processing')
tmp_save.extend(adr_split[6:8])
save_dir = os.path.join(*tmp_save)

# Load data as stack

src_path = path[:-40]

time_stamp = path.split('/')[-1][:15]
save_path = os.path.join(save_dir, time_stamp)
if not os.path.exists(save_path):
     os.makedirs(save_path)
    
data = file_reader(path, lazy = False)
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
reshape = eval(info['reshape'])
add_cross = eval(info['add_cross'])

data_memmap, headers = load_mib_data(path, return_headers=True, return_mmap=True)
# Adding cross
if add_cross == 1:
    data = _add_crosses(data[0]['data'])

else:
    data = data[0]['data']


if no_reshaping:
    # Saving data as stack
    data = pxm.signals.ElectronDiffraction2D(data)
    data.metadata.Signal.flip = "True"
    logger.debug(f'data shape: {data.data.shape}')
    data.save(os.path.join(save_path, f'{time_stamp}_data.zspy'), overwrite = True)
    data.save(os.path.join(save_path, f'{time_stamp}_data.hdf5'), overwrite = True, file_format = 'HSPY')
    
    
elif use_fly_back:
    timestamps = parse_exposures(headers, max_index=-1)
    data_dict = STEM_flag_dict(timestamps)
    print(type(data))
    data = pxm.signals.ElectronDiffraction2D(data)
    data.metadata.Signal.flip = "True"
    print(type(data))
    # Transferring dict info to metadata
    if data_dict["STEM_flag"] == 1:
        data.metadata.Signal.signal_type = "STEM"
    else:
        data.metadata.Signal.signal_type = "TEM"
    data.metadata.Signal.scan_X = data_dict["scan_X"]
    data.metadata.Signal.exposure_time = data_dict["exposure time"]
    data.metadata.Signal.frames_number_skipped = data_dict[
        "number of frames_to_skip"
    ]
    data.metadata.Signal.flyback_times = data_dict["flyback_times"]
    # only attempt reshaping if it is not already reshaped!
    if len(data.data.shape) == 3:
        # try:
        if data.metadata.Signal.signal_type == "TEM":
            print(
                "This mib file appears to be TEM data. The stack is returned with no reshaping."
            )

        print("reshaping using flyback pixel")
        data = reshape_4DSTEM_FlyBack(data)
        logger.debug(f'data shape: {data.data.shape}')
        data.save(os.path.join(save_path, f'{time_stamp}_data.zspy'), overwrite = True)
        data.save(os.path.join(save_path, f'{time_stamp}_data.hdf5'), overwrite = True, file_format = 'HSPY')
        if iBF == 1:
            ibf = data.sum(axis=data.axes_manager.signal_axes)
            # Rescale contrast of IBF image
            ibf = max_contrast8(ibf)
            ibf = change_dtype(ibf)
            ibf.save(os.path.join(save_path, f'{time_stamp}_iBF'), overwrite = True, extension = 'jpg')
        if bin_sig_flag == 1:
            data_bin_sig = bin_sig(data, bin_sig_factor)
            data_bin_sig.save(os.path.join(save_path, f'{time_stamp}_data_bin_sig_factor_{bin_sig_factor}.hspy'), overwrite = True)
        if bin_nav_flag == 1:
            data_bin_sig = bin_sig(data, bin_nav_factor)
            data_bin_sig.save(os.path.join(save_path, f'{time_stamp}_data_bin_nav_factor_{bin_nav_factor}.hspy'), overwrite = True)

        meta_path = find_metadat_file(time_stamp, src_path)
        logger.debug(f'metadata path: {meta_path}')
        write_vds(save_path + '/' + time_stamp + '_data.hdf5', save_path + '/' + time_stamp + '_vds.h5', metadata_path=meta_path)

elif known_shape:
    try:
        data = np.reshape(data, newshape=(Scan_X,Scan_Y,515,515))
        data = pxm.signals.ElectronDiffraction2D(data)
        data.metadata.Signal.flip = "True"
        logger.debug(f'data shape: {data.data.shape}')
        data.save(os.path.join(save_path, f'{time_stamp}_data.zspy'), overwrite = True)
        data.save(os.path.join(save_path, f'{time_stamp}_data.hdf5'), overwrite = True, file_format = 'HSPY')

        if iBF == 1:
            ibf = data.sum(axis=data.axes_manager.signal_axes)
            # Rescale contrast of IBF image
            ibf = max_contrast8(ibf)
            ibf = change_dtype(ibf)
            ibf.save(os.path.join(save_path, f'{time_stamp}_iBF'), overwrite = True, extension = 'jpg')
        if bin_sig_flag == 1:
            data_bin_sig = bin_sig(data, bin_sig_factor)
            data_bin_sig.save(os.path.join(save_path, f'{time_stamp}_data_bin_sig_factor_{bin_sig_factor}.hspy'), overwrite = True)
        if bin_nav_flag == 1:
            data_bin_sig = bin_sig(data, bin_nav_factor)
            data_bin_sig.save(os.path.join(save_path, f'{time_stamp}_data_bin_nav_factor_{bin_nav_factor}.hspy'), overwrite = True)

        meta_path = find_metadat_file(time_stamp, src_path)
        logger.debug(f'metadata path: {meta_path}')
        write_vds(save_path + '/' + time_stamp + '_data.hdf5', save_path + '/' + time_stamp + '_vds.h5', metadata_path=meta_path)

        
    except ValueError:
        logger.debug(f'Could not reshape the data to the requested scan dimensions. Original shape: {data.shape}')
else:
    logger.debug('You have to select one of the actions on reshaping!!!')