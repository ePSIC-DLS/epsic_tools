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
sys.path.append('/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts')
from MIB_convert import *
from MIB_convert import _add_crosses
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