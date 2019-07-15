import hyperspy.api as hs
import numpy as np
from math import floor
import os
import time

#%%


def reshape_4DSTEM_FrameSize(data, scan_x, scan_y):
    """
    Reshapes the lazy-imported stack of dimensions: (xxxxxx|det_size, det_size) to the correct scan pattern shape:
    It gets the scan pattern dimensions from the user as input and reshapes from the end of the stack
    to start to avoid having the fly-back falling in the centre.
    
    NB: The fly-back can be included in the reshaped frame!
    
    Inputs:
        data: hyperspy lazily imported mib file with dimensions of: framenumbers|256, 256
        scan_x: number of lines
        scan_y: number of probe positions in every line
    Outputs:
        data_reshaped : reshaped data (scan_x, scan_y | det_size, det_size)
    """
    det_size = data[0].axes_manager[1].size  #detector size in pixels
    frames_total = scan_x * scan_y
    if frames_total <= data[0].axes_manager[0].size:
        skip = data[0].axes_manager[0].size - frames_total
        print(skip)
        data_skip = data[0].inav[skip:]
        data_skip.data = data_skip.data.reshape(scan_x, scan_y, det_size, det_size)
        data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
        data_skip.get_dimensions_from_data()  # reshaped
    else:
        print('================================================================')
        print('Total number of frames is less then the scan_x*scan_y provided.')
        print('Retuning the stack without reshaping.')
        print('================================================================')

        data_skip = data[0]
    return data_skip

#%%
    
def reshape_4DSTEM_FlyBack(data):
    """
    Reshapes the lazy-imported stack of dimensions: (xxxxxx|det_size, det_size) to the correct scan pattern 
    shape: (x, y | 256,256).
    It gets the number of frames ti skip from STEM_flag_dict function in mib_dask_import.py.
    
    Parameters
    ----------
    data : tuple of hyperspy lazily imported mib file with diensions of: 
        framenumbers|det_size, det_size and the dict containing STEM_flag, exposures, etc
    
    Returns
    -------
    data_skip : reshaped data (x, y | det_size, det_size)
    """
    det_size = data[0].axes_manager[1].size  #detector size in pixels

    n_lines = floor((data[0].data.shape[0] - data[1].get('number of frames_to_skip')) / data[1].get('scan_X'))
    skip_ind = data[1].get('number of frames_to_skip')
    line_len = data[1].get('scan_X')
    data_skip = data[0].inav[skip_ind:skip_ind + (n_lines * line_len)]  # with the skipped frames removed
    data_skip.data = data_skip.data.reshape(n_lines, line_len, det_size, det_size)
    data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
    data_skip.get_dimensions_from_data()  # reshaped
    # print('Number of frames skipped at the beginning: ', skip_ind)
    data_skip = data_skip.inav[1:]
    return data_skip
