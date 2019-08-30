import hyperspy.api as hs
import numpy as np
from math import floor
import os
import time



def reshape_4DSTEM_FrameSize(data, scan_x, scan_y):
    """Reshapes the lazy-imported frame stack to user specified navigation dimensions.
 
    Excess frames are discarded from start of the stack to avoid having the fly-back
    in the center of the navigation space.
                                                           
    
    Parameters
    ----------
    data : pyxem.signals.LazyElectronDiffraction2D
        Lazy loaded electron diffraction data: <framenumbers | det_size, det_size>
    scan_x : int
        Number of probe positions in the slow scan direction.
    scan_y : int
        Number of probe positions in the fast scan direction.
    
    Returns
    -------
    data_skip : pyxem.signals.LazyElectronDiffraction2D
        Reshaped electron diffraction data <scan_x, scan_y | det_size, det_size>
    """
    # Get detector size in pixels
    det_size = data.axes_manager[1].size  #detector size in pixels
    frames_total = scan_x * scan_y
    # Check the expected frames_total less than number of recorded frames
    if frames_total <= data.axes_manager[0].size:
        # Skip excess frames
        skip = data.axes_manager[0].size - frames_total
        print(skip)
        data_skip = data.inav[skip:]
        # Reshape data
        data_skip.data = data_skip.data.reshape(scan_x, scan_y, det_size, det_size)
        data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
        data_skip.get_dimensions_from_data()  # reshaped
    else:
        raise ValueError('Total number of frames is less then the scan_x*scan_y provided. '
                         'Retuning the stack without reshaping.')


    return data_skip

    
def reshape_4DSTEM_FlyBack(data, STEM_flag_dict):
    """Reshapes the lazy-imported frame stack to navigation dimensions determined
    based on stored exposure times.
       
    
    Parameters
    ----------
    data : pyxem.signals.LazyElectronDiffraction2D
        Lazy loaded electron diffraction data: <framenumbers | det_size, det_size>
    STEM_flag_dict : dict
        Dictionary containing STEM_flag, exposures
    
    Returns
    -------
    data_skip : pyxem.signals.LazyElectronDiffraction2D
        Reshaped electron diffraction data <scan_x, scan_y | det_size, det_size>
    """
    # Get detector size in pixels
                              
    det_size = data.axes_manager[1].size  #detector size in pixels
    # Determine frames to skip
    n_lines = floor((data.data.shape[0] - STEM_flag_dict.get('number of frames_to_skip')) / STEM_flag_dict.get('scan_X'))
    skip_ind = STEM_flag_dict.get('number of frames_to_skip')
    line_len = STEM_flag_dict.get('scan_X')
    # Remove skipped frames
    data_skip = data.inav[skip_ind:skip_ind + (n_lines * line_len)]  
    # Reshape signal
    data_skip.data = data_skip.data.reshape(n_lines, line_len, det_size, det_size)
    data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
    data_skip.get_dimensions_from_data()
    # Cropping the bright fly-back pixel
    data_skip = data_skip.inav[1:]
 
    return data_skip
