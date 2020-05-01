
from math import floor

def reshape_4DSTEM_FrameSize(data, scan_x, scan_y):
    """Reshapes the lazy-imported frame stack to user specified navigation dimensions.
 
    Excess frames are discarded from start of the stack to avoid having the fly-back
    in the center of the navigation space.
                                                           
    
    Parameters
    ----------
    data : hyperspy lazy signal.Signal2D 
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

    
def reshape_4DSTEM_FlyBack(data):
    """Reshapes the lazy-imported frame stack to navigation dimensions determined
    based on stored exposure times.
       
    
    Parameters
    ----------
    data : hyperspy lazy Signal2D
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
    # Get detector size in pixels
                              
    det_size = data.axes_manager[1].size  #detector size in pixels
    # Read metadata
    skip_ind = data.metadata.Signal.frames_number_skipped
    line_len = data.metadata.Signal.scan_X
    
    n_lines = floor((data.data.shape[0] - skip_ind) / line_len)

    # Remove skipped frames
    data_skip = data.inav[skip_ind:skip_ind + (n_lines * line_len)]  
    # Reshape signal
    data_skip.data = data_skip.data.reshape(n_lines, line_len, det_size, det_size)
    data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
    data_skip.get_dimensions_from_data()
    # Cropping the bright fly-back pixel
    data_skip = data_skip.inav[1:]
 
    return data_skip
