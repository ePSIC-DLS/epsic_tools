import numpy as np
import hyperspy.api as hs
from scipy import interpolate

def make_uniform(sig, non_uni_vals, n_uni_points):
    tot_x, tot_y = sig.data.shape[0], sig.data.shape[1]
    new_arr = np.zeros((tot_x, tot_y, n_uni_points))
    n_interp_x = np.linspace(non_uni_vals[0], non_uni_vals[-1], n_uni_points)
    for x_pos in range(tot_x):
        for y_pos in range(tot_y):
            t_interp_y = sig.data[x_pos,y_pos]
            interp_func = interpolate.splrep(non_uni_vals,t_interp_y, s=0)
            new_arr[x_pos, y_pos] = interpolate.splev(n_interp_x, interp_func)
    new_sig = hs.signals.Signal1D(new_arr)
    offset = n_interp_x[0]
    new_sig.axes_manager[2].offset= n_interp_x[0]
    new_sig.axes_manager[2].scale = (n_interp_x[-1] - offset)/n_uni_points
    return new_sig, n_interp_x

def resample_data(x, y, new_x_range, sampling_resolution =100, return_x = False):
    
    '''
    x: original x data (np array)
    y: original y data at each x point (np array)
    new_x_range: new lower x bound and new higher x bound (2 value tuple)
    sampling resolution: number of points to contain in a step of value 1 along the x axis (float)
    return_x: whether to return the new x range values (bool)
    
    returns:
    
    either - new_y
    or - (new_x, new_y)
    
    '''

    xl, xu = new_x_range
    new_x_vals = np.linspace(xl, xu, (xu-xl)*sampling_resolution)
    
    if return_x == False:

        return interpolate.interp1d(x, y)(new_x_vals)
    
    if return_x == True:

        return new_x_vals, interpolate.interp1d(x, y)(new_x_vals)
