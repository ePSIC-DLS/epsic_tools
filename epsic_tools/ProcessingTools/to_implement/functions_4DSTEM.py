import hyperspy.api as hs
import numpy as np
import dask.array as da
import matplotlib as plt
from scipy import ndimage as ndi

def get_fft(im):
	fft_im = np.fft.fftshift(np.fft.fft2(im))
	plt.figure()
	plt.imshow(np.log10((abs(fft_im))))
	return fft_im

def get_bf_disk(dp, sub = 20, threshold = 0.05, plot_me = False):
    #returns radius, centre
    # Assuming data is a 4d array.
    # Compute frame average
    #dp = dp_bin
    dp_t = dp.inav[::sub, ::sub]
    dp_sum = dp_t.sum()
    #check if lazy data:
    if dp_t._lazy == True:
        #compute data 
        dp_sum.compute()
    #remove hot pixels
    dp_sum.data[dp_sum.data > 5 * dp_sum.data.mean()] = dp_sum.data.mean()
    dp_sum.plot()
    #estimade bfd area by thresholding 
    dp_mask = ndi.binary_closing(dp_sum.data>threshold*dp_sum.data.max(), iterations=2)
    #get radius
    bfd_r = np.sqrt(dp_mask.sum()/np.pi)
    #get centre from CoM
    bfd_c = c_o_m(dp_mask)
    #c_o_m returns dask array so need to compute
    bfd_c = bfd_c.compute()
    #if data has been binned
    scale_factor = dp.axes_manager[-1].scale
    bf_ap = hs.roi.CircleROI(bfd_c[0]* scale_factor, bfd_c[1] * scale_factor, bfd_r * scale_factor)
    
    if plot_me == True:
        dp_sum.plot()
        dp_bf = bf_ap.interactive(dp_sum, axes = dp_sum.axes_manager.signal_axes)
        
    return bf_ap

    
def get_bf_disk_interactive(dp,bfd_r = 50 , bfd_c = [125,125], sub = 20):
	# select the bright field disk location from sum diffraction pattern
	# use sun of one in every sub diffraction patterns
	dp_t = dp.inav[::sub, ::sub]
	dp_sum = dp_t.sum()
	dp_sum.plot()
	bf_ap = hs.roi.CircleROI(125, 125, 50)
	sum_bf= bf_ap.interactive(dp_sum, axes = dp_sum.axes_manager.signal_axes)
	return bf_ap
	
def get_adf(dp, bf_aperture, outer_r = 400, plot_me = False):
    #plot adf signal
    adf_aperture = hs.roi.CircleROI(bf_aperture.cx, bf_aperture.cy, outer_r, bf_aperture.r * 1.1)
    adf_dp = adf_aperture(dp.T)
    adf_dp = adf_dp.T
    if plot_me == True:
        adf_dp.plot()
    return adf_dp
	
def plot_abf(dp, bf_aperture, inner_r = 20):
	#plot abf singnal
	abf_aperture = hs.roi.CircleROI(bf_aperture.cx, bf_aperture.cy, bf_aperture.r,inner_r)
	abf_dp = abf_aperture(dp.T)
	abf_dp = abf_dp.T
	abf_dp.plot()
	return abf_dp
	
def dpc_quad(dp, bf_aperture, bin_factor = 4):
	#calculate dpc (quadrant detector)
	#bin_factor is by how much data has been binned

	#get data shape and calculate size of quadrants
	_ , _ , x_shape, y_shape = dp.data.shape
	x_shape = x_shape * bin_factor
	y_shape = y_shape * bin_factor
	dx1 =  bf_aperture.cx
	dx2 = x_shape - bf_aperture.cx
	dy1 = bf_aperture.cy
	dy2 = y_shape - bf_aperture.cy
	min_x = min(dx1,dx2)
	min_y = min(dy1,dy2)

	l = bf_aperture.cx - min_x
	r = bf_aperture.cx + min_x
	t = bf_aperture.cy - min_y
	b = bf_aperture.cy + min_y
	#left, top, right, bottom
	tl_ap = hs.roi.RectangularROI(l, t, bf_aperture.cx, bf_aperture.cy)
	tl_dp = tl_ap(dp.T)
	tr_ap = hs.roi.RectangularROI(bf_aperture.cx, t , r, bf_aperture.cy)
	tr_dp = tr_ap(dp.T)

	bl_ap = hs.roi.RectangularROI(l, bf_aperture.cy, bf_aperture.cx,b)
	bl_dp = bl_ap(dp.T)
	br_ap = hs.roi.RectangularROI(bf_aperture.cx,bf_aperture.cy ,r,b)
	br_dp = br_ap(dp.T)

	#cast as integer
	tr_dp.data =tr_dp.data.astype('int')
	tl_dp.data =tl_dp.data.astype('int')
	br_dp.data =br_dp.data.astype('int')
	bl_dp.data =bl_dp.data.astype('int')

	dpc_x = (tr_dp.T + br_dp.T) - (tl_dp.T + bl_dp.T)
	dpc_y = (tr_dp.T + tl_dp.T) - (br_dp.T + bl_dp.T)
	dpc_x.plot()
	dpc_y.plot()
	return dpc_x , dpc_y
	
	
def c_o_m(dp, threshold_value=None, mask_array=None):
    ## modifyed from pixstem - Magnus Nord 
    """Find center of mass of last two dimensions for a dask array.

    The center of mass can be calculated using a mask and threshold.

    Parameters
    ----------
    dask_array : Dask array
        Must have either 2, 3 or 4 dimensions.
    threshold_value : scalar, optional
    mask_array : NumPy array, optional
        Array with bool values. The True values will be masked
        (i.e. ignored). Must have the same shape as the two
        last dimensions in dask_array.

    Returns
    -------
    center_of_mask_dask_array : Dask array

    Examples
    --------
    >>> import dask.array as da
    >>> import pixstem.dask_tools as dt
    >>> data = da.random.random(
    ...     size=(64, 64, 128, 128), chunks=(16, 16, 128, 128))
    >>> output_dask = dt._center_of_mass_array(data)
    >>> output = output_dask.compute()

    Masking everything except the center of the image

    >>> mask_array = np.ones(shape=(128, 128), dtype=bool)
    >>> mask_array[64-10:64+10, 64-10:64+10] = False
    >>> output_dask = dt._center_of_mass_array(data, mask_array=mask_array)
    >>> output = output_dask.compute()

    Masking and thresholding

    >>> output_dask = dt._center_of_mass_array(
    ...     data, mask_array=mask_array, threshold_value=3)
    >>> output = output_dask.compute()

    """
    dask_array = dp.data
    det_shape = dask_array.shape[-2:]
    y_grad, x_grad = np.mgrid[0:det_shape[0], 0:det_shape[1]]
    y_grad, x_grad = y_grad.astype(np.float64), x_grad.astype(np.float64)
    sum_array = np.ones_like(x_grad)

    if mask_array is not None:
        if not mask_array.shape == det_shape:
            raise ValueError(
                    "mask_array ({0}) must have same shape as last two "
                    "dimensions of the dask_array ({1})".format(
                        mask_array.shape, det_shape))
        x_grad = x_grad * np.invert(mask_array)
        y_grad = y_grad * np.invert(mask_array)
        sum_array = sum_array * np.invert(mask_array)
    if threshold_value is not None:
        dask_array = da.threshold_array(
                dask_array, threshold_value=threshold_value,
                mask_array=mask_array)

    x_shift = dask_array * x_grad#, dtype=np.float64)
    y_shift = dask_array * y_grad#, dtype=np.float64)
    sum_array = dask_array * sum_array# dtype=np.float64)

    x_shift = da.sum(x_shift, axis=(-2, -1))#, dtype=np.float64)
    y_shift = da.sum(y_shift, axis=(-2, -1))#, dtype=np.float64)
    sum_array = da.sum(sum_array, axis=(-2, -1))#, dtype=np.float64)

    beam_shifts = da.stack((x_shift, y_shift))
    
    beam_shifts = beam_shifts[:] / sum_array#da.divide(beam_shifts[:], sum_array)#, dtype=np.float64)
    return beam_shifts
	

	
	

