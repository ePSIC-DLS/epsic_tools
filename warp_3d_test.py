# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:27:15 2019
Example of correcting for post specimen abberations.
The procedure is:
    Fitting an ellipse to the bright field disk of calculated PACBED
    Calculating the co-ordinate transform to return to circular
    Applying transform to full data-set lazily
@author: gys37319
"""

import hyperspy.api as hs
import warp_3d as wp
import matplotlib.pyplot as plt
from skimage import transform as tf
import scipy as sc
import time

#%%
#load data
pn = r'Y:\2019\mg22549-3\processing\Merlin\20190706 MoS2 700C\20190706 153407'
fn4d = r'\30M_25cmCL.hdf5'

d_4d = hs.load(pn+fn4d, lazy = True)
da_4d_cut = d_4d.data[100:200, 100:200, :,:]

#compute PACBED of cut data
da_4d_sum = da_4d_cut.sum(axis = (0,1))
d= hs.signals.Signal2D(da_4d_sum.compute())
#d.plot()
#%%
#need flat_field and mask dead px here. 
#get rid of hot pixels
d.data[d.data > 20* d.data.mean()] = d.data.mean()
d.plot()

#%%
#fit ellipse and transform PACBED
d.data = wp.remove_cross(d.data)
params = wp.fit_ellipse(d.data, threshold = 0.15, plot_me = True)
print('orignal params : ', params)
transform = wp.compute_transform(d.data, params)
dst = tf.warp(d.data, transform, order=1)

plt.figure()
plt.imshow(dst)
plt.figure()
plt.imshow(d.data)

params_2 = wp.fit_ellipse(dst, threshold = 0.1, plot_me = True)
print ('final params : ', params_2)


#%%
#apply to entire (cut) data_set
#get coordinates of transform
coords =  tf.warp_coords(transform, (da_4d_cut.shape[-2], da_4d_cut.shape[-1]))
t0 = time.time()
#warp data

warped_data= da_4d_cut.map_blocks(wp.warp_all_np, dtype = 'float32', coords = coords)

#save to hdf5
wf = fn4d[:-5] + '_warp.hdf5'
warped_data.to_hdf5(pn +wf,'data', compression = 'gzip')
print(time.time() - t0)
print(pn +wf)
#%%
#check fit
new_sum = warped_data.sum(axis= (0,1))
wp.plot_ellipse(new_sum, params_2)
#check CoM
print('old CoM ; ' , sc.ndimage.measurements.center_of_mass(da_4d_cut[0,0,:,:].compute()))
print('new CoM : ', sc.ndimage.measurements.center_of_mass(warped_data[0,0,:,:].compute()))
