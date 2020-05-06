# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:16:51 2019

@author: gys37319
"""
import numpy  as np
import matplotlib.pyplot as plt

import hyperspy.api as hs
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


fn = r'Y:/2019/cm22979-8/processing/Merlin/20191114_15kVptycho_graphene/20191114 111900/binned_diff_skipscan_191114_15keV_graphene_15Mx_100um_scan_array_64by64_diff_plane_128by128_.hdf5'
d = hs.load(fn, lazy = True)
d_sum = d.sum(axis = (0,1))
d_sum.compute()
PACBED = hs.signals.Signal2D(data = d_sum.data **0.1)
plt.figure()
plt.imshow(PACBED)
#%%
image = PACBED.data
image = (image / image.max())
edges = canny(image, sigma =1.5, low_threshold=0.001, high_threshold=0.03)


hough_radii = np.arange(10, 20,1)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii  = hough_circle_peaks( hough_res, hough_radii, total_num_peaks = 7)

fig, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1, figsize = (10,4))
image= color.gray2rgb(image)

for center_y, center_x, radius in zip(cy,cx,radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape = image.shape)
    image[circy,circx] = (220,0,0)


ax1.imshow(image, cmap = plt.cm.gray)
ax2.imshow(edges)
plt.show()
#%%
oval_points =[cx[1:], cy[1:]]
