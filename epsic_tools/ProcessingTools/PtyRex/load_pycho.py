# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:02:32 2019

@author: gys37319
"""

import numpy as np
import h5py 
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.measurements import center_of_mass
import os
import glob
#%%
#filename of output from ptycho reconstruction
#fn =r'Y:\2019\cm22979-6\processing\Merlin\20190822_MoS2_15kV_Ptycho\processing\pycho\rotation_tests2\_neg29_20190917-153316.hdf'

#first attempt: 
#r'Y:/2019/cm22979-6/processing/Merlin/20190822_MoS2_15kV_Ptycho/processing/pycho/001_20190912-123437.hdf'
#cropped 128x128 with 0.608Ang step size:
# r'Y:/2019/cm22979-6/processing/Merlin/20190822_MoS2_15kV_Ptycho/processing/pycho/001_20190912-165127.hdf'
#masked to 48 mrad:
#r'Y:/2019/cm22979-6/processing/Merlin/20190822_MoS2_15kV_Ptycho/processing/pycho/001_20190913-120459.hdf'
#masked to 68 mrad (after 200 itterations):
#r'Y:/2019/cm22979-6/processing/Merlin/20190822_MoS2_15kV_Ptycho/processing/pycho/001_20190913-152703.hdf'
#%%
#choose file in directory 
file_number =13

pn =r'Y:\2019\cm22979-6\processing\Merlin\Merlin\20191001_15kV_ptycho\processing\pycho' 
#r'Y:\2019\cm22979-6\processing\Merlin\20190822_MoS2_15kV_Ptycho\processing\pycho\rotation_tests2'
#r'Y:\2019\cm22979-6\processing\Merlin\20190822_MoS2_15kV_Ptycho\processing\pycho\Illumination_modes_test'
#r'Y:\2019\cm22979-6\processing\Merlin\20190822_MoS2_15kV_Ptycho\processing\pycho\rotation_tests2'
pn = pn +'\*.hdf'
#build list of files
fp_list = glob.glob(pn)
n = 0
for this_fp in fp_list:
    print('['+str(n)+']' , os.path.split(this_fp)[1])
    n = n+1
    
#print(fp_list)

fn = fp_list[file_number]#  = os.path.join(root, files[file_number])
print('')
print('['+str(file_number)+']', fn)
#%%
#create h5py file object
d5 = h5py.File(fn, 'r')
#get phase data
dat = d5['entry_1']['process_1']['output_1']['object_phase']
dat = dat[0,0,0,0,0,:,:]
#get modulus data
dat_m = d5['entry_1']['process_1']['output_1']['object_modulus']
dat_m  = dat_m[0,0,0,0,0,:,:]
#rotate
rot_angle = 89
dat = rotate(dat, rot_angle)
dat_m = rotate(dat_m, rot_angle)

#get probe
probe = np.array(d5['entry_1']['process_1']['output_1']['probe_phase'])
#sum seperate coherent modes
probe = probe[:,:,0,0,0,:,:].sum(axis = (0,1))

probe_m = np.array(d5['entry_1']['process_1']['output_1']['probe_modulus'])
probe_m = probe_m[:,:,0,0,0,:,:].sum(axis = (0,1))
#%%
#show phase image
fg1, ([ax1a, ax1b], [ax1c, ax1d]) = plt.subplots(2, 2, figsize = (16,16))
img_size  = 400
img_centre = dat.shape[0]/2
im_lims_x = [img_centre - img_size / 2, img_centre + img_size / 2,]
im_lims_y = im_lims_x[::-1]
#plot data
#plt.figure()
ax1a.imshow(dat, cmap = 'gray')
ax1a.set_xlim(im_lims_x)
ax1a.set_ylim(im_lims_y)
ax1a.set_title('phase, rotation ' + str(-rot_angle) + ' deg')

#show modulus image
ax1b.imshow(dat_m, cmap = 'gray')
ax1b.set_xlim(im_lims_x)
ax1b.set_ylim(im_lims_y)
ax1b.set_title('modulus, rotation ' + str(-rot_angle) + ' deg')

#show probe image
ax1c.imshow(probe, cmap = 'gray')
#ax1c.set_xlim(im_lims_x)
#ax1c.set_ylim(im_lims_y)
ax1c.set_title('probe phase, rotation ' + str(-rot_angle) + ' deg')

#show modulus image
ax1d.imshow(probe_m, cmap = 'gray')
#ax1d.set_xlim(im_lims_x)
#ax1d.set_ylim(im_lims_y)
ax1d.set_title('probe modulus, rotation ' + str(-rot_angle) + ' deg')

#%% fft data
dat_fft = np.fft.fft2(dat)
dat_fft = np.fft.fftshift(dat_fft)
#%%
#show fft
plt.figure()
plt.imshow(np.log10(np.real(dat_fft)**2), cmap = 'gray')#**0.125)#, vmin = 0, vmax = 1e3)
plt.title('log power spectrum, rotation ' + str(rot_angle) + 'deg')
#%%
#filter 50hz noise
def r_mask(dp_bin, r):
    x0, y0 = center_of_mass(dp_bin)
    dx, dy = dp_bin.shape
    qy,qx = np.meshgrid(np.arange(dx),np.arange(dy))
    qr = np.hypot(qx - x0, qy - y0)
    mask = qr > r
    return mask
mask_high = r_mask(np.real(dat_fft)**2,50)
#mask_50 = 
#%%
#show_mask
plt.figure()
plt.imshow(mask_high)
#%%
#filtered fft
dat_fft_filtered_high = dat_fft * mask_high.T
#and invert
dat_filtered_high = np.fft.ifft2(np.fft.ifftshift(dat_fft_filtered_high))
#%%
#show filtered fft and image
plt.figure()
plt.imshow((np.real(dat_fft_filtered_high)**2)**0.25)#, vmin = 0, vmax = 1e3)
plt.figure()
plt.imshow(np.real(dat_filtered_high))
plt.xlim(im_lims_x)
plt.ylim(im_lims_y)
#%% 
#low pass filter
#make mask
mask_low = r_mask(np.real(dat_fft)**2,160)
mask_low = np.invert(mask_low.astype(bool))
#%%
#show mask
plt.figure()
plt.imshow(mask_low)
#%%
#filter fft and image
dat_fft_filtered_low = dat_fft * mask_high * mask_low
dat_filtered_low = np.fft.ifft2(np.fft.ifftshift(dat_fft_filtered_low))
#%%
#show filtered fft and image
plt.figure()
plt.imshow((np.real(dat_fft_filtered_low)**2)**0.25)#, vmin = 0, vmax = 1e3)
plt.figure()
plt.imshow(np.real(dat_filtered_low))
plt.xlim(im_lims_x)
plt.ylim(im_lims_y)
#%%
#attempt to rind rotation using similar algorithm as in py4DSTEM



