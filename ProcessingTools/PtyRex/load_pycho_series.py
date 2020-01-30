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
from scipy import fftpack
import json
import hyperspy.api as hs

#%%
#directory path
pn = r'Y:\2019\cm22979-6\processing\Merlin\Merlin\20191001_15kV_ptycho\cluster\processing\pycho'
#'Y:\2019\cm22979-6\processing\Merlin\Merlin\20191001_15kV_ptycho\MoS2_700C\scan_step_refine\processing\pycho'
#Y:\2019\cm22979-6\processing\Merlin\Merlin\20191001_15kV_ptycho\MoS2_700C\rot_step_refine\processing\pycho'

pn = pn +'\*.hdf'
#pn = pn + '\*.json'
#build list of files
fp_list = glob.glob(pn)
len_dat = len(fp_list)
n = 0 
crop_to = 300
for this_fp in fp_list: 
    fj = this_fp[:-4] + '.json'
    with open(fj) as r:
        params = json.load(r)
    with h5py.File(this_fp, 'r') as d5:
        #get phase data
        dat = d5['entry_1']['process_1']['output_1']['object_phase']
        dat = dat[0,0,0,0,0,:,:]
        #get modulus data
        dat_m = d5['entry_1']['process_1']['output_1']['object_modulus']
        
        dat_m  = dat_m[0,0,0,0,0,:,:]
        #rotate
        rot_angle = 90-params['process']['common']['scan']['rotation']
        dat = rotate(dat, rot_angle)
        dat_m = rotate(dat_m, rot_angle)
        
        #get probe
        probe = np.array(d5['entry_1']['process_1']['output_1']['probe_phase'])
        #sum seperate coherent modes
        probe = probe[:,:,0,0,0,:,:].sum(axis = (0,1))
        
        probe_m = np.array(d5['entry_1']['process_1']['output_1']['probe_modulus'])
        probe_m = probe_m[:,:,0,0,0,:,:].sum(axis = (0,1))
        
        #get complex probe
        probe_c = np.array(d5['entry_1']['process_1']['output_1']['probe'])
        probe_c = probe_c[:,:,0,0,0,:,:].sum(axis = (0,1))
        
        #get error plot
        error = np.array(d5['entry_1']['process_1']['output_1']['error'])
        error = error[error != 0]
        d5.close()
    if n == 0:
        #initiate arrays if first itteration - make bigger than fist data to account for change in size
        shape_dat = int(10 * (np.ceil(dat.shape[0]/10) +1))
        shape_probe = int(10 * (np.ceil(probe.shape[0]/ 10)+1))
        #print(n, shape_dat, shape_probe)
        print(n,len_dat, shape_dat, shape_dat)
        dat_arr = np.empty(shape = (2,len_dat, shape_dat, shape_dat))
        probe_arr = np.empty(shape = (2, len_dat, shape_probe, shape_probe))
        rot_arr = np.empty(shape = len_dat)
        err_arr = np.empty(shape = len_dat)
    #print(n,len_dat, shape_dat)
    dat_diff =shape_dat - dat.shape[0] 
    pad_dat = int(np.ceil(dat_diff / 2))
    del_dat = int(pad_dat - np.floor(dat_diff / 2))
    print(n,len_dat, dat.shape[0] , shape_dat, del_dat, pad_dat)
    probe_diff =shape_probe - probe.shape[0] 
    pad_probe = int(np.ceil(probe_diff / 2))
    del_probe = int(pad_probe - np.floor(probe_diff / 2))
    #print(n, shape_dat, shape_probe, dat.shape, probe.shape, pad_dat)
    if pad_dat > 0:
        dat_arr[0,n,:,:] = np.pad(dat[del_dat:, del_dat:], pad_dat, 'edge') #object phase
        dat_arr[1,n,:,:] = np.pad(dat_m[del_dat:, del_dat:], pad_dat, 'edge')  #object mod   
    else:
        start_ind= int(-np.floor(dat_diff/2))
        end_ind= int(np.ceil(dat_diff/2))    
        if end_ind == 0:
             end_ind = dat.shape[0] 
        dat_arr[0,n,:,:] = dat[start_ind:end_ind, start_ind:end_ind] #object phase
        dat_arr[1,n,:,:] = dat_m[start_ind:end_ind, start_ind:end_ind]  #object mod  
        
    if pad_probe >0:
        probe_arr[0,n,:,:] = np.pad(probe[del_probe:, del_probe:], pad_probe, 'edge') #probe phase
        probe_arr[1,n,:,:] = np.pad(probe_m[del_probe:, del_probe:], pad_probe, 'edge') #probe mod
    else:
        probe_start_ind= int(-np.floor(probe_diff/2))
        probe_end_ind= int(np.ceil(probe_diff/2))
        if probe_end_ind == 0:
            probe_end_ind = probe.shape[0]
        probe_arr[0,n,:,:] = probe[probe_start_ind:probe_end_ind, probe_start_ind:probe_end_ind] #probe phase
        probe_arr[1,n,:,:] = probe_m[probe_start_ind:probe_end_ind, probe_start_ind:probe_end_ind] #probe mod
    
    rot_arr[n] = float(params['process']['common']['scan']['rotation'])#float(params['process']['common']['scan']['dR'][0])
    err_arr[n] = error[-1]
    n = n+1

sort_ind = np.argsort(rot_arr)
rot_sort = rot_arr[sort_ind]
err_sort = err_arr[sort_ind]
dat_sort =  dat_arr[:,sort_ind,:,:]
probe_sort = probe_arr[:,sort_ind,:,:]

d = hs.signals.Signal2D(data = dat_arr)
p = hs.signals.Signal2D(data = probe_arr)
r = hs.signals.Signal1D(data = rot_arr)
e = hs.signals.Signal1D(data = err_arr)

d_s = hs.signals.Signal2D(data = dat_sort)
p_s = hs.signals.Signal2D(data = probe_sort)
r_s = hs.signals.Signal1D(data = rot_sort)
e_s = hs.signals.Signal1D(data = err_sort)
#crop
d.crop(axis = (2), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
d.crop(axis = (3), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
d_s.crop(axis = (2), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
d_s.crop(axis = (3), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))

dat_fft = np.fft.fft2(d.data)
dat_fft = np.fft.fftshift(dat_fft)
d_fft = hs.signals.Signal2D(data = np.log(np.abs(dat_fft)))

dat_sort_fft = np.fft.fft2(d_s.data)
dat_sort_fft = np.fft.fftshift(dat_sort_fft)
d_s_fft = hs.signals.Signal2D(data = np.log(np.abs(dat_sort_fft)))

hs.plot.plot_signals([d_s,p_s,d_s_fft], navigator_list=[r_s, e_s, 'slider'])

##set positions
plt.figure(1)
plt.title('probe array rotation angle')
plt.xlabel('data set number')
plt.ylabel('rotation angle (deg)')
plt.tight_layout()
mngr = plt.get_current_fig_manager()
geom = mngr.window.geometry()
x,y,dx,dy = geom.getRect()

mngr.window.setGeometry(50,100,dx, dy)

plt.figure(2)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(700,100,dx, dy)

plt.figure(3)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(50,700,dx, dy)
plt.title('error')
plt.xlabel('data set number')
plt.ylabel('error')
plt.tight_layout()

plt.figure(4)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(700,700,dx, dy)

plt.figure(5)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(1350,100,dx, dy)




