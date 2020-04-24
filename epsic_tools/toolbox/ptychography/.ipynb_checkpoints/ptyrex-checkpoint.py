# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:02:32 2019

@author: gys37319
"""

import numpy as np
import h5py 
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from skimage.filters import gaussian
import glob
import json
import hyperspy.api as hs
from epsic_tools.toolbox import radial_profile

def load_series(pn,crop_to, sort_by = 'rot', blur = 0, verbose = False):
    ''' 
    loads all ptycho reconstructions in a folder and sorts by a parameter
    
    Parameters
    ----------
    
    pn : String pathname of folder 
    
    sort_by: 'rot' or 'step' parameter by which to sort the data
    
    crop_to: Int size of cropped object reconstruction 
    
    blur: Int to pass as sigma to gaussian blur object phase. 0 = no blur 
    
    verbose: Bool to print detailed output
    
    Returns
    -------
    
    d_s: hyperspy singnal2D object function
    
    p_s: hyperspy singnal2D probe function
    
    d_s_fft: hyperspy singnal2D fourier transfor of object function
    
    rad_fft: hyperspy singnal1D object radial profile of d_s_fft 
    
    r_s: hyperspy signal1D object scan rotation
    
    s_s: hyperspy signal1D object probe step size
    
    e_s: hyperspy signal1D object final error value 
    
    
    Example usage
    -------------
    from epsic_tools.toolbox.ptychography.load_pycho_series import load_series
    pn = r'Y:\2020\cm26481-1\processing\Merlin\20200130_80kV_graphene_600C_pty\cluster\processing\pycho'
    d_s, p_s, d_s_fft, rad_fft, r_s, s_s, e_s = load_series(pn,sort_by = 'rot', crop_to = 80)
    hs.plot.plot_signals([d_s,p_s,d_s_fft, rad_fft], navigator_list=[r_s,s_s, e_s,None])


    '''
    pn = pn +'/*.hdf'
    #build list of files
    fp_list = glob.glob(pn)
    len_dat = len(fp_list)
    n = 0 # counter
    # iterate through files
    print(pn)
    print(fp_list)
    for this_fp in fp_list: 
        fj = this_fp[:-4] + '.json'
        #open json file
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
            d5.close()  # probably not necessary but just in case!
        if n == 0:
            #initiate arrays if first itteration - make bigger than fist data to account for change in size
            shape_dat = int(10 * (np.ceil(dat.shape[0]/10) +1))
            shape_probe = int(10 * (np.ceil(probe.shape[0]/ 10)+1))
            
            if verbose == True: 
                print(n,len_dat, shape_dat, shape_dat)
            dat_arr = np.empty(shape = (2,len_dat, shape_dat, shape_dat))
            probe_arr = np.empty(shape = (2, len_dat, shape_probe, shape_probe))
            rot_arr = np.empty(shape = len_dat)
            step_arr =np.empty(shape = len_dat)
            err_arr = np.empty(shape = len_dat)
            
        # calculate parameters to pad loaded data to same size as initiated array if necessary
        dat_diff =shape_dat - dat.shape[0] 
        pad_dat = int(np.ceil(dat_diff / 2))
        del_dat = int(pad_dat - np.floor(dat_diff / 2))
        
        probe_diff =shape_probe - probe.shape[0] 
        pad_probe = int(np.ceil(probe_diff / 2))
        del_probe = int(pad_probe - np.floor(probe_diff / 2))
        
        # populate rotation, step and error array
        rot_arr[n] = float(params['process']['common']['scan']['rotation'])
        step_arr[n] = float(params['process']['common']['scan']['dR'][0])
        
        err_arr[n] = error[-1]
        
        if verbose == True:
            print(n, ' step : ', step_arr[n], ', rot : ', rot_arr[n], ', err : ', err_arr[n])
        # load data into arrays (padded if necessary)
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
        
        n = n+1
    
    # define structured array and sort 
    w_type = np.dtype([('rot', 'float'), ('step', 'float')])
    w = np.empty(len(rot_arr), dtype = w_type)
    w['rot'] = rot_arr
    w['step'] = step_arr
    
    if sort_by == 'rot':
        sort_ind = np.argsort(w, order = ('rot', 'step'))
    elif sort_by == 'step':
        sort_ind = np.argsort(w, order = ('step', 'rot'))
    print(sort_ind)
    # re-order arrays    
    rot_sort = rot_arr[sort_ind]
    step_sort = step_arr[sort_ind]
    err_sort = err_arr[sort_ind]
    dat_sort =  dat_arr[:,sort_ind,:,:]
    probe_sort = probe_arr[:,sort_ind,:,:]
    
    # unsorted to hs signals
    '''
    d = hs.signals.Signal2D(data = dat_arr)
    p = hs.signals.Signal2D(data = probe_arr)
    r = hs.signals.Signal1D(data = rot_arr)
    s = hs.signals.Signal1D(data = step_arr)
    e = hs.signals.Signal1D(data = err_arr)
    '''
    
    #sorted to hs signals
    d_s = hs.signals.Signal2D(data = dat_sort)
    p_s = hs.signals.Signal2D(data = probe_sort)
    r_s = hs.signals.Signal1D(data = rot_sort)
    s_s = hs.signals.Signal1D(data = step_sort)
    e_s = hs.signals.Signal1D(data = err_sort)
    #crop
    '''
    d.crop(axis = (2), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
    d.crop(axis = (3), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
    '''
    
    d_s.crop(axis = (2), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
    d_s.crop(axis = (3), start =int((shape_dat / 2) - (crop_to/2)), end = int((shape_dat / 2) + (crop_to/2) ))
    
    #gaussian blur 
    print(d_s.data.shape)
    print(type(d_s.data))
    d_s.map(gaussian, sigma = blur)
    
    # fft 
    '''
    dat_fft = np.fft.fft2(d.data)
    dat_fft = np.fft.fftshift(dat_fft)
    
    d_fft = hs.signals.Signal2D(data = np.log10(np.abs(dat_fft)**2))
    d_fft.data = np.flip(d_fft.data, axis = 0)
    '''
    
    dat_sort_fft = np.fft.fft2(d_s.data)
    dat_sort_fft = np.fft.fftshift(dat_sort_fft)
    d_s_fft = hs.signals.Signal2D(data = np.log10(np.abs(dat_sort_fft)**2))
    d_s_fft.data = np.flip(d_s_fft.data, axis = 0)
    fft_mask = np.zeros_like(d_s_fft.data, dtype = 'bool')
    fft_shape = fft_mask.shape
    
    d_s_fft.inav[:,0].data[:,int(fft_shape[-1]/2), :] =0
    d_s_fft.inav[:,0].data[:,:,int(fft_shape[-2]/2)] =0
    
    rad_fft = radial_profile.radial_profile_stack(d_s_fft)
    print(n, ' files loaded successfully')
    return d_s, p_s, d_s_fft, rad_fft, r_s, s_s, e_s


