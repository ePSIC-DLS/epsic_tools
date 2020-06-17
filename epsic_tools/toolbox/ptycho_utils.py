#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:52:15 2020

@author: eha56862
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.filters import gaussian
import glob
import json
import hyperspy.api as hs
from epsic_tools.toolbox import radial_profile
import os


from scipy import constants as pc


def e_lambda(e_0):
    """
    relativistic electron wavelength

    :param e_0: int
        accelerating voltage in volts
    :return:
    e_lambda: float
        wavelength in meters
    """
    import numpy as np

    
    e_lambda = (pc.h * pc.c) / np.sqrt((pc.e * e_0)**2  + 2 * pc.e * e_0 * pc.m_e * pc.c**2)
    
    return e_lambda


def plot_ptyREX_output(json_path, save_fig=None, crop=False):
    """
    To save the ptyREX recon output
    Parameters
    ----------
    json_path: str
        full path of the json file. This figure output will be saved in this folder.
    save_fig: str, default None
        In case we want the figure to be  saved the full path should be given here
        keyword argument.
    crop: Bool
        default False

    Returns
    -------

    """
    json_dict = json_to_dict(json_path)
    name = json_dict['base_dir'].split('/')[-1]
    recon_path = os.path.splitext(json_path)[0]+'.hdf'
    probe = get_probe_array(recon_path)
    if crop is True:
        obj = crop_recon_obj(json_path)
    else:
        obj = get_obj_array(recon_path)
    errors = get_error(recon_path)
            
    fig, axs = plt.subplots(3,2, figsize=(8, 11))
    
    fig.suptitle(name, fontsize = 18)
    
    obj_phase = np.angle(obj)
    s = obj_phase.shape[0]
    vmin_obj_p = np.min(obj_phase[int(s*0.4):int(0.6*s), int(s*0.4):int(0.6*s)])
    vmax_obj_p = np.max(obj_phase[int(s*0.4):int(0.6*s), int(s*0.4):int(0.6*s)])
    
    obj_mod = abs(obj)
    s = obj_mod.shape[0]
    vmin_obj_m = np.min(obj_mod[int(s*0.4):int(0.6*s), int(s*0.4):int(0.6*s)])
    vmax_obj_m = np.max(obj_mod[int(s*0.4):int(0.6*s), int(s*0.4):int(0.6*s)])        
    
    im1 = axs[0,0].imshow(np.angle(probe))
    axs[0,0].set_title('Probe Phase')
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    fig.colorbar(im1, ax = axs[0,0])
    im2 = axs[0,1].imshow(abs(probe))
    axs[0,1].set_title('Probe Modulus')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    fig.colorbar(im2, ax = axs[0,1])
    im3 = axs[1,0].imshow(np.angle(obj), cmap = 'gray', vmin = vmin_obj_p, vmax = vmax_obj_p)
    axs[1,0].set_title('Object Phase')
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    fig.colorbar(im3, ax = axs[1,0])
    im4 = axs[1,1].imshow(abs(obj), cmap = 'gray', vmin = vmin_obj_m, vmax = vmax_obj_m)
    axs[1,1].set_title('Object Modulus')
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])
    fig.colorbar(im4, ax = axs[1,1])
    axs[2,0].plot(errors)
    axs[2,0].set_title('Error vs iter')
    axs[2,1].imshow(np.sqrt(get_fft(obj, crop = 0.66)), cmap = 'viridis')
    axs[2,1].set_title('Sqrt of Object Phase fft')
    axs[2,1].set_xticks([])
    axs[2,1].set_yticks([])


    if save_fig is not None:
        if not os.path.exists(save_fig):
            os.mkdir(save_fig)

        plt.savefig(save_fig)
    return


def crop_recon_obj(json_file):
    json_dict = json_to_dict(json_file)
    pixelSize = get_json_pixelSize(json_file) 
    stepSize = json_dict['process']['common']['scan']['dR'][0]
    stepNum = json_dict['process']['common']['scan']['N'][0]
    dp_array = json_dict['process']['common']['detector']['crop'][0]
    obj_array = int(((stepNum - 1)*stepSize) / pixelSize + dp_array)
    recon_file = os.path.splitext(json_file)[0]+'.hdf'
    obj = get_obj_array(recon_file)
    sh = obj.shape[0]
    
    obj_crop = obj[int(sh/2 - obj_array / 2):int(sh/2 + obj_array / 2),\
                   int(sh/2 - obj_array / 2):int(sh/2 + obj_array / 2)]
    return obj_crop


def get_fft(obj_arr, crop = None):
    """
    Returns the abs array of the fft of the obj phase
    crop: fraction of FOV to cop
    """
    if crop is None:
        obj_fft = abs(np.fft.fftshift(np.fft.fft2(np.angle(obj_arr))))
    else:
        sh = obj_arr.shape[0]
        to_crop = crop * sh / 2
        obj_crop = obj_arr[int(sh / 2 - to_crop):int(to_crop + sh / 2),\
                   int(sh / 2 - to_crop):int(to_crop + sh / 2)]
        obj_fft = abs(np.fft.fftshift(np.fft.fft2(np.angle(obj_crop))))
        
    return obj_fft


def get_json_pixelSize(json_file):
    """
    Parameters
    ----------
    json_file: str
        full path of the json file. 

    Returns
    pixelSize: float
        reconstruction pixel size in (m)
    -------
    """
    json_dict = json_to_dict(json_file)
    wavelength = e_lambda(json_dict['process']['common']['source']['energy'][0])
    camLen = json_dict['process']['common']['detector']['distance']
    N = json_dict['process']['common']['detector']['crop'][0]
    dc = json_dict['process']['common']['detector']['pix_pitch'][0]
    
    pixelSize = (wavelength * camLen) / (N * dc)
    
    return pixelSize


def json_to_dict(json_path):
    """
    Parameters
    ----------
    json_path: str
        full path of the json file. 

    Returns
    -------
    json_dict: dictionary
    """
    with open(json_path) as jp:
        json_dict = json.load(jp)
    json_dict['json_path'] = json_path

    return json_dict



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

    to do 
    ------
    Break loading from hdf file into seperate functions 
    '''
    pn = pn +'/*.hdf'
    #build list of files
    fp_list = glob.glob(pn)
    len_dat = len(fp_list)
    n = 0 # counter
    # iterate through files
    print(pn)
    if verbose:
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
            #full_err_arr = []#np.empty( (len_dat, len(error),))
            
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
        #full_err_arr.append(error)
        
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
    #full_err_arr from list to padded array
    #temp_arr = np.zeros([len(full_err_arr),len(max(full_err_arr,key = lambda x: len(x)))])
    #for i,j in enumerate(full_err_arr):
    #    temp_arr[i][0:len(j)] = j
    #full_err_arr = temp_arr
    
    #full_err_sort = full_err_arr[sort_ind, :]
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
    #fe_s = hs.signals.Signal1D(data = full_err_sort)
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
    return d_s, p_s, d_s_fft, rad_fft, r_s, s_s, e_s#, fe_s

def load_recon(fn):
    params = get_json_params(fn)

    with h5py.File(fn, 'r') as h5_file:
        err = get_hdf5_error(h5_file)
        dat_phase = get_hdf5_object_phase(h5_file, params)
        dat_mod = get_hdf5_object_modulus(h5_file, params)
        probe_phase = get_hdf5_probe_phase(h5_file)
        probe_mod = get_hdf5_probe_modulus(h5_file)
    
    dat_shape = dat_phase.shape
    dat = np.empty(shape = (2, dat_shape[0], dat_shape[1]))
    dat[0,:,:] = dat_phase
    dat[1,:,:] = dat_mod
    
    probe_shape = probe_phase.shape
    probe = np.empty(shape = (2, probe_shape[0], probe_shape[1]))
    probe[0,:,:] = probe_phase
    probe[1,:,:] = probe_mod
    
    #to hyperspy objects
    dat = hs.signals.Signal2D(data = dat)
    probe = hs.signals.Signal2D(data = probe)
    err = hs.signals.Signal1D(data = err)

    return dat, probe, err



def get_error(file_path):
    """
    Parameters
    ----------
    file_path: str
        full path of the recon file. It check if it is frtom ptypy or ptyREX

    Returns
    -------
    errors: np.array
        error as numpy array
    """

    if os.path.splitext(file_path)[1] == '.ptyr':
        ptyr_file_path = file_path
        f = h5py.File(ptyr_file_path,'r')
        content = f['content']
        iter_info = content['runtime']['iter_info']
        iter_num = len(iter_info.keys())
        #print(iter_num)
        errors = []
        index = '00000'
        for i in range(iter_num):
            next_index = int(index) + i
            next_index = str(next_index)
            if len(next_index) == 1:
                next_index = '0000' + next_index
            elif len(next_index) == 2:
                next_index = '000' + next_index
            elif len(next_index) == 3:
                next_index = '00' + next_index
            errors.append(content['runtime']['iter_info'][next_index]['error'][:])
        errors = np.asarray(errors) 
    elif os.path.splitext(file_path)[1] == '.hdf':
        f = h5py.File(file_path,'r')
        errors = f['entry_1']['process_1']['output_1']['error'][:]
        
    return errors


def get_hdf5_error(h5_file):
    error = np.array(h5_file['entry_1']['process_1']['output_1']['error'])
    return error


def get_hdf5_object_phase(h5_file, params):
    #get object_phase
    dat = h5_file['entry_1']['process_1']['output_1']['object_phase']
    dat = dat[0,0,0,0,0,:,:]
    #rotate
    rot_angle = 90-params['process']['common']['scan']['rotation']
    dat = rotate(dat, rot_angle)
    return dat


def get_hdf5_object_modulus(h5_file, params):
    #get modulus data
    dat_m = h5_file['entry_1']['process_1']['output_1']['object_modulus']
    dat_m  = dat_m[0,0,0,0,0,:,:]
    #rotate
    rot_angle = 90-params['process']['common']['scan']['rotation']
    dat_m = rotate(dat_m, rot_angle)
          

def get_probe_array(file_path):
    """
    Parameters
    ----------
    file_path: str
        full path of the recon file. It check if it is frtom ptypy or ptyREX

    Returns
    -------
    probe_data_arr: np.array
        complex probe numpy array 
    """

    if os.path.splitext(file_path)[1] == '.ptyr':
        f = h5py.File(file_path,'r')
        content = f['content']
        obj = content['obj']
        probe = content['probe']
        dataname = list(obj.keys())[0]
        probe_data = probe[dataname]
        probe_data_arr = probe_data['data'][0]
        
    elif os.path.splitext(file_path)[1] == '.hdf':
        f = h5py.File(file_path,'r')
        probe = f['entry_1']['process_1']['output_1']['probe'][0]
        probe_data_arr = np.squeeze(probe)
    
    return probe_data_arr


def get_obj_array(file_path):
    """
    Parameters
    ----------
    file_path: str
        full path of the recon file. It check if it is frtom ptypy or ptyREX

    Returns
    -------
    data_arr: np.array
        complex object numpy array 
    """
    if os.path.splitext(file_path)[1] == '.ptyr':
        f = h5py.File(file_path,'r')
        content = f['content']
        obj = content['obj']
        dataname = list(obj.keys())[0]
        data = obj[dataname]
        data_arr = data['data'][0]
    elif os.path.splitext(file_path)[1] == '.hdf':
        f = h5py.File(file_path,'r')
        data = f['entry_1']['process_1']['output_1']['object'][0]
        data_arr = np.squeeze(data)
    return data_arr

# the following functions could be replaced by get_obj_array and get_probe_array

def get_hdf5_probe_phase(h5_file):
    #get probe
    probe = np.array(h5_file['entry_1']['process_1']['output_1']['probe_phase'])
    #sum seperate coherent modes
    probe = probe[:,:,0,0,0,:,:].sum(axis = (0,1))
    return probe


def get_hdf5_probe_modulus(h5_file):
    probe_m = np.array(h5_file['entry_1']['process_1']['output_1']['probe_modulus'])
    probe_m = probe_m[:,:,0,0,0,:,:].sum(axis = (0,1))
    return probe_m


def get_hdf5_complex_probe(h5_file):
    #get complex probe
    probe_c = np.array(h5_file['entry_1']['process_1']['output_1']['probe'])
    probe_c = probe_c[:,:,0,0,0,:,:].sum(axis = (0,1))
    return probe_c   


def get_json_params(h5_file):  
    # Could be replaced by json_to_dict    
    fj = h5_file[:-4] + '.json'
    #open json file
    with open(fj) as r:
        params = json.load(r)
    return params


def get_sampling_factor(x,d,n,dr):
    """
    Parameters
    ----------
    x: float
        pixelSize * number of pixels in probe
    d: float
        probe diameter in m
    n: int
        number of probe positions
    dr: float
        step size 

    Returns
    -------
    s: float
        sampling factor 
        
    Ref: Darren Batey PhD Thesis, Page 68.
    """
    du = 1/float(x)
    s = 1 / ( 2*du*( (d/n) + (dr*(1 - (1/n) ) ) ) )
    return s



def plot_ptyr(filename):
    """
    Plots the real and imaginary parts of a ptypy recon file
    """
    f = h5py.File(filename,'r')
    content = f['content']
    obj = content['obj']
    dataname = list(obj.keys())[0]
    data = obj[dataname]
    data_arr = data['data']
    
    probe = content['probe']
    probe_data = probe[dataname]
    probe_data_arr = probe_data['data']
    
    plt.rcParams['image.cmap'] = 'viridis'
    
    plt.subplot(141); plt.imshow(data_arr[0].real); plt.title('Object real'); plt.axis('off');
    plt.subplot(142); plt.imshow(data_arr[0].imag); plt.title('Object imaginary'); plt.axis('off');
    plt.subplot(143); plt.imshow(probe_data_arr[0].real); plt.title('Probe real'); plt.axis('off');
    plt.subplot(144); plt.imshow(probe_data_arr[0].imag); plt.title('Probe imaginary'); plt.axis('off');
