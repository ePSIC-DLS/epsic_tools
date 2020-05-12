#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts looks into series of folders finds ptypy recon folders and saves 
formatted figures of the recon

@author: eha56862
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import argparse
import json
from epsic_tools.toolbox.sim_utils import e_lambda
#from sim_utils import e_lambda, get_potential, _sigma
import collections


def get_raw_dir_list(sim_matrix_path, get_all = False):
    '''
    checks for the folders with only two files and identify them as raw
    get_all set to True returns all the folders 
    
    Parameters
    ___________
    sim_matrix_path: str
        full path holding the sim matrix
    get_all: bool
        Default False. Set to True if all folders needed
    
    Returns:
    _________
    raw_dirs: list
        list of directories
    '''
    raw_dirs = []
    it =  os.scandir(sim_matrix_path)
    if get_all:
        for entry in it:
            if entry.is_dir():
                raw_dirs.append(entry.path)
    else:
        for entry in it:
            if entry.is_dir():
     #           if 'recons' not in os.listdir(os.path.dirname(entry.path)): 
                if len(os.listdir(entry.path)) == 2:
                    raw_dirs.append(entry.path)
    return raw_dirs


def get_ptyREX_ready(sim_matrix_path):
    '''
    checks for the folders that have ptyREX json file 
    
    Returns
    ptyREX_dirs: list
        list of dirs
    '''
    ptyREX_dirs = []
    it =  os.scandir(sim_matrix_path)
    for entry in it:
        if entry.is_dir():
            it2 = os.scandir(entry.path)
            for entry2 in it2:
                if entry2.is_file():
                    if entry2.name.startswith('ptyREX_'):
                        ptyREX_dirs.append(entry.path)
    return ptyREX_dirs



def parse_params_file(params_file, h5_file, drop_unneeded = True):
    '''
    Reads the parameters text file into a dict to be fed into ptypy / pycho recons
    '''
    exp_dict = {}
    with open(params_file) as f:    
        for line in f: 
            line = line.strip('-')
            exp_dict[line.strip().partition(':')[0]] = line.strip().partition(':')[-1]
    
    original_sim_file = exp_dict['output-file'].split('/')[-1]
    
    if original_sim_file != h5_file.split('/')[-1]:
        exp_dict['data'] = exp_dict.pop('output-file')
        exp_dict['data'] = h5_file
    else:
        exp_dict['data'] = exp_dict.pop('output-file')
        
    exp_dict['cell_dimension(m)'] = [1e-10*float(i) for i in exp_dict['cell-dimension'].split(' ')]
    exp_dict.pop('cell-dimension')
    exp_dict['tile_uc'] = [float(i) for i in exp_dict['tile-uc'].split(' ')]
    
    if 'skip' in h5_file:
        exp_dict['data_key'] = 'dataset'
    else:
        exp_dict['data_key'] = '4DSTEM_simulation/data/datacubes/hdose_noisy_data'
    exp_dict['xyz'] = exp_dict.pop('input-file')
    exp_dict['accel_voltage(eV)'] = int(exp_dict.pop('energy')) * 1000
    exp_dict['semi_angle(rad)'] = float(exp_dict.pop('probe-semiangle')) * 1e-3
    if 'skip' in h5_file:
        skip_ind = h5_file.index('skip')
        step_factor = int(h5_file[skip_ind + 4])
        exp_dict['step_size(m)'] = step_factor * float(exp_dict.pop('probe-step-x')) * 1e-10
    else:
        exp_dict['step_size(m)'] = float(exp_dict.pop('probe-step-x')) * 1e-10
    exp_dict['sim_pixel_size(m)'] = float(exp_dict.pop('pixel-size-x')) * 1e-10
    exp_dict['defocus(m)'] = float(exp_dict.pop('probe-defocus'))* 1e-10
    exp_dict['C3(m)'] = float(exp_dict.pop('C3'))* 1e-10
    exp_dict['C5(m)'] = float(exp_dict.pop('C5'))* 1e-10
    
    
    wavelength = e_lambda(exp_dict['accel_voltage(eV)'])
    
    exp_dict['wavelength'] = wavelength
    
    # getting rid of unneeded stuff:
    
    if drop_unneeded:
        exp_dict.pop('num-threads')
        exp_dict.pop('algorithm')
        exp_dict.pop('potential-bound')
        exp_dict.pop('num-FP')
        exp_dict.pop('slice-thickness')
        exp_dict.pop('num-slices')
        exp_dict.pop('zstart-slices')
        exp_dict.pop('alpha-max')
        exp_dict.pop('batch-size-cpu')
        exp_dict.pop('tile-uc')
        exp_dict.pop('detector-angle-step')
        exp_dict.pop('probe-xtilt')
        exp_dict.pop('probe-ytilt')
        exp_dict.pop('scan-window-x')
        exp_dict.pop('scan-window-y')
        exp_dict.pop('scan-window-xr')
        exp_dict.pop('scan-window-yr')
        exp_dict.pop('random-seed')
        exp_dict.pop('4D-amax')
        for k in ['thermal-effects', 'save-3D-output', 'save-4D-output', '4D-crop', 'save-DPC-CoM', 
                  'save-potential-slices','save-real-space-coords',  'occupancy', 'nyquist-sampling',
                  'probe-step-y', 'pixel-size-y']:
            exp_dict.pop(k)
    # using the sim parameters to calculate the bf disc rad
    det_pix_num = int((exp_dict['cell_dimension(m)'][0] * exp_dict['tile_uc'][0]) / (2 * exp_dict['sim_pixel_size(m)']))
    a_max = wavelength / (4 * exp_dict['sim_pixel_size(m)']) # alpha max
    pix_per_rad = (det_pix_num / 2) / a_max
    exp_dict['pupil_rad(pixels)'] = pix_per_rad * exp_dict['semi_angle(rad)']
    
    exp_dict['detector_pixel_size(m)'] = 55e-6 # assuming the same as Medipix
    exp_dict['detector_distance(m)'] = (det_pix_num / 2) * exp_dict['detector_pixel_size(m)'] / a_max
    
    exp_dict['output_base'] = os.path.dirname(exp_dict['data'])
    
    exp_dict['rotation_angle(degrees)'] = 0
    
    return exp_dict
#%%

#%%
class NestedDefaultDict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))
    
def write_ptyrex_json(exp_dict, iter_num):
    with h5py.File(exp_dict['data'], 'r') as f:
        data = f.get(exp_dict['data_key'])
        data_arr = np.array(data)
    
    scan_y = data_arr.shape[1]
    scan_x = data_arr.shape[0]
    
    N_x = data_arr.shape[2]
    N_y = data_arr.shape[3]

    params = NestedDefaultDict()
    
    params['process']['gpu_flag'] = 1
    params['process']['save_interval'] = 10
    params['process']['PIE']['iterations'] = iter_num
    params['process']['common']['source']['energy'] = [exp_dict['accel_voltage(eV)']]
    params['process']['common']['source']['radiation'] = 'electron'
    params['process']['common']['source']['flux'] = -1
    
    params['process']['common']['detector']['pix_pitch'] = list([exp_dict['detector_pixel_size(m)'], exp_dict['detector_pixel_size(m)']])
    params['process']['common']['detector']['distance'] = exp_dict['detector_distance(m)']
    params['process']['common']['detector']['bin'] = list([1, 1]) 
    params['process']['common']['detector']['min_max'] = list([0, 1000000])
    params['process']['common']['detector']['optic_axis']= list([N_x / 2, N_x/2])
    params['process']['common']['detector']['crop'] = list([N_x, N_y])
    params['process']['common']['detector']['orientation'] = '00'
    params['process']['common']['detector']['mask_flag'] = 0
    
    params['process']['common']['probe']['convergence'] = 2*exp_dict['semi_angle(rad)']
    params['process']['common']['probe']['distance'] = -1
    params['process']['common']['probe']['focal_dist'] = -1
    params['process']['common']['probe']['load_flag'] = 0
    params['process']['common']['probe']['diffuser'] = 0
    params['process']['common']['probe']['aperture_shape'] = 'circ'
    params['process']['common']['probe']['aperture_size'] = exp_dict['pupil_rad(pixels)']*exp_dict['detector_pixel_size(m)']

    params['process']['common']['object']['load_flag'] = 0
    
    params['process']['common']['scan']['rotation'] = exp_dict['rotation_angle(degrees)']
    params['process']['common']['scan']['fast_axis'] = 1
    params['process']['common']['scan']['orientation'] = '00'
    params['process']['common']['scan']['type'] = 'tv'
    params['process']['common']['scan']['load_flag'] = 0
    params['process']['common']['scan']['dR'] = list([exp_dict['step_size(m)'], exp_dict['step_size(m)']])
    params['process']['common']['scan']['N'] = list([scan_x, scan_y])
    
    params['experiment']['data']['data_path'] = exp_dict['data']
    params['experiment']['data']['dead_pixel_flag'] = 0 
    params['experiment']['data']['flat_field_flag'] = 0 
#    params['experiment']['data']['dead_pixel_path'] = exp_dict['mask']
#    params['experiment']['data']['flat_field_path'] = exp_dict['mask']
    params['experiment']['data']['load_flag'] = 1
    params['experiment']['data']['meta_type'] = 'hdf'
    params['experiment']['data']['key'] = exp_dict['data_key']
    
    params['experiment']['sample']['position'] = list([0, 0, 0])

    params['experiment']['detector']['position'] = list([0, 0, exp_dict['detector_distance(m)']])

    params['experiment']['optics']['lens']['alpha'] = 2*exp_dict['semi_angle(rad)']
    params['experiment']['optics']['lens']['defocus'] = list([exp_dict['defocus(m)'], exp_dict['defocus(m)']])
    params['experiment']['optics']['lens']['use'] = 1
    params['experiment']['optics']['diffuser']['use'] = 0
    params['experiment']['optics']['FZP']['use'] = 0
    params['experiment']['optics']['pinhole']['use'] = 0

    params['base_dir'] = exp_dict['output_base']
    params['process']['save_dir'] = exp_dict['output_base']
    params['process']['cores'] = 1
    
    json_file = os.path.join(exp_dict['output_base'], 'ptyREX_' + exp_dict['data'].split('/')[-1].split('.')[0] + '.json')
    exp_dict['ptyREX_json_file'] = json_file    
    with open(json_file, 'w+') as outfile:
        json.dump(params, outfile, indent = 4)


#%%
def get_ptyREX_recon_list(sim_matrix_path, run_id=None):
    """

    Parameters
    ----------
    sim_matrix_path: str
        full path of the simulation matrix
    run_id: str
        default None. If provided the json files with run_id in their names will be
        returned.

    Returns
    -------
    common_dirs: list
        List of json files that have identically named hdf5 file in the same folder
    """
    recon_dirs = []
    json_dirs = []
    common_dirs = []
    if run_id is None:
        run_id = ''
    for dirname, dirnames, filenames in os.walk(sim_matrix_path):
        for filename in filenames:
            if (os.path.splitext(filename)[1] == '.hdf') and (run_id in filename):
                recon_dirs.append(os.path.join(dirname, filename))
            if (os.path.splitext(filename)[1] == '.json') and (run_id in filename):
                json_dirs.append(os.path.join(dirname, filename))
    for json_file in json_dirs:
        if os.path.splitext(json_file)[0]+'.hdf' in recon_dirs:
            common_dirs.append(json_file)

    return common_dirs
#%%

def get_figs(sim_matrix_path):
    '''
    returns the list of the png figure of the final recon
    '''
    fig_list = []
    for dirname, dirnames, filenames in os.walk(sim_matrix_path):

        for filename in filenames:
            if 'recons' in os.path.join(dirname, filename):
                if os.path.splitext(filename)[1] == '.png':
                    fig_list.append(os.path.join(dirname, filename))
    
    return fig_list
    
#def parse_file_name(recon_file_path):
#    '''
#    takes out useful params from recon file name
#    '''
#    file_name = os.path.splitext(os.path.basename(recon_file_path))[0]
#    
def get_probe_func(file_path):
    '''
    Gets the path for a ptypy recon and returns the probe as a complex array
    works for both ptypy and ptyREX recons
    '''
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

def get_obj_func(file_path):
    '''
    Gets the path for a recon file and returns the object as a complex array
    works for both ptypy and ptyREX recons
    '''
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

def get_error(file_path):
    '''
    input: pile_path of a recon file - it determines if ptypy / ptyREX
    
    output: 
        error as numpy array
    '''
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

#%%
def json_to_dict(json_path):
    with open(json_path) as jp:
        json_dict = json.load(jp)
    json_dict['json_path'] = json_path
    base_path = os.path.dirname(json_path)
    for file in os.listdir(base_path):
        if file.endswith('h5'):
            sim_path = os.path.join(base_path, file)
    json_dict['sim_path'] = sim_path
    return json_dict

#%%

def save_ptyREX_output(json_path, fig_dump = None, crop = True):
    """
    To save the ptyREX recon output
    Parameters
    ----------
    json_path: str
        full path of the json file. This figure output will be saved in this folder.
    fig_dump: str, default None
        In case we want the figures to be also saved in a secondary folder, this new path can be passed as this
        keyword argument.

    Returns
    -------

    """
    json_dict = json_to_dict(json_path)
    name = json_dict['base_dir'].split('/')[-1]
    recon_path = os.path.splitext(json_path)[0]+'.hdf'
    probe = get_probe_func(recon_path)
    if crop is True:
        obj = crop_recon_obj(json_path)
    else:
        obj = get_obj_func(recon_path)
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
    fig.colorbar(im1, ax = axs[0,0])
    im2 = axs[0,1].imshow(abs(probe))
    axs[0,1].set_title('Probe Modulus')
    fig.colorbar(im2, ax = axs[0,1])
    im3 = axs[1,0].imshow(np.angle(obj), cmap = 'gray', vmin = vmin_obj_p, vmax = vmax_obj_p)
    axs[1,0].set_title('Object Phase')
    fig.colorbar(im3, ax = axs[1,0])
    im4 = axs[1,1].imshow(abs(obj), cmap = 'gray', vmin = vmin_obj_m, vmax = vmax_obj_m)
    axs[1,1].set_title('Object Modulus')
    fig.colorbar(im4, ax = axs[1,1])
    axs[2,0].plot(errors)
    axs[2,0].set_title('Error vs iter')
    axs[2,1].imshow(np.sqrt(get_fft(obj)), cmap = 'viridis')
    axs[2,1].set_title('Sqrt of Object Phase fft')

    
    saving_path1 = os.path.splitext(recon_path)[0]+ name +'.png'
    plt.savefig(saving_path1)
    
    if fig_dump is not None:
        #base_path2 = '/dls/e02/data/2020/cm26481-1/processing/pty_simulated_data_MD/output_figs_ptREX_20200213/'
        if not os.path.exists(fig_dump):
            os.mkdir(fig_dump)
        saving_path = os.path.join(fig_dump, os.path.splitext(recon_path)[0].split('/')[-1] +'_'+ name +'.png')
        plt.savefig(saving_path)
    
    plt.close('all')
        
    return

#%%
def get_json_pixelSize(json_file):
    json_dict = json_to_dict(json_file)
    wavelength = e_lambda(json_dict['process']['common']['source']['energy'][0])
    camLen = json_dict['process']['common']['detector']['distance']
    N = json_dict['process']['common']['detector']['crop'][0]
    dc = json_dict['process']['common']['detector']['pix_pitch'][0]
    
    pixelSize = (wavelength * camLen) / (N * dc)
    
    return pixelSize
    

#%%
def crop_recon_obj(json_file):
    json_dict = json_to_dict(json_file)
    pixelSize = get_json_pixelSize(json_file) 
    stepSize = json_dict['process']['common']['scan']['dR'][0]
    stepNum = json_dict['process']['common']['scan']['N'][0]
    dp_array = json_dict['process']['common']['detector']['crop'][0]
    obj_array = int(((stepNum - 1)*stepSize) / pixelSize + dp_array)
    recon_file = os.path.splitext(json_file)[0]+'.hdf'
    obj = get_obj_func(recon_file)
    sh = obj.shape[0]
    
    obj_crop = obj[int(sh/2 - obj_array / 2):int(sh/2 + obj_array / 2),\
                   int(sh/2 - obj_array / 2):int(sh/2 + obj_array / 2)]
    return obj_crop
    
#%%
    
def get_fft(obj_arr):
    
    obj_fft = abs(np.fft.fftshift(np.fft.fft2(np.angle(obj_arr))))
    return obj_fft
#%%


#%%        
def main(scan_path):
#    ptypy_recon_dirs = get_ptypy_recon_list(scan_dir)
#    for recon_file in ptypy_recon_dirs:
#        save_recon_fig(recon_file)
    path = '/dls/e02/data/2020/cm26481-1/processing/pty_simulated_data_MD/sim_matrix_ptyREX_v2/all_figures'
    json_files = get_ptyREX_recon_list(scan_path)
    
    for json_file in json_files:
        save_ptyREX_output(json_file, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_path', help='path to scan for ptypy recon to save figures')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()

    main(args.scan_path)
