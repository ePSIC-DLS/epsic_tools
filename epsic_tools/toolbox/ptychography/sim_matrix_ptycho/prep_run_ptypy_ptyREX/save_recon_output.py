#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts looks into series of folders finds ptypy recon folders and saves 
formatted figures of the recon

@author: eha56862
"""

#import matplotlib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
#import ptypy
#from ptypy import utils as u
#from ptypy import io
import os
import h5py
import argparse
import scandir


def get_ptypy_recon_list(sim_matrix_path):
    '''
    checks for the folders with recon dir in them
    '''
    recon_dirs = []

    for dirname, dirnames, filenames in os.walk(sim_matrix_path):
        # print path to all subdirectories first.
        #for subdirname in dirnames:
        #    if 'recons' in subdirname:
        #        # print(os.path.join(dirname, subdirname))
        #        recon_dirs.append(os.path.join(dirname, subdirname))
        for filename in filenames:
            if 'recons' in os.path.join(dirname, filename):
                if os.path.splitext(filename)[1] == '.ptyr':
                    recon_dirs.append(os.path.join(dirname, filename))
    
    return recon_dirs
def get_ptyREX_recon_list(sim_matrix_path):
    '''
    checks for the folders with recon dir in them
    '''
    recon_dirs = []

    for dirname, dirnames, filenames in os.walk(sim_matrix_path):
        # print path to all subdirectories first.
        #for subdirname in dirnames:
        #    if 'recons' in subdirname:
        #        # print(os.path.join(dirname, subdirname))
        #        recon_dirs.append(os.path.join(dirname, subdirname))
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.hdf':
                recon_dirs.append(os.path.join(dirname, filename))
    return recon_dirs

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

def save_recon_fig(file_path):
    '''
    Gets the path for a ptypy recon and saves the corresponding formatted figure in the same dir
    '''
    if os.path.splitext(file_path)[1] == '.ptyr':
        ptyr_file_path = file_path
        obj = get_obj_func(file_path)
        probe = get_probe_func(file_path)
        errors = get_error(file_path)
        
        fig, axs = plt.subplots(3,2, figsize=(8, 11))
        
        fig.suptitle(os.path.splitext(os.path.basename(ptyr_file_path))[0], fontsize = 18)
        
        obj_phase = np.angle(obj)
        s = obj_phase.shape[0]
        vmin_obj_p = np.min(obj_phase[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        vmax_obj_p = np.max(obj_phase[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        
        obj_mod = abs(obj)
        s = obj_mod.shape[0]
        vmin_obj_m = np.min(obj_mod[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        vmax_obj_m = np.max(obj_mod[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        
        im1 = axs[0,0].imshow(np.angle(probe))
        axs[0,0].set_title('Probe Phase')
        fig.colorbar(im1, ax = axs[0,0])
        im2 = axs[0,1].imshow(abs(probe))
        axs[0,1].set_title('Probe Modulus')
        fig.colorbar(im2, ax = axs[0,1])
        im3 = axs[1,0].imshow(np.angle(obj), vmin = vmin_obj_p, vmax = vmax_obj_p)
        axs[1,0].set_title('Object Phase')
        fig.colorbar(im3, ax = axs[1,0])
        im4 = axs[1,1].imshow(abs(obj), vmin = vmin_obj_m, vmax = vmax_obj_m)
        axs[1,1].set_title('Object Modulus')
        fig.colorbar(im4, ax = axs[1,1])
        axs[2,0].plot(errors[:,0])
        axs[2,0].set_title('Fourier magnitude error vs iter')
        axs[2,1].plot(errors[:,2])
        axs[2,1].set_title('Error exit wave vs iter')
        
        
        saving_path = os.path.splitext(ptyr_file_path)[0]+'.png'
        plt.savefig(saving_path)
        
        base_path2 = '/dls/e02/data/2020/cm26481-1/processing/pty_simulated_data_MD/output_figs_ptypy_20200213/'
        if not os.path.exists(base_path2):
            os.mkdir(base_path2)
        saving_path2 = base_path2 + file_path.split('/')[-2]+'.png'
        plt.savefig(saving_path2)
        
        plt.close('all')
        
    elif os.path.splitext(file_path)[1] == '.hdf':
        probe = get_probe_func(file_path)
        obj = get_obj_func(file_path)
        errors = get_error(file_path)
                
        fig, axs = plt.subplots(3,2, figsize=(8, 11))
        
        fig.suptitle(os.path.splitext(os.path.basename(file_path))[0], fontsize = 18)
        
        obj_phase = np.angle(obj)
        s = obj_phase.shape[0]
        vmin_obj_p = np.min(obj_phase[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        vmax_obj_p = np.max(obj_phase[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        
        obj_mod = abs(obj)
        s = obj_mod.shape[0]
        vmin_obj_m = np.min(obj_mod[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])
        vmax_obj_m = np.max(obj_mod[int(s*0.3):int(0.7*s), int(s*0.3):int(0.7*s)])        
        
        im1 = axs[0,0].imshow(np.angle(probe))
        axs[0,0].set_title('Probe Phase')
        fig.colorbar(im1, ax = axs[0,0])
        im2 = axs[0,1].imshow(abs(probe))
        axs[0,1].set_title('Probe Modulus')
        fig.colorbar(im2, ax = axs[0,1])
        im3 = axs[1,0].imshow(np.angle(obj), vmin = vmin_obj_p, vmax = vmax_obj_p)
        axs[1,0].set_title('Object Phase')
        fig.colorbar(im3, ax = axs[1,0])
        im4 = axs[1,1].imshow(abs(obj), vmin = vmin_obj_m, vmax = vmax_obj_m)
        axs[1,1].set_title('Object Modulus')
        fig.colorbar(im4, ax = axs[1,1])
        axs[2,0].plot(errors)
        axs[2,0].set_title('Error vs iter')
        fig.delaxes(axs[2,1])
        
        saving_path1 = os.path.splitext(file_path)[0]+ file_path.split('/')[-2]+'.png'
        plt.savefig(saving_path1)
        
        base_path2 = '/dls/e02/data/2020/cm26481-1/processing/pty_simulated_data_MD/output_figs_ptREX_20200213/'
        if not os.path.exists(base_path2):
            os.mkdir(base_path2)
        saving_path2 = base_path2 + file_path.split('/')[-2]+'.png'
        plt.savefig(saving_path2)
        
        plt.close('all')
        
    return
        
def main(scan_dir):
    ptypy_recon_dirs = get_ptypy_recon_list(scan_dir)
    for recon_file in ptypy_recon_dirs:
        save_recon_fig(recon_file)
        
    ptyREX_recon_files = get_ptyREX_recon_list(scan_dir)
    for recon_file in ptyREX_recon_files:
        save_recon_fig(recon_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_path', help='path to scan for ptypy recon to save figures')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()

    main(args.scan_path)
