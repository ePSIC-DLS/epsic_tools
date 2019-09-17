# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:47:53 2019

@author: gys37319
"""
import hyperspy.api as hs
import numpy as np
import os
from IdentifyHDF5_files import get_HDF5_files
import functions_4DSTEM as fs
import py4DSTEM
from py4DSTEM.process.calibration import get_probe_size
from py4DSTEM.process.dpc import get_CoM_images, get_rotation_and_flip, get_phase_from_CoM
from py4DSTEM.process.dpc import get_wavenumber, get_interaction_constant

def scan_processing_file(proc_path):
    # scans the 'processing_params.txt' file and returns a dictionary of contents
    # 'processing_params.txt' must be of the format
    # key:value
    
    proc_path  = os.path.split(proc_path)[0]
    proc_file_path =  os.path.join(proc_path, 'processing_params.txt')
    if os.path.exists(proc_file_path) :
        proc_dict = {}
        #read value from text parameter file into dictionary
        with open(proc_file_path) as f:
            for line in f:
                #ignore commented lines
                if line.startswith('#'):
                    next
                #ignore blank lines
                elif line.startswith('\n'):
                    next
                else:
                    (key, val) = line.split(sep = ':')
                    proc_dict[key] = float(val)
    else:
        #processing_params.txt doesn't exist  - jump out. 
        print('no processing_params.txt file in ', proc_path)
        exit()
    #return the dictionary
    return proc_dict
    
def define_bf_disk(dp, proct_dict):
    #define bf disk 
    try:
        bf_thresh = proc_dict['BF_thresh']
    except KeyError:
        print('BF_thresh not set, default to 0.05')
        bf_thresh = 0.05
    #print(bf_thresh)
    #bf= fs.get_bf_disk(dp, sub = 20, threshold = bf_thresh, plot_me = True)
    PACBED = np.average(dp.data, axis = (0,1))
    r, x0, y0 = get_probe_size(PACBED)
    bf = [r,x0,y0]
    bf_exist = 1
    return bf, bf_exist

def get_adf(dp, bf, expand):
    #r, x0, y0 = bf
    #px,py,dx,dy = dp.data.shape
    #qy,qx = np.meshgrid(np.arange(dx),np.arange(dy))
    #qr = np.hypot(qx-x0,qy-y0)
    mask = get_mask(dp, bf, expand, bf_df = 'df')
    ADF = np.average(mask * dp.data, axis = (2,3))
    return ADF
    
def get_mask(dp, bf, expand, bf_df = 'bf'):
    #dp: 4D data set
    #bf: bright field disc object
    #expand = integer ; number of px to step out of bright field disc 
    #bf_df  = string ; 'bf': bright field mask ; 'df' : dark field mask
    r, x0, y0 = bf
    try:
        #if hs object
        px,py,dx,dy = dp.data.shape
    except:
        #if py4DSTEM object
        px,py,dx,dy = dp.data4D.shape
    qy,qx = np.meshgrid(np.arange(dx),np.arange(dy))
    qr = np.hypot(qx-x0,qy-y0)
    if bf_df == 'df':
        mask = qr > r + expand
    elif bf_df =='bf':
        mask = qr < r + expand
    return mask    
    
def process_data(proc_path,proc_bin_path, proc_dict):
    if 'Overwrite' in proc_dict:
        Overwrite = bool(proc_dict['Overwrite'] )
    else:
        Overwrite = False
    #load data lazily
    dp = hs.load(proc_path, lazy = True)
    dp_bin = hs.load(proc_bin_path)
    #flag to tell if bf has already been calculated
    bf_bin_exist = 0
    bf_exist = 0
    
    #ADF analysis
    if 'ADF' in proc_dict:
        #run adf analysis
        if proc_dict['ADF'] == 1:
            if bf_bin_exist == 0:
                #get bf thrershold value
                bf_bin, bf_bin_exist = define_bf_disk(dp_bin, proc_dict)
            #get ADF inner angle
            if 'ADF_expand' in proc_dict:
                ADF_expand = proc_dict['ADF_expand']
            else:
                ADF_expand = 20
            #get ADF image
            ADF = get_adf(dp_bin, bf_bin, ADF_expand)
            #save ADF image
            ADF_file = proc_bin_path.rpartition('.')[0] + '_ADF'
            hs_ADF = hs.signals.Signal2D(ADF)
            hs_ADF.save(ADF_file, overwrite = Overwrite) 
            hs_ADF.save(ADF_file, overwrite = Overwrite, extension = 'png') 
    
    #CoM analysis
    if 'CoM' in proc_dict:
        #run CoM analysis
        if proc_dict['CoM'] ==1:
            if 'bin_CoM' in proc_dict:
                if proc_dict['bin_CoM'] ==1:
                    if bf_bin_exist == 0:
                        #get BF thrershold value
                        bf_bin, bf_bin_exist = define_bf_disk(dp_bin, proc_dict)
                    bf_CoM = bf_bin
                    dp_CoM = py4DSTEM.file.datastructure.DataCube(dp_bin.data)
                    file_path = proc_bin_path.rpartition('.')[0]
                elif proc_dict['bin_CoM'] ==0:
                    if bf_exist ==0:
                        bf, bf_exist = define_bf_disk(dp, proc_dict)
                    bf_CoM = bf
                    dp_CoM = py4DSTEM.file.datastructure.DataCube(dp.data)
                    file_path = proc_path.rpartition('.')[0]
            #get BF outer angle
            if 'BF_expand' in proc_dict:
                BF_expand = proc_dict['ADF_expand']
            else:
                BF_expand = 20
            #build mask
            mask = get_mask(dp_CoM, bf_CoM, BF_expand, bf_df = 'bf')     
            #set normalise CoM parameter 
            if 'Normalize_CoM' in proc_dict:
                Normalize_CoM = bool(proc_dict['Normalize_CoM'])
            else:
                Normalize_CoM = True
            #get CoM 
            CoMx, CoMy = get_CoM_images(dp_CoM, mask = mask, normalize = Normalize_CoM)
            #pass to hyperspy and save
            CoMx_file = file_path + '_CoMx'
            hs_CoMx = hs.signals.Signal2D(CoMx)
            hs_CoMx.save(CoMx_file, overwrite = Overwrite)
            hs_CoMx.save(CoMx_file, overwrite = Overwrite, extension = 'png')
            
            CoMy_file = file_path +  '_CoMy'
            hs_CoMy = hs.signals.Signal2D(CoMy)
            hs_CoMy.save(CoMy_file, overwrite = Overwrite)
            hs_CoMy.save(CoMy_file, overwrite = Overwrite, extension = 'png')
            
    if 'DPC' in proc_dict:
        if proc_dict['DPC'] ==1:
            #get parameters
            theta = proc_dict['DPC_theta']
            flip = bool(proc_dict['DPC_flip'])
            pad_factor = int(proc_dict['DPC_pad'])
            low_pass = proc_dict['DPC_lowpass']
            high_pass = proc_dict['DPC_highpass']
            step_size = proc_dict['DPC_stepsize']
            niter = int(proc_dict['DPC_niter'])
            #calulate phase from CoM
            phase, error = get_phase_from_CoM(CoMx, CoMy, theta = theta, flip = flip, paddingfactor=pad_factor, regLowPass=low_pass, regHighPass= high_pass, stepsize=step_size, n_iter=niter )
            #pass to hyperspy object and save
            phase_file = file_path + '_phase'
            hs_phase = hs.signals.Signal2D(phase)
            hs_phase.save(phase_file, overwrite=Overwrite)
            hs_phase.save(phase_file, overwrite=Overwrite, extension = 'png')
            
            
            

        
    
            
            

            
        
    
#%%
#testing useage
beamline = 'e02'
year = '2019'
visit = 'mg22549-6'
#get hdf5 files
HDF5_dict= get_HDF5_files(beamline, year, visit)
#get processing parameters
proc_path = HDF5_dict['processing_path']
proc_dict = scan_processing_file(proc_path)
print(proc_dict)
print(HDF5_dict)
#just do one file 
file_n = 4 #3
this_fp = os.path.join(HDF5_dict['HDF5_paths'][file_n], HDF5_dict['HDF5_files'][file_n])
this_bin_fp = os.path.join(HDF5_dict['binned_HDF5_paths'][file_n], HDF5_dict['binned_HDF5_files'][file_n])
print(this_fp)
dp_bin = process_data(this_fp, this_bin_fp, proc_dict)
