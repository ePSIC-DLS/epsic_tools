# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:46:53 2020

@author: gys37319
"""

import numpy as np
import h5py

#raw file from dr. probe simulation
fn_r =r'Y:/2019/cm22979-8/processing/Merlin/20191114_15kVptycho_graphene/probe_sims/15kV_100um_Cs987um_alpha_25mrad_dfneg70_real_1024'
fn_i = r'Y:/2019/cm22979-8/processing/Merlin/20191114_15kVptycho_graphene/probe_sims/15kV_100um_Cs987um_alpha_25mrad_dfneg70_imag_1024'
bin_by = 4

data_r  = np.fromfile(fn_r, dtype = 'f4')
arr_size = int(np.sqrt(data_r.shape)[0])
data_r = data_r.reshape((arr_size, arr_size))
data_r = data_r.reshape( arr_size // bin_by, bin_by,  arr_size // bin_by, bin_by).sum(3).sum(1)

data_i  = np.fromfile(fn_i, dtype = 'f4')
data_i = data_i.reshape((arr_size, arr_size))
data_i = data_i.reshape( arr_size // bin_by, bin_by,  arr_size // bin_by, bin_by).sum(3).sum(1)

data = data_r + 1j * data_i
data = data[np.dnewaxis, np.newaxis, np.newaxis, np.newaxis,np.newaxis, :,:] 

d5 = h5py.File(fn_r + '_bin_' + str(bin_by) + '_complex.hdf5' , 'w')
d5.create_dataset('entry_1/process_1/output_1/probe', data = data)
d5.create_dataset('entry_1/process_1/PIE_1/detector/binning', data = [1,1])
d5.create_dataset('entry_1/process_1/PIE_1/detector/upsample', data = [1,1])
d5.create_dataset('entry_1/process_1/PIE_1/detector/crop',data = [arr_size, arr_size])
d5.create_dataset('entry_1/process_1/common_1/dx', data = [4.52391605e-11 , 4.52391605e-11 ]) # px size
d5.close()
