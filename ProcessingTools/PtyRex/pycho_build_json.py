# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:09:03 2019

@author: gys37319
"""

import json
import hyperspy.api as hs
import numpy as np
#path to template json file
fp = r"C:\Users\gys37319\Documents\GitHub\Merlin-Medipix\ProcessingTools\pycho_json.json"
with open(fp) as r:
    params = json.load(r)

#user variables
data_fp =  r'Y:/2020/cm26481-1/processing/Merlin/20200130_80kV_graphene_600C_pty/20200130 163208/sub_skipscan2_binned_diff_80kV_600C_CLA_40um_CL_10cm_8C_20Mx_A2_4p71_dfneg10nm_scan_array_32by32_diff_plane_128by128_.hdf5'
#r'Y:/2019/cm22979-6/processing/Merlin/Merlin/20191001_15kV_ptycho/MoS2_700C/20191002 150849/subscan_32x32_bin2_MoS2_15kV_50umClAp_3Mx_77def_A2_1p07.hdf5'
energy = 80000 #15000 # eV
convergence_semi_angle =0.03173#0.0235 # rad
camera_length =0.1732# 0.136 # m
scan_rotation = 50#79.00 # deg
defocus = -5e-9 # m
scan_step = 0.744e-10#2.58e-10# m 
bin_detector = 1
iterations = 500 #numer of EPIE iterations
decay = [1,0,1] # decay of update parameter [start, stop, power of decay]
############### variable to itterate
rot_step = 1#0.02
n_rot_step = 20
###############
#output will be saved here in processing/ptycho
base_dir =r'/dls/e02/data/2020/cm26481-1/processing/Merlin/20200130_80kV_graphene_600C_pty/cluster'
# r'/dls/e02/data/2019/cm22979-6/processing/Merlin/Merlin/20191001_15kV_ptycho/cluster'

#interrogate data
dat = hs.load(data_fp, lazy = True)
dat_shape = dat.data.shape
print('data shape : ', dat_shape)

#determine other variables from data and update params
#scan array shape
scan_x = dat_shape[0]
scan_y = dat_shape[1]
#detector array shape
N_x = dat_shape[2]
N_y = dat_shape[3]
#calculate detector binning for quad chip merlin detector
binning = np.floor(515 / N_x)
# get mask 
mask_fp = r'//dls/e02/data/2019/cm22979-6/processing/Merlin/Reconstruction_masks/Merlin_bin' + str(int(binning)) +'.h5'
#convert path to linux 
lnx_fp = r'//dls/e02/data'+ data_fp[2:] 
#medipix px size 
px_size = 55e-6 #m
#adjust for binning
adj_px_size = px_size * binning
optic_axis =  N_x / 2
#update params
params['experiment']['data']['data_path'] = lnx_fp
params['experiment']['data']['dead_pixel_path'] = mask_fp
params['process']['common']['source']['energy'][0] = energy
params['process']['common']['scan']['N'][0] = scan_y
params['process']['common']['scan']['N'][1] = scan_x
params['process']['PIE']['scan']['area'][0] = scan_y
params['process']['PIE']['scan']['area'][1] = scan_x
params['process']['common']['scan']['dR'][0] = scan_step
params['process']['common']['scan']['dR'][1] = scan_step
params['process']['common']['detector']['pix_pitch'][0] = adj_px_size
params['process']['common']['detector']['pix_pitch'][1] = adj_px_size
params['process']['common']['detector']['distance'] = camera_length
params['experiment']['detector']['position'][2] = camera_length
params['process']['common']['probe']['convergence'] = convergence_semi_angle *2
params['experiment']['optics']['lens']['alpha'] = convergence_semi_angle *2

params['experiment']['optics']['lens']['defocus'] = [defocus, defocus]
params['process']['common']['detector']['optic_axis'][0] = optic_axis
params['process']['common']['detector']['optic_axis'][1] = optic_axis
params['process']['common']['detector']['crop'][0] = N_x
params['process']['common']['detector']['crop'][1] = N_y
params['process']['common']['detector']['bin'][0] = bin_detector
params['process']['common']['detector']['bin'][1] = bin_detector
params['base_dir'] = base_dir

params['process']['PIE']['iterations'] = iterations
params['process']['PIE']['decay'] = decay


for scan_rot in np.arange(scan_rotation, scan_rotation + (n_rot_step * rot_step), rot_step):
    scan_rot = np.round(100*scan_rot) / 100
    params['process']['common']['scan']['rotation'] = scan_rot
    json_fp = 'Y:/'+ base_dir[14:] +'/pycho_json_rot_step_' + str(scan_rot).replace('.', 'p') + '.json'
    with open(json_fp, 'w+') as outfile:
       json.dump(params, outfile, indent = 4)
       #print(np.round(i*100)/100)
