#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script reads the pyprismatic sim parameters text file and creates ptypy and
ptyREX scripts to run pty reconstructions on DLS cluster

"""

import numpy as np
from ptypy.core.xy import raster_scan
from ptypy import utils as u
from ptypy import io
import os
from scipy import ndimage as ndi
import h5py
import argparse
import json
import collections

def parse_params_file(params_file, drop_unneeded = True):
    '''
    Reads the parameters text file into a dict to be fed into ptypy / pycho recons
    '''
    exp_dict = {}
    with open(params_file) as f:    
        for line in f: 
            line = line.strip('-')
            exp_dict[line.strip().partition(':')[0]] = line.strip().partition(':')[-1]
    
    
    exp_dict['data'] = exp_dict.pop('output-file')
    exp_dict['xyz'] = exp_dict.pop('input-file')
    exp_dict['accel_voltage(eV)'] = int(exp_dict.pop('energy')) * 1000
    exp_dict['semi_angle(rad)'] = float(exp_dict.pop('probe-semiangle')) * 1e-3
    exp_dict['step_size(m)'] = float(exp_dict.pop('probe-step-x')) * 1e-10
    exp_dict['sim_pixel_size(m)'] = float(exp_dict.pop('pixel-size-x')) * 1e-10
    exp_dict['defocus(m)'] = float(exp_dict.pop('probe-defocus'))* 1e-10
    exp_dict['C3(m)'] = float(exp_dict.pop('C3'))* 1e-10
    exp_dict['C5(m)'] = float(exp_dict.pop('C5'))* 1e-10
    
    electron_rest_mass_eV = 510998.9461
    hc_m_eV = 1.23984197e-6 
    wavelength = hc_m_eV / np.sqrt(exp_dict['accel_voltage(eV)']*(exp_dict['accel_voltage(eV)'] + 2*electron_rest_mass_eV))
    xray_energy_kev = u.nm2keV(wavelength*1e9)
    
    exp_dict['wavelength'] = wavelength
    exp_dict['xray_energy_kev'] = xray_energy_kev
    
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
    # for now hard coding some pty params - to refine later:
    if exp_dict['semi_angle(rad)'] == 0.01:
        exp_dict['pupil_rad(pixels)'] = 6.5
    elif exp_dict['semi_angle(rad)'] == 0.015:
        exp_dict['pupil_rad(pixels)'] = 9.5
    elif exp_dict['semi_angle(rad)'] == 0.02:
        exp_dict['pupil_rad(pixels)'] = 12.5
    elif exp_dict['semi_angle(rad)'] == 0.025:
        exp_dict['pupil_rad(pixels)'] = 15.5
    
    exp_dict['detector_distance(m)'] = 0.143
    exp_dict['detector_pixel_size(m)'] = 220e-6
    
    with h5py.File(exp_dict['data']) as f:
        sh = f['4DSTEM_simulation/data/datacubes/hdose_noisy_data'].shape
    
    if sh[-1] == 128:
        exp_dict['mask'] = '/dls/e02/data/2020/cm26481-1/processing/pty_simulated_data_MD/masks/mask_128by128.h5'
    elif sh[-1] == 72:
        exp_dict['mask'] = '/dls/e02/data/2020/cm26481-1/processing/pty_simulated_data_MD/masks/mask_72by72.h5'
    exp_dict['output_base'] = os.path.dirname(exp_dict['data'])
    
    exp_dict['rotation_angle(degrees)'] = 0
    
    return exp_dict

# TODO: find out the pupil radius from data directly
# def pty_recon_params(data_path):


    
def prep_ptypy_data(exp_dict):
    
    with h5py.File(exp_dict['data']) as f:
        sh = f['4DSTEM_simulation/data/datacubes/hdose_noisy_data'].shape
        print('Dataset shape is %s' % str(sh))
        data = f.get('4DSTEM_simulation/data/datacubes/hdose_noisy_data')
        data = np.array(data)
    mask = io.h5read(exp_dict['mask'])['mask']
    
    th = exp_dict['rotation_angle(degrees)']
    
    data1 = ndi.rotate(data, th, axes=(2,3), reshape=False)
    mask1 = ndi.rotate(mask.astype(int), th, reshape=False)
    
    positions = raster_scan(dy=exp_dict['step_size(m)'], dx=exp_dict['step_size(m)'], ny=data1.shape[1], nx=data1.shape[0])
    d1 = data1.reshape((-1, data1.shape[-2], data1.shape[-1]))
    
    prepared_datafile = os.path.join(exp_dict['output_base'], 'ptypy_' + exp_dict['data'].split('/')[-1])
    io.h5write(prepared_datafile, data=d1, mask =mask1, posx=positions[:,0], posy=positions[:,1])
    
    exp_dict['ptypy_prepared_file'] = prepared_datafile
    
    return exp_dict

def write_ptypy_runfile(exp_dict):
    """
    Writes the .py file that is run for the recon
    """
    
    script_name = 'ptypy_recon_' + exp_dict['data'].split('/')[-1].split('.')[0]+'.py'
    pupil_r = exp_dict['pupil_rad(pixels)']
    merlin_pixel_size = exp_dict['detector_pixel_size(m)']
    #pixel_size = exp_dict['pixel_size']
    detector_distance = exp_dict['detector_distance(m)']
    xray_energy_kev = exp_dict['xray_energy_kev']
    
    # Dump reconstruction script
    with open(os.path.join(exp_dict['output_base'], script_name), 'w') as f:
        f.write(r"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import os
import numpy as np

# Results from analysis
""")
    
        f.write('pupil_diameter = %e\n' % (2*pupil_r*merlin_pixel_size))
        f.write('pixel_size = %e\n' % merlin_pixel_size)
        if exp_dict['defocus(m)'] == 0.0:
            f.write('defocus = %e\n' % 1e-10) # zero defocus case put def as 1A
        else:
            f.write('defocus = %e\n' % exp_dict['defocus(m)'])
        f.write('detector_distance = %e\n' % detector_distance)
        f.write('xray_energy_kev = %e\n' % xray_energy_kev)
        f.write('base_path = "%s"\n' % exp_dict['output_base'])
        f.write('prepared_datafile = "%s"\n' % exp_dict['ptypy_prepared_file'])
        ptydfile = os.path.splitext(exp_dict['ptypy_prepared_file'])[0]+'.ptyd'
        f.write('prepared_ptydfile = "%s"\n' % ptydfile)
    
        f.write(r"""

p = u.Param()
p.verbose_level= 3

# IO
p.io = u.Param()
p.io.autoplot = u.Param()
p.io.autoplot.active = False
p.io.interaction = u.Param()
p.io.interaction.server = u.Param()
p.io.interaction.server.active = True             # Activation switch
p.io.interaction.server.poll_timeout = 10.0       # Network polling interval
p.io.interaction.server.pinginterval = 2          # Interval to check pings
p.io.interaction.server.pingtimeout = 10          # Ping time out

p.io.interaction.client = u.Param()
p.io.interaction.client.poll_timeout = 100.0      # Network polling interval
p.io.interaction.client.pinginterval = 1          # Interval to check pings
p.io.interaction.client.connection_timeout = 3600000.0 # Timeout for dead server

# Scan
p.scans = u.Param()
p.scans.epsic = u.Param()
p.scans.epsic.name = 'Full'
p.scans.epsic.coherence = u.Param()
p.scans.epsic.coherence.num_probe_modes = 1

p.scans.epsic.sample = u.Param()

p.scans.epsic.illumination = u.Param()
p.scans.epsic.illumination.aperture = u.Param()
p.scans.epsic.illumination.aperture.size = pupil_diameter
p.scans.epsic.illumination.aperture.form = 'circ'
p.scans.epsic.illumination.propagation = u.Param()
p.scans.epsic.illumination.propagation.focussed = detector_distance
p.scans.epsic.illumination.propagation.parallel = defocus
#p.scans.epsic.illumination.propagation.antialiasing = 1
p.scans.epsic.illumination.diversity = u.Param()
p.scans.epsic.illumination.diversity.power = 0.1
p.scans.epsic.illumination.diversity.noise = (np.pi, 3.0)

p.scans.epsic.data = u.Param()
p.scans.epsic.data.name = 'Hdf5Loader'
p.scans.epsic.data.dfile = prepared_ptydfile # File path where prepared data will be saved in the ``ptyd`` format.
p.scans.epsic.data.save = 'append'             # Saving mode
p.scans.epsic.data.auto_center = True          # Determine if center in data is calculated automatically
p.scans.epsic.data.load_parallel = 'data'      # Determines what will be loaded in parallel
p.scans.epsic.data.rebin = None                # Rebinning factor
p.scans.epsic.data.orientation = (1, 1, 0)          # Data frame orientation
p.scans.epsic.data.num_frames = None           # Maximum number of frames to be prepared
p.scans.epsic.data.label = 'ePSIC'        # The scan label
p.scans.epsic.data.shape = None                # Shape of the region of interest cropped from the raw data.
p.scans.epsic.data.center = 'fftshift'         # Center (pixel) of the optical axes in raw data
p.scans.epsic.data.psize = pixel_size           # Detector pixel size
p.scans.epsic.data.distance = detector_distance           # Sample to detector distance
p.scans.epsic.data.energy = xray_energy_kev    # Photon energy of the incident radiation in keV

p.scans.epsic.data.intensities = u.Param()
p.scans.epsic.data.intensities.file = prepared_datafile
#p.scans.epsic.data.intensities.key = 'Experiments/MOS2_30kV_Pty1_40Mx_15cm_8C.hdr/data'
p.scans.epsic.data.intensities.key = 'data'

p.scans.epsic.data.positions = u.Param()
p.scans.epsic.data.positions.file = prepared_datafile
p.scans.epsic.data.positions.slow_key = 'posx'
p.scans.epsic.data.positions.fast_key = 'posy'
p.scans.epsic.data.mask = u.Param()            # This parameter contains the mask data.
p.scans.epsic.data.mask.file = prepared_datafile
p.scans.epsic.data.mask.key = 'mask'             # This is the key to the mask entry in the hdf5 file.

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100


#p.engines = u.Param()
#p.engines.ML = u.Param()
#p.engines.ML.name = 'ML'
#p.engines.ML.numiter = 200
#p.engines.ML.ML_type = 'Gaussian'
#p.engines.ML.floating_intensities = False
#p.engines.ML.numiter_contiguous = 10
#p.engines.ML.probe_support = 0.4
#p.engines.ML.reg_del2 = True                      # Whether to use a Gaussian prior (smoothing) regularizer
#p.engines.ML.reg_del2_amplitude = .05             # Amplitude of the Gaussian prior if used
#p.engines.ML.scale_precond = True
#p.engines.ML.scale_probe_object = 1.
#p.engines.ML.smooth_gradient = 10.
#p.engines.ML.smooth_gradient_decay = 1/10.
#p.engines.ML.subpix_start = 0
#p.engines.ML.subpix = 'linear'

P = Ptycho(p, level=5)
""")
    print('Wrote %s' % script_name)

class NestedDefaultDict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))
    
def write_ptyrex_json(exp_dict):
    with h5py.File(exp_dict['data']) as f:
        data = f.get('4DSTEM_simulation/data/datacubes/hdose_noisy_data')
        data_arr = np.array(data)
    
    scan_y = data_arr.shape[1]
    scan_x = data_arr.shape[0]
    
    N_x = data_arr.shape[2]
    N_y = data_arr.shape[3]
    
    # binning = np.floor(256 / N_x)
    
    # adj_px_size = exp_dict['detector_pixel_size(m)'] * binning

    

    params = NestedDefaultDict()
    
    params['process']['gpu_flag'] = 1
    params['process']['save_interval'] = 10
    params['process']['PIE']['iterations'] = 100
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
    params['experiment']['data']['dead_pixel_path'] = exp_dict['mask']
    params['experiment']['data']['flat_field_path'] = exp_dict['mask']
    params['experiment']['data']['load_flag'] = 1
    params['experiment']['data']['meta_type'] = 'hdf'
    params['experiment']['data']['key'] = '4DSTEM_simulation/data/datacubes/hdose_noisy_data'
    
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


def main(params_file, ptypy, ptyrex):
    exp_dict = parse_params_file(params_file)
    print('starting dict: \n', exp_dict)
    
    if ptypy:
        prep_ptypy_data(exp_dict)
        write_ptypy_runfile(exp_dict)
    if ptyrex:
        write_ptyrex_json(exp_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('params_txtFile', help='text file containing all the parameters used for simulation')
    parser.add_argument('ptypy')
    parser.add_argument('ptyrex')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()

    main(args.params_txtFile, args.ptypy, args.ptyrex)
