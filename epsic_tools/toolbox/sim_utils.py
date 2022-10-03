#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:31:39 2020

@author: eha56862
These functions are utilities for determining the parameters for the simulation matrix
"""
import numpy as np
import hyperspy.api as hs
import h5py
import matplotlib.pyplot as plt
import os
import collections
import json

from epsic_tools.toolbox.ptycho_utils import e_lambda




def sim_to_hs(sim_h5_file, h5_key = 'hdose_noisy_data'):
    '''
    reads simulated 4DSTEM file into hs object
    Parameters
    __________
    sim_h5_file: str
        full path of the simulated 4DSTEM dataset h5 file
    h5_key: str
        the h5 key of the dataset - default is 'hdose_noisy_data'
        if h5_key is passed as 'skip': for skipped probe data
        if h5_key is passed as 'raw': as-output sim - each dp sums to ~1 intensity
        if another h5_key is provided that location is looked up for data
        
    Returns
    ________
    data_hs: hyperspy Signal2D object
    '''
    if h5_key == 'hdose_noisy_data':
        with h5py.File(sim_h5_file, 'r') as f:
            sh = f['4DSTEM_simulation/data/datacubes/hdose_noisy_data'].shape
            print('Dataset shape is %s' % str(sh))
            data = f.get('4DSTEM_simulation/data/datacubes/hdose_noisy_data')
            data = np.array(data)
            data_hs = hs.signals.Signal2D(data)
    elif h5_key == 'skip':
        with h5py.File(sim_h5_file, 'r') as f:
            sh = f['dataset'].shape
            print('Dataset shape is %s' % str(sh))
            data = f.get('dataset')
            data = np.array(data)
            data_hs = hs.signals.Signal2D(data)
        
    elif h5_key == 'raw':
        with h5py.File(sim_h5_file, 'r') as f:
            sh = f['4DSTEM_simulation/data/datacubes/CBED_array_depth0000/datacube'].shape
            print('Dataset shape is %s' % str(sh))
            data = f.get('4DSTEM_simulation/data/datacubes/CBED_array_depth0000/datacube')
            data = np.array(data)
            data_hs = hs.signals.Signal2D(data)
    else:
        with h5py.File(sim_h5_file, 'r') as f:
            sh = f[h5_key].shape
            print('Dataset shape is %s' % str(sh))
            data = f.get(h5_key)
            data = np.array(data)
            data_hs = hs.signals.Signal2D(data)
    return data_hs



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
    
    json_file = os.path.join(exp_dict['output_base'], 'ptyREX_' + os.path.splitext(exp_dict['data'])[0].split('/')[-1] + '.json')
    exp_dict['ptyREX_json_file'] = json_file    
    with open(json_file, 'w+') as outfile:
        json.dump(params, outfile, indent = 4)




def get_bf_disc(data_hs):
    """
    Interactively gets the radius and centre position of the bright filed disc
    :param
    data_hs: hyperspy Signal2D object

    :return:
    circ_roi: hs roi object
        roi object with centre and radius
    """
    cent_x = int(data_hs.axes_manager[-1].size / 2)
    cent_y = int(data_hs.axes_manager[-2].size / 2)
    circ_roi = hs.roi.CircleROI(cent_x,cent_y, 10)
    data_sum = data_hs.sum()
    data_sum.plot()
    imc = circ_roi.interactive(data_sum)

    return circ_roi
    


def get_adf(data_hs, bf_rad):
    """
    provides an adf image - as outer angle it defaults to centre pixel position plus the bf disc radius
    and as inner it uses the bf disc radius
    :param data_hs: hyperspy Signal2D object

    :param bf_rad: int
        radius of the bright field disc in pixels
    :return:
    adf: hyperspy Signal2D object

    """
    scale = data_hs.axes_manager[3].scale
    cent_x = int(data_hs.axes_manager[3].size * scale / 2)
    cent_y = int(data_hs.axes_manager[2].size * scale / 2)
    circ_roi = hs.roi.CircleROI(cent_x,cent_y, bf_rad+cent_x, bf_rad)
    data_T = data_hs.T
    data_T.plot()
    imc = circ_roi.interactive(data_T)
    imc.sum().plot()
    adf = imc.sum()
    return adf

def calc_camera_length(data_hs, bf_rad, angle, pixel_size):
    '''
    it returns the camera length based on the detector pixel size and a known value
    in the diffraction plane - it plots the sum dp with the known value marked with circle roi
    input:
        data_hs: 4D-STEM hypespy object
        bf_rad: int
            bf radius in pixels
        angle: float
            known angle in diff plane in rad
        pixel_size: float
            physcial size of detector pix in m
    returns
        CL: float
            camera length in meters
    '''
    CL = bf_rad * pixel_size / angle
    data_sum = data_hs.sum()
    cent_x = int(data_hs.axes_manager[-1].size / 2)
    cent_y = int(data_hs.axes_manager[-2].size / 2)
    circ_roi = hs.roi.CircleROI(cent_x,cent_y, bf_rad)
    data_sum.plot()
    circ_roi.interactive(data_sum)
    return CL

    
def get_disc_overlap(rad, dist, print_output=False):
    """
    More suitable for Wigner - as it aims to minimse triple overlap regions
    for ptyREX will be using the simpler function get_overlap
    to calculate disc over lap in image or diff plane

    rad: float
        radius of probe or probe semi-angle
    dist: float
        step distance or the angle to the reflection of interest
    returns
    
    Percentage_overlap: float
        
    """
    x_pos  = 0.5 * dist # x coordinate of circle intersection
    y_pos = np.sqrt(rad**2 - x_pos**2) # y coordinate of circle intersection
    theta = 2*np.arctan(y_pos / x_pos) # angle subtended by overlap
    A_overlap =  (rad**2 * theta) - (2*x_pos*y_pos) # area of overlap
    A_probe = np.pi * rad**2
    Percentage_overlap =100 *  A_overlap / A_probe
    if print_output:
        print('x, y : ', x_pos, y_pos)
        print('theta : ', theta )
        print('overlap area : ', A_overlap)
        print('probe area : ', A_probe)
        print('overlap  % : ', Percentage_overlap)
    return Percentage_overlap


def get_overlap(probe_rad, step_size):
    """
    probe_rad: float
        probe radius in m (or rad)
    step_size: float
        scan step size in m (or known diffraction disc position in rad)
    Returns
    probe_overlap: float
        percentage probe overlap
    """
    probe_overlap = 1 - step_size / (2 * probe_rad)
    
    return 100 * probe_overlap


def get_step_size(probe_rad, target_overlap):
    """
    knowing the probe radius and the target overlap percentage this function returns
    the suitable step size.
    Parameters
    ___________
    probe_rad: float
        probe radius in m
    target_overlap: float
        overlap fraction
    Returns
    _________
    step_size: float
        the step size  in m needed to get the target overlap
    """
    step_size = (1 - target_overlap) * (2 * probe_rad)
    
    return step_size


def calc_pixelSize(acc_voltage, pixel_array, det_pixelSize, camera_length):
    """
    Calculates the pixelSize in ptycho recon
    
    Parameters
    _____________
    acc_voltage: int
        accelerating voltage in V
    pixel_array: int
        number of pixels in detector in x or y (assumed square)
    det_pixelSize: float
        detector physical pixel size in m
    camera_length: float
        camera length
    Returns
    _________
    pixelSize: float
        recon pixel size in m
    """
    l = e_lambda(acc_voltage)
    theta = (pixel_array * det_pixelSize) / camera_length
    pixelSize = l / theta
    
    return pixelSize


def calc_probe_size(pixelSize, imageSize, _lambda, probe_def, probe_semiAngle, \
                    plot_probe=True, return_probeArr = False):
    """
    this function is for giving an estimate of the probe size to set up the sim ptycho data accordingly.
    :param pixelSize: float
        pixel size in (m)
    :param imageSize: list of ints
        image size in (pixels)
    :param _lambda: float
        in (m) - electron wavelength
    :param probe_def: float
        probe defocus in m
    :param probe_semiAngle: float
        (rad) probe semi-angle
    :param plot_probe: boolean
        default to True - to plot the probe imag / real
    :param return_probeArr: boolean
        default to False - if the probe array is needed as output
    :return:
    * plots probe in the real and fourier space - if plot_probe is True
    probe_rad: float
        in (m) probe radius
        if return_probeArr set to True:
            [probe_rad, psiShift]: with
            psiShift: np.array
                probe in real space
    """
    pixelSize = pixelSize * 1e10 # to A
    _lambda = _lambda * 1e10 # to A
    probe_def = probe_def * 1e10 # to A
    imageSize = np.asanyarray(imageSize)
    [qxa,qya] = makeFourierCoordinates(imageSize,pixelSize)
    q2 = qxa**2 + qya**2
    
    # real space coordinates
    x = (np.arange(imageSize[0])+1) * pixelSize
    y = (np.arange(imageSize[1])+1) * pixelSize
    [ya,xa] = np.meshgrid(y,x)
    
    # Make probe in Fourier space, apply defocus
    qMax = probe_semiAngle / _lambda
    chi = np.pi*_lambda*q2*probe_def
    Psi = [q2 <= qMax][0] * np.exp(-1j*chi)
    # Probe in real space, shifted probe
    psi = np.fft.ifft2(Psi)
    psiShift = np.fft.fftshift(psi)

    # Calculate probe size
    probeInt = abs(psiShift)**2
    
    # First, probe origin

    x0 = np.sum(probeInt*xa) / np.sum(probeInt)
    y0 = np.sum(probeInt*ya) / np.sum(probeInt)
    # RMS probe size
    rmsProbe = np.sum(np.ravel(probeInt)*(np.ravel(xa) - x0)**2) / np.sum(probeInt) \
        + np.sum(np.ravel(probeInt)*(np.ravel(ya) - y0)**2) / np.sum(probeInt)

    # Write probe size to console
    # print('Probe RMS size = %2.3f'%rmsProbe ,'Angstroms')
    
    probe_rad = rmsProbe / 2
    probe_rad = probe_rad * 1e-10 # to (m)

    if plot_probe:
        fig, axs = plt.subplots(1,2, figsize=(5, 5))
        im1 = axs[0].imshow(np.fft.fftshift(abs(Psi)))
        axs[0].set_title('Fourier space')
        fig.colorbar(im1, ax = axs[0])
        im2 = axs[1].imshow(abs(psiShift))
        axs[1].set_title('Real space')
        fig.colorbar(im2, ax = axs[1])
    if return_probeArr:
        return [probe_rad, psiShift]

    return probe_rad
    

def max_defocus(pixelSize, imageSize, _lambda, probe_semiAngle):
    """
    this function returns the max defocus to be used for ptycho experiment given 
    the Nyquist restrictions on the probe size.
    
    Input
    ________
    pixelSize: float
        pixel size in (m)
    imageSize: list of ints
        image size in (pixels)
    _lambda: float
        (m) electron wavelength
    probe_semiAngle: float
        (rad) probe semi-angle
    
    Returns
    _________
    max_def: float
        max defocus in (m)
    """
    imageSize = np.asanyarray(imageSize)
    max_probe_rad_target = pixelSize * imageSize[0] / 4
    print('target probe radius(m):', max_probe_rad_target)
    def_val = max_probe_rad_target / np.tan(probe_semiAngle) 
    probe_rad = calc_probe_size(pixelSize, imageSize, _lambda, def_val, probe_semiAngle, plot_probe = False)
    if probe_rad < max_probe_rad_target:
        while probe_rad < max_probe_rad_target:
            def_val = def_val + 1e-10
            probe_rad = calc_probe_size(pixelSize, imageSize, _lambda, def_val, probe_semiAngle, plot_probe = False)
    else:
        while probe_rad > max_probe_rad_target:
            def_val = def_val - 1e-10
            probe_rad = calc_probe_size(pixelSize, imageSize, _lambda, def_val, probe_semiAngle, plot_probe = False)

    def_val = def_val
    return def_val


def makeFourierCoordinates(N, pixelSize):
    """
    this function creates a set of coordinates in the Fourier space
    Input
    ________
    N: np.array of int
        image size in pixels. np.array((Nx,Ny))
    pixelSize: float
        pixel size in A
    
    Returns
    ________
    qx: np.array 
    qy: np.array 
        fourier coordinates
    """
    
    qx = np.roll(np.arange(int(-N[0]/2),int(N[0]/2))/(N[0]*pixelSize), int(N[0]/2))
    qy = np.roll(np.arange(int(-N[1]/2),int(N[1]/2))/(N[1]*pixelSize), int(N[1]/2))
   
    qx, qy = np.meshgrid(qx, qy)
    return qx, qy
  
    
def get_potential(sim_file_path):
    '''
    gets the pyprismatic h5 file and outputs the potential - in V.Angstroms
    '''
    with h5py.File(sim_file_path, 'r') as f:
        pots = f['4DSTEM_simulation']['data']['realslices']['ppotential']['realslice'][:]
        pots = np.squeeze(pots)
    return pots



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



def _sigma(e_0):
    """
    From Pete Nellist MATLAB code
    return the interaction parameter sigma in radians/(Volt-Angstroms)
    ref: Physics Vade Mecum, 2nd edit, edit. H. L. Anderson
    The American Institute of Physics, New York) 1989
     page 4.

    :param e_0: accelerating voltage in eV
    :return: sigma - the interaction parameter sigma in radians/(Volt-Angstroms)
    """
    emass = 510.99906 # electron rest mass in keV
    l = e_lambda(e_0)*1e10 # wavelength in A
    x = (emass + e_0 / 1000) / (2.0 * emass + e_0 / 1000)
    s = 2.0 * np.pi * x / (l * e_0 / 1000)
    s = s / 1000 # in radians / (V.A)
    return s

def json_to_dict_sim(json_path):
    with open(json_path) as jp:
        json_dict = json.load(jp)
    json_dict['json_path'] = json_path
    base_path = os.path.dirname(json_path)
    for file in os.listdir(base_path):
        if file.endswith('h5'):
            sim_path = os.path.join(base_path, file)
    json_dict['sim_path'] = sim_path
    return json_dict


def shift_probe(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X
