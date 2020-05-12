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
    #pixelSize = pixelSize * 1e10 # to A
    #_lambda = _lambda * 1e10 # to A
    imageSize = np.asanyarray(imageSize)
    max_probe_rad_target = pixelSize * imageSize[0] / 4
    # print('maximum probe radius (A):', max_probe_rad_target * 1e10)
    #probe_rad_zeroDef = calc_probe_size(pixelSize, imageSize, _lambda, 0, probe_semiAngle, plot_probe = False)
    def_val = 0 #(max_probe_rad_target - probe_rad_zeroDef) / np.tan(probe_semiAngle)
    #print(def_val)
    probe_rad = calc_probe_size(pixelSize, imageSize, _lambda, def_val, probe_semiAngle, plot_probe = False)
    while probe_rad < max_probe_rad_target:
        def_val = def_val + 0.1
        probe_rad = calc_probe_size(pixelSize, imageSize, _lambda, def_val * 1e-10, probe_semiAngle, plot_probe = False)
    def_val = def_val *1e-10
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