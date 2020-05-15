#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:52:15 2020

@author: eha56862
"""


import h5py
import matplotlib.pyplot as plt



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
