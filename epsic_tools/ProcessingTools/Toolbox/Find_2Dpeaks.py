# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:44:24 2015

@author: Chris
"""

#find minimum positions in image
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def detect_peaks(data, neighborhood_size = 30,threshold = 100, max_min = 'min' ): # or 'max'

    data_max = filters.maximum_filter(data, neighborhood_size)
    #maxima = (data == data_max) #uncomment this to find maxima
    data_min = filters.minimum_filter(data, neighborhood_size)
    if max_min =='min':
        maxima = (data == data_min)
    else:
        maxima = (data == data_max)
        
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    
    return x, y
