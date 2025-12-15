#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:43:42 2025

@author: ejr78941
"""

#imports

import h5py
import hyperspy.api as hs
import sys
import os
import numpy as np
import dask.array as da


def dask_bin(array,binning):
    if len(binning) == 2:
        dask_array = da.asarray(array)
        dask_shape = dask_array.shape
        print(dask_shape)
        new_shape  = (dask_shape[0],dask_shape[1],dask_shape[2]/binning[0],binning[0],dask_shape[3]/binning[1],binning[1])
        dask_reshape = da.reshape(dask_array,new_shape)
        dask_binned = da.sum(dask_reshape,axis=(3,5))
        return np.asarray(dask_binned,dtype=np.uint32)
    else:
        print('binning is expect to be tuple of two elements')
        
        
        
def main():
    ''' determine inputs from the command line'''

    '''due how the data is saved in E01 the save name has spaces which causes issues 
    #when inputting the name into the command line so this line recombines all 
    #of the strings back together in the correct way - assumming the how the files 
    #are saved does not change'''

    data_path = ''
    counter = 0 
    print(len(sys.argv))
    for x in sys.argv:
        #print(counter)
        if counter != 0 and counter != len(sys.argv)-1: 
            if counter == 1:
                data_path = x
            else:
                data_path =  data_path + ' ' + x
        counter = counter + 1
    print('\n')
    
    binning = int(sys.argv[len(sys.argv)-1])
    print(data_path)
    print(binning)
    
    #load the data from the path provided by the command line
    dps = hs.load(data_path)

    #find the step_size of the scan
    step = np.array(dps._original_metadata['ImageList']['TagGroup0']['ImageData']['Calibrations']['Dimension']['TagGroup2']['Scale'])
    #print the units of the step size for reference
    print(dps._original_metadata['ImageList']['TagGroup0']['ImageData']['Calibrations']['Dimension']['TagGroup2']['Units'])

    #find the pixel size and binned pixel size
    pixel_size = dps._original_metadata['ImageList']['TagGroup0']['ImageTags']['Acquisition']['Frame']['CCD']['Pixel_Size_um'][0]*1e-6
    print(f'The pixel of the experiment: {pixel_size}')
    unbinned_pixel_size = dps._original_metadata['ImageList']['TagGroup0']['ImageTags']['Acquisition']['Device']['CCD']['Pixel_Size_um'][0]*1e-6
    print(f'the binned pixel size of the detector is: {unbinned_pixel_size}')
    
    #output the shape of the loaded data to confrim its resonable
    print(np.shape(dps))
    
    #bin the data by the desired amount, it is assummed that binning is uniform in the detector
    if binning >= 2:
        binned_dps = dask_bin(dps.data,(binning,binning))
    else:
        binned_dps = dps

    print(f'pixel size after post experiment binnig: {binning*pixel_size}')
    
    #determine save path from the directory the data was loaded from
    save_path, file = os.path.split(data_path)
    save_path = save_path + '/'
    print('save_path =', save_path)
    #index = data_path.find('(')
    #save_num = data_path[index+1]
    default_prefix = 'data'
    default_postfix = '.hdf5'
    save_name = save_path + default_prefix + default_postfix # + save_num
    print('save_name =', save_name)
    
    #now save the data to associated directory with a consistent key
    save_dir = data_path
    key = '/data/frames'
    with h5py.File(save_name, 'w') as f:
        dset = f.create_dataset(name=key, data=binned_dps.data)
        try:
            dset = f.create_dataset(name='/scan/scale', data=step)
            dset = f.create_dataset(name='/detector/pixel_size', data=pixel_size*binning)
            dset = f.create_dataset(name='/detector/unbinned_pixel_size', data = unbinned_pixel_size)
        except:
            print('meta data was not in the excepted place')
    print('data reformating and binning complete')
    
if __name__ == "__main__":
    main()
