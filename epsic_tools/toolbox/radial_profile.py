# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:37:27 2020

@author: gys37319
"""
import numpy as np

#%%
def radial_profile(data, center):
    ''' 
    calculate radial profile for a 2D array
    
    Parameters
    ----------
    
    data : 2D numpy array
    
    center: tuple of x,y center positions
    
    
    Returns
    -------
    
    radialprofile : 1D numpy array 
    '''
    
    x,y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = np.rint(r - 0.5).astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = np.nan_to_num(tbin / nr)
    return radialprofile 

def radial_profile_stack(hs_obj, center= None):
    ''' 
    calculate radial profile from a hyperspy Signal2D object
    
    Parameters
    ----------
    
    hs_obj : hyperspy signal2D object
    
    center: tuple of x,y center positions
    
    
    Returns
    -------
    
    radial_profiles : hyperspy signal1D object 
    
    '''
    if center == None:
        center = ((hs_obj.data.shape[-1] / 2) - 0.5, (hs_obj.data.shape[-2] / 2) - 0.5)
        #print(center)
    radial_profiles = hs_obj.map(radial_profile,inplace = False, parallel = True,  center = center)
    radial_profiles = radial_profiles.as_signal1D((-1))
    return radial_profiles