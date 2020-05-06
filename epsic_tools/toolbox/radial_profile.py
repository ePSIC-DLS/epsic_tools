# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:37:27 2020

@author: gys37319
"""
import numpy as np

#%%
def radial_profile(data, center, nRad = 1 ):
    ''' 
    calculate radial profile for a 2D array
    
    Parameters
    ----------
    
    data : 2D numpy array
    
    center: tuple of x,y center positions
    
    nRad: integer number of radial slices 
    Returns
    -------
    
    radialprofile : nD numpy array corresponding to nRad
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


def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask