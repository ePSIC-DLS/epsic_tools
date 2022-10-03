import numpy as np
#import pyxem as pxm
import h5py
import matplotlib.pyplot as plt
import os

def make_mask(flat_field, hot_pix_factor,mask_cross = False,  show_mask=True, dest_h5_path=None, show_hist=False):
    """
    Creates mask for Merlin Medipix 
    Parameters:
    _____________
    flat_field: str
        full path of the flat-field mib data
    hot_pix_factor: float
        intensity factor above average intensity to determine hot pixels
    mask_cross: bool
        Default False. Mask out the central cross on quad chip
    show_mask: bool
        Default True. For plotting the mask.
    dest_h5_path: str
        default None. full path of the h5 file to be saved. 
        key for mask: 'merlin_mask'. A pyxem version of the mask is also 
        saved in the same destination.
    show_hist: bool
        If passed as True a histogram of the flat-feild data is shown 
        (averaged if the data is a stack)

    Returns
    ________ 
    mask: pyxem ElectronDiffraction2D
        mask pyxem object
    """
    #flat_data = pxm.load_mib(flat_field)
    #flat_data.compute()
    ff =  h5py.File(flat_field, 'r')
    flat_data =ff['Experiments']['__unnamed__']['data'][()]
    ff.close()
    shape = flat_data.shape#flat_data.axes_manager[-1].size * flat_data.axes_manager[-2].size
    # in  case a stack, replace with mean
    #if flat_data.axes_manager[0].size > 1:
    #   flat_data = flat_data.mean()
    if mask_cross:
        mask_cross = np.zeros_like(flat_data)
        if shape[0] == 515:
            mask_cross[255:260,:] = 1
            mask_cross[:, 255:260] = 1
            
        elif shape[0] == 128:
            mask_cross[63:65,:] = 0
            mask_cross[:, 65:65] = 0
            
        flat_data = np.ma.masked_array(flat_data, mask_cross)
        
    if show_hist:
        flat_int_array = np.reshape(flat_data, (shape[0]*shape[1],))
        plt.figure()
        plt.hist(flat_int_array, bins = 100, log = True)
    flat_int_ave = np.mean(flat_data)
    print('flat field intensity average is: ', flat_int_ave)

    mask_dead = flat_data.astype(bool)
    
    mask_hot = flat_data > (flat_int_ave * hot_pix_factor)
    mask_hot_and_dead = np.logical_and(mask_dead, mask_hot)
    # Do we need to save these separately?
    mask = mask_hot_and_dead    
    #mask = pxm.ElectronDiffraction2D(mask)
    if show_mask:
        plt.figure()
        plt.imshow(mask_hot_and_dead)
        plt.figure()
        plt.imshow(mask_dead)
    if dest_h5_path is not None:
        h5f = h5py.File(dest_h5_path, 'w')
        h5f.create_dataset('merlin_mask', data = mask_hot_and_dead)
        h5f.close()
        #pxm_name, ext = os.path.splitext(dest_h5_path)
        #mask.save(pxm_name)
    return mask


def add_to_mask(mask, pix_list, save_path=None):
    """
    Adds an arbitrary list of pixels to a mask.
    Parameters
    __________
    mask: pyxem ElectronDiffraction2D
        mask as pyxem object
    pix_list: list of tuples of ints
        list of pixels expressed as tuples to be added to a mask
    save_path: str
        default None. If desired to save this new version, full path of
        h5 file to be provided here.
    Returns
    ___________
    mask_new: pyxem ElectronDiffraction2D
    """
    mask_new = mask.data
    for pix in range(len(pix_list)):
            mask_new[pix_list[pix][1], pix_list[pix][0]] = False
    if save_path is not None:
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('merlin_mask', data = mask_new)
        h5f.close()
        mask_new = pxm.ElectronDiffraction2D(mask_new)
        pxm_name, ext = os.path.splitext(save_path)
        mask.save(pxm_name)
    else:
        mask_new = pxm.ElectronDiffraction2D(mask_new)
    return mask_new

    
