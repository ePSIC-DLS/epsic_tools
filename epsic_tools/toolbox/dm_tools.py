# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:37:27 2020

@author: gys37319
"""
import numpy as np
import hyperspy.api as hs
from skimage.transform import rescale
from skimage.feature import match_template
import matplotlib.pyplot as plt

def get_img_stage_coords(img):
    """
    Returns image stage co-ordinates from a dm3 / dm4 file loaded into hyperspy.
    
    Parameters
    ----------
    img: hyperspy obj

    Returns
    -------
    x, y : tuple 
        x, y stage co-ordinates 
    """
    x = img.original_metadata['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Stage Position']['Stage X']
    y = img.original_metadata['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Stage Position']['Stage Y']
    return [x,y]
    
def get_img_px_size(img):
    """
    Returns image pixel size in metres from a dm3 / dm4 file loaded into hyperspy.
    
    Parameters
    ----------
    img: hyperspy obj

    Returns
    -------
    img_px_size : float 
        image px size in m 
    """
    img_px_size = img.axes_manager[0].scale
    img_px_units = img.axes_manager[0].units
    if img_px_units == 'µm':
        img_px_size = img_px_size *1e-6
    elif img_px_units == 'nm':
        img_px_size = img_px_size *1e-9
    return img_px_size
    
def get_img_FoV(img):
    """
    Returns image field of view in metres from a dm3 / dm4 file loaded into hyperspy.
    
    Parameters
    ----------
    img: hyperspy obj

    Returns
    -------
    img_FoV : float 
        image field of view in m 
    """
    img_px_size = get_img_px_size(img)
    img_px_num = img.axes_manager[0].size
    img_FoV = img_px_num * img_px_size
    return img_FoV
    
def show_ROI(high_res_image, low_res_image):
    """
    Plots the postiion of high_res_image on low_res_image from a dm3 / dm4 file pair loaded into hyperspy.
    
    Parameters
    ----------
    high_res_image: hyperspy object
    low_res_image: hyperspy object

    Returns
    -------
    high_res_rel_center[0], high_res_rel_center[1]
    """
    
    low_res_stage = get_img_stage_coords(low_res_image)
    low_res_px = get_img_px_size(low_res_image)
    high_res_stage = get_img_stage_coords(high_res_image)
    #print('high_res_stage', high_res_stage)
    #print('low_res_stage', low_res_stage)
    high_res_FoV = get_img_FoV(high_res_image) * 1e6

    high_res_rel_pos = [(high_res_stage[0] - low_res_stage[0]),(high_res_stage[1] - low_res_stage[1])]
    #print('high_res_rel_pos', high_res_rel_pos)
    #print('high_res_FoV', high_res_FoV)
    #high_res_rel_px = [high_res_rel_pos[0] / low_res_px,high_res_rel_pos[1] / low_res_px  ]
    #hig_res_FoV_px = high_res_FoV / low_res_px

    #low_res_image.plot()
    centre_px_pos = (low_res_image.axes_manager[0].size / 2) * (1e6 *low_res_px)
    #print('centre_px_pos', centre_px_pos)
    high_res_rel_center = centre_px_pos + high_res_rel_pos[0], centre_px_pos - high_res_rel_pos[1]
    l, t, r, b = high_res_rel_center[0] - high_res_FoV/2, high_res_rel_center[1] - high_res_FoV/2, high_res_rel_center[0]+  high_res_FoV/2 , high_res_rel_center[1] + high_res_FoV/2
    low_res_image.plot()
    cen = hs.markers.point(centre_px_pos, centre_px_pos)
    low_res_image.add_marker(cen)
    #print('high_res_rel_center', high_res_rel_center)
    c  = hs.markers.point(high_res_rel_center[0], high_res_rel_center[1])
    low_res_image.add_marker(c)
    #print(l,t,r,b)
    m = hs.plot.markers.rectangle(x1= l, y1= t, x2= r, y2= b, color='red')
    low_res_image.add_marker(m)
    high_res_file_num = high_res_image.metadata.General.original_filename.split('_')[-1].split('.')[0]
    t = hs.markers.text(high_res_rel_center[0], high_res_rel_center[1], high_res_file_num)
    low_res_image.add_marker(t)
    return high_res_rel_center[0], high_res_rel_center[1]
   
def equalize_res(low_res_image, high_res_image):
    """
    downsample high_res_image so that px size matches low_res_image
    
    Parameters
    ----------
    high_res_image: hyperspy object
    low_res_image: hyperspy object

    Returns
    -------
    bin_high_res_image: hyperspy object
                        downsampled high_res_image
    """ 
    high_res_px = get_img_px_size(high_res_image)
    low_res_px = get_img_px_size(low_res_image)
    scale = low_res_px / high_res_px 
    bin_high_res_image = high_res_image.rebin(scale=[scale, scale])
    #bin_high_res_image.data = bin_high_res_image.data / (scale*scale)
    return bin_high_res_image 
   
def template_match(image, template):
    """
    template match between two hyperspy images with the same px size 
    
    Parameters
    ----------
    image: hyperspy object
    template: hyperspy object

    Returns
    -------
    x,y: tuple
           px position of match
    """ 
    result = match_template(image.data, template.data)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(template.data, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image.data, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hmatch, wmatch = template.data.shape
    rect = plt.Rectangle((x, y), wmatch, hmatch, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()
    return x,y