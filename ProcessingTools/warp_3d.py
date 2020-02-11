# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:15:48 2019

@author: gys37319

Functions to fit and warp 4DSTEM diffraction data
"""

import numpy as np
from numpy.linalg import eig, inv
from skimage.feature import canny
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage import transform as tf
import time
from skimage.util import apply_parallel as sk_par
import dask.array as da


def remove_cross(data):
    ''' 
    normalize bright px and add data to zero px cross in quad chip data
    
    Parameters
    ----------
    
    data : 2D numpy array with bright and zero px cross
    
    
    Returns
    -------
    
    data : 2D numpy array with normalised / filled cross
    '''
    #get mean of columns adjacent to bright columns for all quadrents
    
    vn_tl = data[254,:254].mean()
    vn_bl = data[254,260:].mean()
    vn_tr = data[260,:254].mean()
    vn_br = data[260,260:].mean()
    
    #get mean of rows adjacent to bright rows for all quadrents
    
    hn_tl = data[:254,254].mean()
    hn_bl = data[260:,254].mean()
    hn_tr = data[:254,260].mean()
    hn_br = data[260:,260].mean()
    
    n_all  =np.array([vn_tl, vn_bl, vn_tr, vn_br, hn_tl, hn_bl, hn_tr, hn_br])
    
    #take the mean of bright columns
    
    vb_tl = data[255,:254].mean()
    vb_bl = data[255,260:].mean()
    vb_tr = data[259,:254].mean()
    vb_br = data[259,260:].mean()
    
    #and rows
    
    hb_tl = data[:254,255].mean()
    hb_bl = data[260:,255].mean()
    hb_tr = data[:254,259].mean()
    hb_br = data[260:,259].mean()
    
    b_all  = np.array([vb_tl, vb_bl, vb_tr, vb_br, hb_tl, hb_bl, hb_tr, hb_br])
    
    #get the ratio of normal to bright px
    
    ratio_all = n_all / b_all
    
    print('normal mean : ' , n_all )
    print('bright mean : ' , b_all)
    print('bright:normal ratios : ',  ratio_all  )

    #calculate the mean of all ratios
    
    norm_factor = ratio_all.mean()
    print(norm_factor)
    
    #normalize the bright rows/columns
    data[255,:] = data[255, :] * norm_factor
    data[259,:] = data[259, :] * norm_factor
    data[:, 255] = data[:, 255] * norm_factor
    data[:, 259] = data[:, 259] * norm_factor
    
    # fill in dark px with mean of previous 3 px on row/column
    # fill middle dark px with mean of normalised bright px.  
    
    data[256, : ] = data[253:255,:].mean(axis = 0)
    data[258, : ] = data[259:262,:].mean(axis = 0)
    data[257, : ] = (data[256, :] + data[258, : ]) / 2 
    
    data[:, 256 ] = data[:, 253:255].mean(axis = 1)
    data[: , 258 ] = data[:, 259:262].mean(axis = 1)
    data[:, 257 ] = (data[:, 256] + data[:, 258 ]) / 2 

    return data

def fitEllipse(x,y):
    ''' 
    fit ellipse to x,y co-ordinates
    
    Parameters
    ----------
    
    x : 1D numpy array
    y :1D numpy array
    
    Returns
    -------
    
    a : fit parameters to be fed into functions
    '''
    
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    ''' 
    Calculate ellipse center from fit params
    
    Parameters
    ----------
    
    a : fit parameters
    
    Returns
    -------
    
    x0, y0 as a numpy array
    '''
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_axis_length( a ):
    ''' 
    Calculate ellipse major / minor axis length from fit params
    
    Parameters
    ----------
    
    a : fit parameters
    
    Returns
    -------
    
    len1 , len2 as a numpy array
    '''
    
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    if a>c:
        res = [res1, res2]
    else:
        res = [res2, res1]
    return np.array(res)
    
def ellipse_angle_of_rotation( a ):
    ''' 
    Calculate ellipse angle of rotation from fit params
    
    Parameters
    ----------
    
    a : fit parameters
    
    Returns
    -------
    
    angle in radians
    '''
    
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else: 
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2 
        

def fit_ellipse(data, threshold = 5, sigma = 0, plot_me = False):
    ''' 
    function to fit ellipse to a single image
    
    Parameters
    ----------
    
    data : Input image, 2D numpy array
    thershold : threshold for binarising image, integer
    plot_me : boolean to plot results
    
    Returns
    -------
    
    center, phi, axes
    '''
    
    # create binary image
    data_binary = np.where(data < data.mean()*threshold , 0, 1)
    # fill in any dead px holes
    data_binary = binary_fill_holes(data_binary)
    # get edges 
    edges = canny(data_binary, sigma=sigma,low_threshold=0.1, high_threshold=0.2)
    
    image_size = data_binary.shape
    
    # construct points for fit
    xi = np.linspace(0, image_size[1] - 1, image_size[1])
    yi = np.linspace(0, image_size[0] - 1, image_size[0])
    x, y = np.meshgrid(xi, yi)
    
    pts = np.array([x[edges > 0].ravel(), y[edges > 0].ravel()])
    
    # and fit
    res = fitEllipse(pts[0],pts[1])
    center = ellipse_center(res)
    phi = ellipse_angle_of_rotation(res)
    axes = ellipse_axis_length(res)
    
    
    if plot_me == True:
        plot_ellipse(data, [center, phi, axes])
        #plot_ellipse(data_binary, [center, phi, axes])
        plt.figure()
        plt.imshow(edges)
    return center, phi, axes

def plot_ellipse(data, params, line_width = 2):
    ''' 
    function to plot ellipse fits 
    
    Parameters
    ----------
    
    data : Input image, 2D numpy array
    params : center [x0,y0], phi (in rad), axes [major,minor]
    line_width : thickness of plot line
    
    Returns
    -------

    '''
    R = np.arange(0,2*np.pi, 0.01)
    center = params[0]
    phi = params[1]
    axes = params[2]
    a, b = axes
    
    xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)

    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(data)
    ax.plot(xx, yy, 'b-', lw = line_width)

def compute_transform(data, params):
    ''' 
    compute the tranform from fit parameters
    
    Parameters
    ----------
    
    data : Input image, 2D numpy array
    params : center [x0,y0], phi (in rad), axes [major,minor]
    
    
    Returns
    -------
    transformation
    '''
    image_size = data.shape
    
    b2= params[2][1]**2
    a2 = params[2][0]**2
    phi2 = -params[1]#1.149
    scaling = np.array([[1, 0], [0, (a2/b2)**-0.5]])
    rotation = np.array([[np.cos(phi2), -np.sin(phi2)],[np.sin(phi2), np.cos(phi2)]])
    correction = np.linalg.inv(np.dot(rotation.T,np.dot(scaling, rotation)))
    #compute affine array
    affine = np.array([[correction[0, 0], correction[0, 1], 0.00],
                       [correction[1, 0], correction[1, 1], 0.00],
                       [0.00, 0.00, 1.00]])
    
    shift_x = (image_size[1] - 1) / 2
    shift_y = (image_size[0] - 1) / 2
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
    
    distortion = tf.AffineTransform(matrix=affine)
    transformation = (tf_shift + (distortion + tf_shift_inv)).inverse
    
    return transformation


def get_coords_4d(coords, shape_4d):
    ''' 
    build the 4d transformation co-ordinates 
    
    Parameters
    ----------
    
    coords : 2D coordinates
    shape_4d : shape of the 4d data set to be transformed
    
    
    Returns
    -------
    co_ords4d : 4d coordinates for the transform
    '''
    
    dat_type = 'float16'
    coords = coords.astype(dat_type)
    co_ords4d_a = np.tile(coords[0], (shape_4d[0],shape_4d[1],1,1))
    co_ords4d_a = co_ords4d_a[np.newaxis, :,:,:,:]
    co_ords4d_b = np.tile(coords[1], (shape_4d[0],shape_4d[1],1,1))
    co_ords4d_b = co_ords4d_b[np.newaxis, :,:,:,:]
    z = np.arange(shape_4d[0], dtype = dat_type)
    Z = np.ones_like(co_ords4d_a, dtype = dat_type)
    Z = Z * z[None, :, None,  None, None]
    
    z2 = np.arange(shape_4d[1], dtype = dat_type)
    Z2 = np.ones_like(co_ords4d_a, dtype = dat_type)
    Z2 = Z2 * z2[None, None,  :, None, None] 
    
    co_ords4d =  np.append(Z, Z2, axis = 0)
    co_ords4d =  np.append(co_ords4d, co_ords4d_a, axis = 0)
    co_ords4d =  np.append(co_ords4d, co_ords4d_b, axis = 0)
    #print(co_ords3d.dtype)
    return co_ords4d
    
    
def warp_all_np(data, coords,  order = 1, preserve_range = True, plot_me = False):
    ''' 
    apply transformation to full 4d data set 
    
    Parameters
    ----------
    
    data : inpud 4d data set (dask array)
    coords : 4D coordinates
    plot_me : boolean
    
    
    Returns
    -------
    dat_temp : warped 4D data 
    '''
    shape_4d = data.shape
    #get rid of hot px
    #data[data > 20* data.mean()] = data.mean()

    #print(coords.shape)
    #t0 = time.time()
    co_ords3d =get_coords_4d(coords, shape_4d)# sk_par(get_coords_3d, coords,extra_keywords = {'shape_4d': shape_4d})
   
    #print(time.time() - t0)
    co_ords3d = co_ords3d.astype('float32')
    #t1 = time.time()

    dat_temp = tf.warp(data, co_ords3d, order = order, preserve_range = preserve_range)

    #print('time : ', time.time() - t1)
    im_num = 5,5
    if plot_me == True:
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (8,8))
        
        ax1.imshow(dat_temp[im_num[0],im_num[1], :, : ])
        ax2.imshow(data[im_num[0],im_num[1], :, :])
        
    #print('dat : ', dat_temp[im_num[0],im_num[1], :, : ].min(), dat_temp[im_num[0],im_num[1], :, : ].max(), dat_temp[im_num[0],im_num[1], :, : ].mean(), sc.ndimage.measurements.center_of_mass(dat_temp[im_num[0],im_num[1], :, : ]))
    #print('dat : ', data[im_num[0],im_num[1], :, : ].min(), data[im_num[0],im_num[1], :, : ].max(), data[im_num[0],im_num[1], :, : ].mean(), sc.ndimage.measurements.center_of_mass(data[im_num[0],im_num[1], :, : ]))
    
    return dat_temp


def warp_all(data, coords, dat_type, plot_me = False):
    ''' 
    NOT CURRENTLY USED
    
    apply transformation to full 4d data set 
    
    Parameters
    ----------
    
    data : inpud 4d data set (dask array)
    coords : 4D coordinates
    plot_me : boolean
    
    
    Returns
    -------
    dat_temp : warped 4D data 
    '''
    #correct for distortions in all diff patterns 
    #working for 4D data ONLY
    coords = da.array(coords)
    #dat_type = 'float32'
    shape_4d = data.shape#test_dp.shape
    #shape_3d = #test_dp.shape
    #get rid of hot px
    data[data > 20* data.mean()] = data.mean()
    
    co_ords_shape = list(shape_4d)
    co_ords_shape.insert(0, 1)
    co_ords_shape = tuple(co_ords_shape)
    
    co_ords3d_a = da.stack([coords[0] for i in range(shape_4d[0])], axis = 2)
    co_ords3d_a = da.stack([co_ords3d_a for i in range(shape_4d[0])], axis = 3)
    co_ords3d_a = co_ords3d_a.T
    
    co_ords3d_a = co_ords3d_a.reshape(co_ords_shape)
    print(type(co_ords3d_a))
    co_ords3d_b = da.stack([coords[1] for i in range(shape_4d[0])], axis = 2)
    co_ords3d_b = da.stack([co_ords3d_b for i in range(shape_4d[0])], axis = 3)
    co_ords3d_b = co_ords3d_b.T
    
    co_ords3d_b = co_ords3d_b.reshape(co_ords_shape)
    
    z = np.arange(shape_4d[0], dtype = dat_type)
    Z = da.ones_like(co_ords3d_a, dtype = dat_type)#(shape = (100, 128, 128))
    Z = Z * z[None, :, None,  None, None]
    
    z2 = np.arange(shape_4d[1], dtype = dat_type)
    Z2 = da.ones_like(co_ords3d_a, dtype = dat_type)#(shape = (100, 128, 128))
    Z2 = Z2 * z2[None, None,  :, None, None] 
    
    co_ords3d=  da.concatenate((Z, Z2), axis = 0)
    co_ords3d =  da.concatenate((co_ords3d, co_ords3d_a), axis = 0)
    co_ords3d =  da.concatenate((co_ords3d, co_ords3d_b), axis = 0)
    print(type(co_ords3d_a), type(co_ords3d_b), type(co_ords3d), type(Z), type(Z2))
    #co_ords3d = co_ords3d.compute()
    print(co_ords3d.shape, co_ords3d.dtype)
    t1 = time.time()
    #dat_temp = tf.warp(data,  inverse_map = co_ords3d , order =1, preserve_range  = True)
    kwords_dict = {
            'inverse_map': co_ords3d,
            #'coordinates': co_ords3d,
            'order': 1,
            #'dtype': dat_type
            #'preserve_range': True            
            }
    co_ords3d = co_ords3d.compute()
    data = data.compute()
    dat_temp = sk_par(tf.warp, data, extra_keywords=kwords_dict)
    #dat_temp = sk_par(map_coordinates, data, extra_keywords=kwords_dict)#tf.warp(data, inverse_map = co_ords3d)
    print('time : ', time.time() - t1)
    im_num = 5,5
    if plot_me == True:
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (8,8))
        
        ax1.imshow(dat_temp[im_num[0],im_num[1], :, : ])
        ax2.imshow(data[im_num[0],im_num[1], :, :])
        
    #print('dat : ', dat_temp[im_num[0],im_num[1], :, : ].min(), dat_temp[im_num[0],im_num[1], :, : ].max(), dat_temp[im_num[0],im_num[1], :, : ].mean(), sc.ndimage.measurements.center_of_mass(dat_temp[im_num[0],im_num[1], :, : ]))
    #print('dat : ', data[im_num[0],im_num[1], :, : ].min(), data[im_num[0],im_num[1], :, : ].max(), data[im_num[0],im_num[1], :, : ].mean(), sc.ndimage.measurements.center_of_mass(data[im_num[0],im_num[1], :, : ]))
    
    return dat_temp, co_ords3d

    
    