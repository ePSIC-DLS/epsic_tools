#Alpha 0.01 Frederick Allars 03-10-2025

'''path imports'''
import sys
import os
import glob

'''load imports'''
import h5py

'''Visulise imports'''
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import skimage
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


'''does stuff imports'''
import numpy as np
from skimage.restoration import unwrap_phase
from PIL import Image
import imageio
from skimage.filters import window
import json
import cv2
#import atomap.api as am
import scipy
import pandas as pd
from scipy.ndimage import gaussian_filter as gf
import tifffile


'''Error imports'''
import traceback

'''command line imports'''
import subprocess


'''Ipython widgets'''
import ipywidgets
from ipywidgets.widgets import *



'''general readme for the flow/catorgation of this code'''
'''
step 0 misc functions needed by other functions
step 1: open the data and get into numpy array
step 2 make sure the array is square and minial diemonsions
step 3 some filtering functions which can be applied to the stack of images if needed (can skip)
step 4 auto create figure functions
step 5 create image functions tiff/png/saved numpy arrays
step 6 create videos
step 7 whole experiment processing videos and excel sheets
step 8 ipython widget wrappers for the above functions

the code should work in that higher functions makes calls to function below them in step value
'''

'''varible list'''
'''
recon_path - the location of the recon in the e02/dls file system will be constructed from basir, year, session, subfolder, etc


holder - a temporary array (2D/3D) within a function 

#ToDO make a conistent list of varible names and keep updated


'''

'''----------Misc functions----------'''

def quick_fft(array):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array)))
def quick_ifft(array):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(array)))

def QuickPixelSize(recon_path,flux_type='electron'):
    #load parameters from the corresponding json file to determine the rotation and real space pixel size
    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        rot = f['process']['common']['scan']['rotation']

        try:
            CL   = f['experiment']['detector']['position'][-1]
        except:
            print('Camera length was not in expected place')
            try:
                CL   = f['experiment']['detector']['position_z']
            except Exception as err:
                print(f'An error has occurred, this likely due to the camera length key used (but might be something else)...\n{err}={type(err)}')
                pass
            
        N    = f['process']['common']['detector']['crop'][-1]
        detx = f['process']['common']['detector']['pix_pitch'][-1]

        try:
            energy  = f['process']['common']['source']['energy'][-1]
        except:
            try:
                energy  = f['process']['common']['source']['energy']
            except Exception as err:
                print(f'unable to obtain the energy of the {flux_type}\n{err}={type(err)}')
            
        save_name = f['process']['save_prefix']
        print(save_name)

    #determine the real space and Fourier space pixel Size
    if flux_type == 'electron':
        wav = (1.23e3/(np.sqrt(energy*(1+9.78e-7*energy))))*1e-12
    elif flux_type == 'photon':
        wav = 6.626e-34*0.299792458e9/(1.60217e-19*energy)

    theta = N*detx/CL
    sp_dx = wav/theta
    return sp_dx


'''----------functions which load the data in to numpy arrays for post processing----------'''


#def ptyrex2numpy(recon_path,load_list,verbose=False):
#    '''
#    recon_path   - path to the recon hdf to be loaded
#    load_list    - a list of hdf keys to load from the reconstruction
#    vebose       - print all of the things being loaded
#    ptyrex_recon - a dictionary containing all elements requested in load_list
#    '''
#    ptyrex_recon = dict()
#    ptyrex_recon['recon_path'] = recon_path
#    with h5py.File(recon_path,'r') as f:
#        for num, key in enumerate(load_list):
#            if verbose:
#                print(f'{num}:{key}')
#            ptyrex_recon[key] =  f['entry_1']['process_1']['output_1'][key][()]
#    return ptyrex_recon


def ptyrex2numpy(recon_path,load_list,verbose=False):
    '''
    recon_path   - path to the recon hdf to be loaded
    load_list    - a list of hdf keys to load from the reconstruction
    vebose       - print all of the things being loaded
    ptyrex_recon - a dictionary containing all elements requested in load_list
    '''
    ptyrex_recon = dict()
    ptyrex_recon['recon_path'] = recon_path
    with h5py.File(recon_path,'r') as f:
        for num, key in enumerate(load_list):
            if verbose:
                print(f'{num}:{key}')
            ptyrex_recon[key] =  f['entry_1']['process_1']['output_1'][key][()]
    
    '''perfrom inital post-processing such reducing object/probe dims'''
    for key, val in ptyrex_recon.items():
        if key in {'object_phase', 'object_modulus'}:
            if verbose:
                print(key)
            json_dir = ptyrex_recon['recon_path'].replace('.hdf','.json')
            with open(json_dir) as json_data:
                f = json.load(json_data)
                rot = f['process']['common']['scan']['rotation']
            #ToDO better way of precutting the reconstruction
            ptyrex_recon[key] = reduce_rotate(ptyrex_recon[key],0,np.shape(ptyrex_recon[key])[-2],0,np.shape(ptyrex_recon[key])[-1],-rot)
        elif key in  {'probe' or 'probe_modulus' or 'probe_phase'}:
            if verbose:
                print(key)
            ptyrex_recon[key] = reduce_probes(ptyrex_recon[key])


    return ptyrex_recon


'''----------functions which reduce the dimensions of ptyrex data so then can be easily accessed----------'''

def reduce_rotate(slices,left,right,top,bottom,angle):
    '''
    Electron ptyrex data starts as a rotated 7 dimensional array
    this function takes and reduces it to a 3D array .i.e. a through thickness
    stack of images

    objects have the following strucure
    0 - empty (in theory object modes)
    1 - object slices
    2 - objects from different wavelengths/energies
    3 - empty
    4 - empty (position corrrection)
    5/6 - sptial array (actual object)

    '''
    holder = np.zeros([np.shape(slices)[1],right-left,bottom-top])
    reduce_dim = slices[0,:,0,0,0,:,:]
    for counter1 in np.arange(0,np.shape(slices)[1],1):
        reduce_dim[counter1,:,:]  = np.flip(skimage.transform.rotate(reduce_dim[counter1,:,:],angle),1)
        holder[counter1,:,:] = reduce_dim[counter1,left:right,top:bottom]
    return holder


def reduce_rotate_auto(slices,angle,pos,offset):
    '''
    determine the corners of the reconstruction using the scan postions and uses this then to determine
    spatial slicing of the array such that majority of the field of view is reconstruction rather
    than padding
    '''

    index = np.int32(np.sqrt(np.shape(pos)[2]))
    y0,x0 = np.int32(pos[0,:,0])
    y1,x1 = np.int32(pos[0,:,index-1])
    y2,x2 = np.int32(pos[0,:,-index])
    
    holder = np.zeros([np.shape(slices)[1], (x2-x0) + (2*offset),(y0-y1)+(2*offset)])
    reduce_dim = slices[0,:,0,0,0,x0-offset:x2+offset,y1-offset:y0+offset]
    for counter1 in np.arange(0,np.shape(slices)[1],1):
        reduce_dim[counter1,:,:]  = np.flip(skimage.transform.rotate(reduce_dim[counter1,:,:],angle),1)
        holder[counter1,:,:] = reduce_dim[counter1,:,:]
    return holder



def reduce_probes(probes_states):
    '''
    the code attempts to recude the probe array into it minimal aspects
    i.e. just modes, slices and spatial array
    the probe diemonsions are as follows:
    0 - modes - sx 
    1 - slices
    2 - energy
    3 -  empty
    4 - position correction
    5/6 - spatial array (actual probe)
    '''
    return probes_states[:,:,0,0,0,:,:]

'''----------Filtering functions----------'''

def Gen_mean_kern(kern_wid):
    '''
    generate a basic kernel for psudeo flat field correction
    #ToDo this and other similar function likely exists in numpy so these should be used instead
    this currently used by the psudeo_flat_fild_correction
    '''
    kern = np.ones([kern_wid,kern_wid])
    #print(np.shape(kern))
    return kern/np.sum(kern)


def Pseudo_Flat_field_correction(array,kern_wid):
    '''
    calls a kernal of certain size to convolve with to produce a blurred image which is then subtracted from the inputted signal
    typically used to remove low spatial frequencies
    '''
    kern = Gen_mean_kern(kern_wid)
    Psflat_field = scipy.signal.convolve2d(array,kern,mode='same',boundary='fill')
    corrected_array = array - Psflat_field
    return corrected_array

def reduce_rotate_pseudoflat(slices,left,right,top,bottom,angle,kern_wid):
    '''
    function to perfrom reduce_rotate with flat field correction
    '''
    holder = np.zeros([np.shape(slices)[1],right-left,bottom-top])
    reduce_dim = slices[0,:,0,0,0,:,:]
    for counter1 in np.arange(0,np.shape(slices)[1],1):
        reduce_dim[counter1,:,:]  = np.flip(skimage.transform.rotate(reduce_dim[counter1,:,:],angle),1)
        holder[counter1,:,:] = Pseudo_Flat_field_correction(reduce_dim[counter1,left:right,top:bottom],kern_wid)
    return holder

def pseudoflat_stack(slices,kern_wid):
    holder = np.zeros(np.shape(slices))
    for counter1 in np.arange(0,np.shape(slices)[0],1):
        holder[counter1,:,:] = Pseudo_Flat_field_correction(slices[counter1,:,:],kern_wid)
    return holder



def missing_wedge_filter(array,slice_thickness,px,beta=0.5,offset=1e-3):
    '''
    kz -  spatial frequices corresponding the real space z-direction 
    kx -  spatial frequices corresponding the real space x-direction
    ky -  spatial frequices corresponding the real space y-direction
    fft shift required in order to ensure frequices are mapped correctly
    '''
    freq_z = np.fft.fftshift(np.fft.fftfreq(np.shape(array)[0],d=slice_thickness))
    freq_x = np.fft.fftshift(np.fft.fftfreq(np.shape(array)[1],d=px))
    freq_y = np.fft.fftshift(np.fft.fftfreq(np.shape(array)[2],d=px))
    kx,ky,kz = np.meshgrid(freq_x,freq_y,freq_z)
    numu = (beta**2)*(np.abs(kz)**2)
    domn = (kx**2)+(ky**2)+offset
    W_filter = 1 - (2/np.pi)*np.arctan(numu/domn)
    return W_filter


def postivity_constraint(slices,cplex=False):
    if complex:
        slices_mag = np.abs(slices)
        slices = np.angle(slices)

    new_phase = np.where(slices>0,slices,0)

    if complex:
        return slices_mag*np.exp(1j*new_phase)
    else:
        return new_phase






#ToDo create generate application of filter to stack 3D images/slices


'''----------auto create figure functions----------'''

def recon2uint8(array):
    '''
    A function to take an array currently only phase or magntuide not both
    #ToDO make a function like this to deal with complex arrays
    '''
    array = array - np.amin(array)
    array = array/np.amax(array)
    array = array*255
    array = array.astype('uint8')
    return array

def obtain_json_data(recon_path,extra_name):
    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        rot = f['process']['common']['scan']['rotation']
        save_name = f['process']['save_prefix'] + extra_name


def Recon2imarrays(recon_path,left,right,top,bottom,sav_loc,extra_name=''):
    '''
    takes a recon path and produces a set pngs (the number pngs is determined
    by whether the reconstruction is a single or multi-slice reconstruction)

    #ToDo: ame a test whether the object is a single or multi-slice reconstruction
    '''

    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        rot = f['process']['common']['scan']['rotation']
        save_name = f['process']['save_prefix'] + extra_name

    #load the actual object phase
    with h5py.File(recon_path,'r') as g:
        objPhase = g['entry_1']['process_1']['output_1']['object_phase'][()]

    #rotate and reduce the objects diemsons
    rot_obj = reduce_rotate(objPhase,left,right,top,bottom,-rot)
    objPhaseSummed = np.sum(rot_obj,0)

    objPhaseSummed = recon2uint8(objPhaseSummed)

    #save the summed array as png
    save_string = sav_loc + '/' + save_name + 'Summed_phase.png'
    cv2.imwrite(save_string,objPhaseSummed)

    #save the numpy arrays as well
    save_string = sav_loc + '/' + save_name + 'Phase_Numpyarray.npy'
    np.save(save_string, rot_obj)
    
    #save the slices serpately as pngs
    for counter1 in np.arange(0,np.shape(rot_obj)[0],1):
        save_string = sav_loc + '/' + save_name + 'slice' + str(counter1+1) + 'phase.png'
        slice_phase = rot_obj[counter1,:,:]
        slice_phase = recon2uint8(slice_phase)
        cv2.imwrite(save_string,slice_phase)


def ObjFigureMS(data_dir,recon_path,start,end,save_loc,append=None):
    #load parameters from the corresponding json file to determine the rotation and real space pixel size
    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        rot = f['process']['common']['scan']['rotation']
        CL   = f['experiment']['detector']['position'][-1]
        N    = f['process']['common']['detector']['crop'][-1]
        detx = f['process']['common']['detector']['pix_pitch'][-1]
        energy  = f['process']['common']['source']['energy'][-1]
        #print(f['process'].keys())
        save_name = f['process']['save_prefix']
        print(save_name)

    #determine the real space and Fourier space pixel Size
    wav = (1.23e3/(np.sqrt(energy*(1+9.78e-7*energy))))*1e-12
    theta = N*detx/CL
    sp_dx = wav/theta

    #load the actual object phase
    with h5py.File(data_dir,'r') as g:
        objPhase = g['entry_1']['process_1']['output_1']['object_phase'][()]

    rot_obj = reduce_rotate(objPhase,start,end,start,end,-rot)

    plt.figure()
    #plot a figure for each of the slices 
    for counter1 in np.arange(0,np.shape(rot_obj)[0],1):
        slice_phase = rot_obj[counter1,:,:]
        col0 = plt.imshow(slice_phase,cmap='inferno')
        plt.title('Object phase slice '+ str(counter1+1))
        plt.colorbar(col0,label='Phase (rad)')
        ax = plt.gca()
        bar1 = AnchoredSizeBar(ax.transData, np.round(1e-9/sp_dx), '1nm', 4)
        ax.add_artist(bar1)
        ax.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
        
        #save summed phase figure
        if append == None:
            summed_phase_save_string = save_loc + '/' + save_name + 'slice' + str(counter1+1) + 'phase.png'
        else:
            print('\nappending')
            summed_phase_save_string = save_loc + '/' + save_name + 'slice' + str(counter1+1) + 'phase' + append + ' .png'
        saving_text = 'saving the summed phase figure to the following location: ' + summed_phase_save_string
        print(saving_text)
        plt.savefig(summed_phase_save_string,dpi=400)
        plt.clf()

def Recon2tiffs(recon_path,left,right,top,bottom,sav_loc,append=''):
    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        rot = f['process']['common']['scan']['rotation']
        save_name = f['process']['save_prefix']

    #load the actual object phase
    with h5py.File(recon_path,'r') as g:
        objPhase = g['entry_1']['process_1']['output_1']['object_phase'][()]

    #rotate and reduce the objects diemsons
    rot_obj = reduce_rotate(objPhase,left,right,top,bottom,-rot)
    objPhaseSummed = np.sum(rot_obj,0)

    objPhaseSummed = recon2uint8(objPhaseSummed)

    #save the summed array as png
    save_string = sav_loc + '/' + save_name + 'Summed_phase_' + append + '.png'
    cv2.imwrite(save_string,objPhaseSummed)

    #save the numpy arrays as well
    save_string = sav_loc + '/' + save_name + 'Phase_Numpyarray_' + append + '.npy'
    np.save(save_string, rot_obj)
    
    #save the slices serpately as pngs
    for counter1 in np.arange(0,np.shape(rot_obj)[0],1):
        save_string = sav_loc + '/' + save_name + 'slice' + str(counter1+1) + 'phase_' + append + '.tiff'
        slice_phase = rot_obj[counter1,:,:]
        slice_phase = recon2uint8(slice_phase)
        cv2.imwrite(save_string,slice_phase)
        tifffile.imwrite(save_string, slice_phase)


def array2tiff(array, sav_loc, save_name, append='', file_type = '.tiff'):

    objPhaseSummed = recon2uint8(np.sum(array,axis=0))

    #save the summed array as png
    save_string = sav_loc + '/' + save_name + 'Summed_' + append + file_type
    tifffile.imwrite(save_string,objPhaseSummed)
    
    #save the slices serpately as pngs
    for counter1 in np.arange(0,np.shape(array)[0],1):
        save_string = sav_loc + '/' + save_name + 'slice' + str(counter1+1) + 'phase_' + append + file_type
        slice_phase = array[counter1,:,:]
        slice_phase = recon2uint8(slice_phase)
        cv2.imwrite(save_string,slice_phase)
        tifffile.imwrite(save_string, slice_phase)

def ProbeFigure(data_dir,recon_path,save_loc):
    #find the associated json path with the reconstruction to extract information such that we plot scalebars
    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        CL   = f['experiment']['detector']['position'][-1]
        N    = f['process']['common']['detector']['crop'][-1]
        detx = f['process']['common']['detector']['pix_pitch'][-1]
        energy  = f['process']['common']['source']['energy'][-1]
        save_name = f['process']['save_prefix']

    #determine the real space and Fourier space pixel Size
    wav = (1.23e3/(np.sqrt(energy*(1+9.78e-7*energy))))*1e-12
    theta = N*detx/CL
    sp_dx = wav/theta
    theta_dx = detx/CL
    fr_dx = theta_dx/wav
    
    #load the actual probe reconstruction
    with h5py.File(data_dir,'r') as g:
        probe = g['entry_1']['process_1']['output_1']['probe'][()]
    #sum over the probe modes and select the first slice
    probe_sumModes = np.sum(probe[:,0,0,0,0,:,:],0)
    #Fourier transfrom the summed probe
    probe_sumModes_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(probe_sumModes)))

    #plot the probe figure - spatial and fourier plots
    fig, _ax = plt.subplots(nrows=2, ncols=2)
    ax = _ax.flatten()

    ax[0].set_title('Real space probe magnitude')
    col0 = ax[0].imshow(np.abs(probe_sumModes),cmap='inferno')
    ax[1].set_title('Real space probe phase')
    col1 = ax[1].imshow(np.angle(probe_sumModes),cmap='inferno')
    ax[2].set_title('Fourier space probe magnitude')
    col2 = ax[2].imshow(np.log10(np.abs(probe_sumModes_fft)),cmap='inferno')
    ax[3].set_title('Fourier space probe phase')
    col3 = ax[3].imshow(np.angle(probe_sumModes_fft),cmap='inferno')
    #remove axis labels and ticks
    for counter1 in np.arange(0,4,1):
        ax[counter1].tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)

    #add scalebars to each figure
    bar_text = '1nm'
    bar1 = AnchoredSizeBar(ax[0].transData, np.round(2e-9/sp_dx), '2nm', 4)
    bar2 = AnchoredSizeBar(ax[1].transData, np.round(2e-9/sp_dx), '2nm', 4)
    bar3 = AnchoredSizeBar(ax[2].transData, np.round(60e-3/theta_dx), '60mrad', 4)
    bar4 = AnchoredSizeBar(ax[3].transData, np.round(60e-3/theta_dx), '60mrad', 4)
    ax[0].add_artist(bar1)
    ax[1].add_artist(bar2)
    ax[2].add_artist(bar3)
    ax[3].add_artist(bar4)

    #add colorbars
    fig.colorbar(col0, ax=ax[0],label='Magnitude (a.u.)')
    fig.colorbar(col1, ax=ax[1],label='Phase (rad)')
    fig.colorbar(col2, ax=ax[2],label='Magnitude (a.u.)')
    fig.colorbar(col3, ax=ax[3],label='Phase (rad)')
    
    #now save the figure to the desired location
    probe_save_string = save_loc + '/' + save_name + 'probe_figure.png'
    saving_text = 'saving the summed phase figure to the following location: ' + probe_save_string
    print(saving_text)
    plt.savefig(probe_save_string)

def disp_probes(probe_array,disp='fft_abs',verbose=False):
    try:
        if len(np.shape(probe_array))>4:
            probe_red = reduce_probes(probe_array)
            if verbose:
                print('Reducing probes array dimensions')
                print(np.shape(probe_red))
        else:
            probe_red = probe_array
        fig,xxs_ = plt.subplots(2,np.ceil(np.shape(probe_red)[0]/2).astype('uint32'))
        xxs = xxs_.flatten()
        for counter1 in np.arange(0,np.shape(probe_red)[0],1):            
            if len(np.shape(probe_array))<4:
                array = probe_red[counter1,:,:]
                array_fft = quick_fft(np.copy(probe_red[counter1,:,:]))
            else:
                array = probe_red[counter1,0,:,:]
                array_fft = quick_fft(np.copy(probe_red[counter1,0,:,:]))
            if disp == 'fft_abs':
                xxs[counter1].imshow(np.abs(array_fft),cmap='inferno',norm='symlog')
            elif disp == 'fft_angle':
                xxs[counter1].imshow(np.angle(array_fft),cmap='inferno')
            elif disp == 'abs':
                xxs[counter1].imshow(np.abs(array),cmap='inferno')
            elif disp == 'angle':
                xxs[counter1].imshow(np.angle(array),cmap='inferno')
    except:
        print(traceback.format_exc())


    '''----------create videos----------'''

def ptycho_MS_vid(recon_path,left,right,top,bottom,sav_loc,postfix='',fps=5,flat_field=False):
    json_dir = recon_path.replace('.hdf','.json')
    with open(json_dir) as json_data:
        f = json.load(json_data)
        rot = f['process']['common']['scan']['rotation']
        save_name = f['process']['save_prefix']
        slice_thickness = np.round(f['process']['PIE']['MultiSlice']['S_distance']*1e9,2)

    #load the actual object phase
    with h5py.File(recon_path,'r') as g:
        objPhase = g['entry_1']['process_1']['output_1']['object_phase'][()]

    #rotate and reduce the objects diemsons
    rot_obj = reduce_rotate(objPhase,left,right,top,bottom,-rot)


    #save the numpy arrays as well
    save_string = sav_loc + '/' + save_name + 'Phase_Numpyarray.npy'
    np.save(save_string, rot_obj)

    frame_size = np.shape(rot_obj[0,:,:])
    print(frame_size)

    output_str = sav_loc + '/' + save_name +postfix +'vid.mp4'
    print(output_str)
    out = cv2.VideoWriter(output_str, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_size[1], frame_size[0]), False)
    duration = np.shape(rot_obj)[0]
    
    #save the slices serpately as pngs
    for counter1 in range(fps * duration):
        index = int(counter1//(fps*1))
        #print(index)
        save_string = sav_loc + '/' + save_name + 'slice' + str(counter1+1) + 'phase.png'
        slice_phase = rot_obj[index,:,:]
        if flat_field == True:
            slice_phase = Pseudo_Flat_field_correction(slice_phase,5)
        
        slice_phase = recon2uint8(slice_phase.copy())
        #print(np.shape(slice_phase))
        print_string = 'Thickness = ' + str(np.round(slice_thickness*(index),2)) + '-' + str(np.round(slice_thickness*(index+1),2)) + 'nm'
        cv2.putText(slice_phase,print_string,(40,40), fontFace = cv2.FONT_HERSHEY_COMPLEX ,fontScale = 0.8, color = (250,225,100))
        out.write(slice_phase)
    out.release()

def array_vid(array, save_name, sav_loc, slice_thickness, postfix='', fps=5, unit='nm', add_text = True):

    frame_size = np.shape(array[0,:,:])
    output_str = sav_loc + '/' + save_name + postfix +'vid.mp4'
    print(output_str)
    out = cv2.VideoWriter(output_str, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_size[1], frame_size[0]), False)
    duration = np.shape(array)[0]
    
    for counter1 in range(fps * duration):
        index = int(counter1//(fps*1))
        slice_phase = array[index,:,:]
        slice_phase = recon2uint8(slice_phase.copy())
        print_string = 'Thickness = ' + str(np.round(slice_thickness*(index),2)) + '-' + str(np.round(slice_thickness*(index+1),2)) + unit
        if add_text:
            cv2.putText(slice_phase,print_string,(40,40), fontFace = cv2.FONT_HERSHEY_COMPLEX ,fontScale = 0.8, color = (250,225,100))
        out.write(slice_phase)
    out.release()


def array_gif(array, save_name, sav_loc, postfix='',verbose=False):
    array = recon2uint8(array)
    imgs = [Image.fromarray(array) for array in array]
    output_str = sav_loc + save_name + postfix + '.gif'
    if verbose:
        print('The gif will be saved as: %s' % output_str)
    imgs[0].save(output_str, save_all=True, append_images=imgs[1:], duration=50, loop=0)




'''----------whole experiment processing videos and excel sheets----------'''

def find_ptycho_recons(basedir,year,session,subfolder,pty_name=''):
    '''this function is to find ptyrex hdf reconstruction files'''

    '''determine path path being searched and change directory to that path for glob'''
    dest_path = f'{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
    #print('\nlength of string: ' + str(len(dest_path)) + '\n')
    string_length = len(dest_path)+1
    
    os.chdir(dest_path)
    print('File being searched: ' + dest_path)

    hdf_list = []
    hdf_counter = -1
    current_time_stamp = ''


    '''
    use glob and os to find hdf files. This relies on the files being the required 
    depth within the file system, see the number */* within the glob search.
    '''
    for num,file in enumerate(sorted(list(glob.glob(f'*/*/*/*{pty_name}*.hdf')))):

        '''
        only take the most recent hdf, here we used the fact that all hdf's end in a 
        time stamp to determine the latest one
        '''
        #print(str(num) + ': ' + os.path.join(dest_path,file))
        time_stamp = os.path.join(dest_path,file)[string_length:string_length+15]
        if time_stamp == current_time_stamp:
            hdf_list[hdf_counter] = os.path.join(dest_path,file)
        else:
            current_time_stamp = time_stamp
            #print('new time stamp' + ': ' + current_time_stamp)
            hdf_counter = hdf_counter+1
            hdf_list.append(os.path.join(dest_path,file))

    return hdf_list

def gen_index_hdf_list_via_meta(hdf_list,print_key,is_recon=False):
    '''
    when given a list of reconstructions/metadata this function should generate an index
    based off the meta file and a print_key which determines the order of the index
    '''
    print_val = []
    for num,file in enumerate(hdf_list):
        if is_recon:
            '''using the structure of the file system here to obtain the meta data path'''
            meta_path = '/'.join(file.split('/')[:10]) + '/' + file.split('/')[9] + '.hdf'
        else:
            meta_path = file

        with h5py.File(meta_path,'r') as h:
            print_val.append(h['metadata'][print_key][()])

    index = np.argsort(np.array(np.round(print_val))).tolist()
    return index

def ptycho_series_to_numpy(hdf_list,save_location,print_key=None,is_recon=False,verbose=False):
    '''
    This function is just for easy creation of numpy files for a whole reconstruction series
    in a seperate file location defined by the user
    '''
    if print_key !=None:
        index = gen_index_hdf_list_via_meta(hdf_list,print_key,is_recon=False)

    os.chdir(save_location)
    for num,file in enumerate(hdf_list):
        '''
        first attempt to create the folders in which the numpy files will be saved.
        ToDo maybe it better is save one large stack?
        '''
        if verbose:
            timestamp = file.split('/')[9]
            print(f'attempting to create a folder called: {timestamp}...')
        if os.path.isdir(file.split('/')[9]) == False:
            os.mkdir(file.split('/')[9])
        else:
            if verbose:
                print('skipping creating this folder as it already exists')
        
        '''
        now to save the numpy stack to this new file location
        '''
        recon = ptyrex2numpy(file,['object_phase'],verbose=False)
        save_string = save_location + '/' + file.split('/')[9] + '/' + 'Phase_Numpyarray.npy'
        np.save(save_string, recon['object_phase'])

def ptycho_recon_series(hdf_list,sav_loc,save_name,data_key,meta_dir,print_key=None,sort=False,datasize=(2000,2000)):
    error_list = []
    print_val = []

    '''
    print where the video is going to be saved
    '''
    output_str = sav_loc  +  save_name +'vid.avi'
    numpy_out_str = sav_loc  +  save_name +'arrays.npy'
    print(f'the save locations is: {output_str}')

    '''
    open up the json associated with a particular reconstruction and from figure out the rotation angle
    and find the meta data path
    '''
    #ToDo make it so that this is a try statement in case the file does not exist
    for num,file in enumerate(hdf_list):
        json_dir = file.replace('.hdf','.json')
        with open(json_dir) as json_data:
            f = json.load(json_data)
            rot = f['process']['common']['scan']['rotation']
            meta_data_path = f['experiment']['data']['data_path']
            

        #load the actual object phase
        '''load the actual images'''
        try:
            with h5py.File(file,'r') as g:
                objPhase = g['entry_1']['process_1']['output_1']['object_phase'][()]
                pos = g['entry_1']['process_1']['output_1']['scan_positions'][()]
                rot_obj = reduce_rotate_auto(objPhase,-rot,pos,200)
                #rot_obj = reduce_rotate(objPhase,0,np.shape(objPhase)[-2],0,np.shape(objPhase)[-1],-rot)
                objPhaseSummed = np.sum(np.copy(rot_obj),0)
                if objPhaseSummed.shape[0] != datasize[0] or objPhaseSummed.shape[1] != datasize[1]:
                    print('data is not the same size padding')
                    data = objPhaseSummed
                    print(np.shape(objPhaseSummed))
                    data = np.pad(data,((0,datasize[0]-objPhaseSummed.shape[0]),(0,datasize[1]-objPhaseSummed.shape[1])))
                    print(np.shape(data))
                    print(objPhaseSummed.shape[0])
                    print(objPhaseSummed.shape[1])
                    print(objPhaseSummed.shape[0]+datasize[0]-objPhaseSummed.shape[0])
                    print(objPhaseSummed.shape[1]+datasize[1]-objPhaseSummed.shape[1])
                else:
                    print('data is the same size')
                    data = g[data_key][()]
        except:
            '''likely a error due to data being request not being in file due to a processing error therefore add to the error list'''
            error_list.append(file)
            print(traceback.format_exc())

        #rotate and reduce the objects diemsons

        if num == 0:
            session_recons = data[:,:,np.newaxis]
        else:
            session_recons = np.append(session_recons,data[:,:,np.newaxis],axis=2)

        '''
        search for the requested meta data such that is displayed on each frame of the video, if meta does not have 
        the requested key leave balnk and add meta path to the error list.
        '''
        if print_key != None:
            try:
                with h5py.File(meta_data_path,'r') as h:
                    print_val.append(h['metadata'][print_key][()])
            except:
                error_list.append(meta_dir)
                print_val.append('')
        else:
            print_val.append(file.split('/')[-4])
    print_key = 'Timestamp'
    
    '''now create a video/numpy array out of the stacked images'''
    print('\nSaving ptycho session video as: ' + output_str + '...\n')
    print('\nSaving ptycho session numpy file as: ' + numpy_out_str + '...\n')

    if sort == True: 
        np.save(numpy_out_str,session_recons[:,:,np.argsort(np.array(np.round(print_val)))],allow_pickle=False)
        print(np.argsort(np.array(np.round(print_val))))
    else:
        np.save(numpy_out_str,session_recons,allow_pickle=False)

    frame_size = np.shape(session_recons)
    fps = 1
    out = cv2.VideoWriter(output_str, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_size[1], frame_size[0]), False)
    duration = frame_size[2]
    
    
    #save the slices serpately as pngs
    for counter1 in range(fps * duration):
        if sort == False:
            index = int(counter1//(fps*1))
            print_string = print_key + ': ' + str(print_val[counter1])
        else:
            index = int(np.argsort(np.array(np.round(print_val)))[counter1]//(fps*1))
            print_string = print_key + ': ' + str(print_val[np.argsort(np.array(np.round(print_val)))[counter1]])
        frame = session_recons[:,:,index]
        frame = recon2uint8(frame.copy())
        cv2.putText(frame,print_string,(80,80), fontFace = cv2.FONT_HERSHEY_COMPLEX ,fontScale = 3, color = (250,225,100))
        out.write(frame)
    out.release()


def h5clear_flags_series(hdf_list):
    for num,file in enumerate(hdf_list):
        cmd_string = 'h5clear -s ' + file
        #print(cmd_string)
        p = subprocess.Popen(cmd_string.split(),
                     stdout=subprocess.PIPE)
        

def get_key_and_value(dest_path,index):
    '''this function searchs the dest_path for metadata files ending in a .hdf file extension, it then obtains the time stamp
    of the metadata. the meta keys and meta vals are returned at the end of the function.'''
    meta_keys = ['filename']
    '''the line below gets the time stamp within current ePSIC file system'''
    meta_vals = [os.path.join(dest_path,sorted(list(glob.glob('*/*.hdf')))[index]).split('/')[-1][:-4]]
    
    '''now access the rest of the data within hdf file'''
    with h5py.File(os.path.join(dest_path,list(glob.glob('*/*.hdf'))[index]), 'r') as microscope_meta:
        '''loop through the keys - this could be faster without looping'''
        for meta_key, meta_val in microscope_meta['metadata'].items():
            if meta_key != 'deflector_values':
                if meta_key != 'lens_values':
                    meta_keys.append(meta_key)
                    '''debug statements'''
                    #print(meta_key)
                    #print(meta_key + ': ' + str(meta_val[()]))
                    meta_vals.append(meta_val[()])
    return meta_keys, meta_vals


def meta_show_session(year,session,subfolder,sort='filename'):
    '''The user provides a path to particular E02 session. This code then iterates through all of the hdf files 
    within a folder to obtain the meta data for the whole experiment. there is option to include a parameter to
    sort by '''
    
    dest_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/{subfolder}'
    os.chdir(dest_path)
    print('File being searched: ' + dest_path)
    
    '''initalise the pandas frame with meta data from the first hdf file'''
    meta_keys, meta_vals = get_key_and_value(dest_path,0)

    t = pd.DataFrame(data=[meta_vals], columns=meta_keys)
    
    for num,file in enumerate(sorted(list(glob.glob('*/*.hdf')[1:]))):
        '''debug print file number and order accessed'''
        #print(str(num) + ': ' + file)
        meta_keys, meta_vals = get_key_and_value(dest_path,num)
        t2 = pd.DataFrame(data=[meta_vals], columns=meta_keys)
        t = pd.concat([t,t2])
    return t



'''----------ipython widget wrappers----------'''

class Recon_output_helper():
    
    def __init__(self,path=None):
        self.get_reconstruction_path_widget(path)



    def prefill_boxes(self,path=None):
        '''
        This function is used to prefill the basedir, year and session boxes in 
        the ipython widgets such that users do not have repeatly enter in the 
        same information if they are running different cells within the same notebook
        this only works if the folder which the note book is in is associated with a 
        particular user session and this can work with staged data as well.
        '''
        #ToDo make these varibles be stored in self such that once it has been filled out one

        st = {"description_width": "initial"}
        current_dir = os.getcwd()
        if path == None:
            if current_dir[:13] == '/dls/e02/data' or '/dls/e01/data':
                basedir = Text(value='/'.join(current_dir.split('/')[1:4]), description='Base data directory path:', style=st)
                year = Text(value=current_dir.split('/')[4],description='Year:', style=st)
                session = Text(value=current_dir.split('/')[5],description='Session:', style=st)
            elif current_dir == '/dls/staging/dls/e02/data':
                basedir = Text(value='/'.join(current_dir.split('/')[1:6]), description='Base data directory path:', style=st)
                year = Text(value=current_dir.split('/')[6],description='Year:', style=st)
                session = Text(value=current_dir.split('/')[7],description='Session:', style=st)
            else:
                basedir = Text(description='Base data directory path:', style=st)
                year = Text(description='Year:', style=st)
                session = Text(description='Session:', style=st)
        else:
            if path.split('/')[2] == 'staging':
                basedir = Text(value='/'.join(path.split('/')[1:6]), description='Base data directory path:', style=st)
                year = Text(value='/'.join(path.split('/')[6:7]),description='Year:', style=st)
                session = Text(value='/'.join(path.split('/')[7:8]),description='Session:', style=st)
            else:
                basedir = Text(value='/'.join(path.split('/')[1:4]), description='Base data directory path:', style=st)
                year = Text(value='/'.join(path.split('/')[4:5]),description='Year:', style=st)
                session = Text(value='/'.join(path.split('/')[5:6]),description='Session:', style=st)
        return basedir, year, session

    def get_reconstruction_path(self, basedir, year, session, subfolder, ptycho_config_name, Choose_timestamp, Choose_recon_timestamp, verbose=False):
        self.tmp_list = []
        self.timestamp_list = []
        self.recon_timestamp_list = []
        self.recon_list = []

        '''use glob and os to find the meta data files'''
        if verbose:
            print(f'Path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')
        if basedir == '' or year == '' or session == '' or subfolder == '' or ptycho_config_name == '':
            print('\nwaiting for the folowing inputs: basedir, year, session, subfolder and config_name\n')
        else:
            '''this part of the code finds all of the timestamps to populate timestamp_list'''
            if basedir.split('/')[1] == 'e01':
                src_path = f'/{basedir}/{year}/{session}/raw/{subfolder}'
                search_depth = '*/*/*'
                if verbose:
                    print('e01')
            elif basedir.split('/')[1] == 'e02':
                src_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
                search_depth = '*/*/*/*'
                if verbose:
                    print('e02')
            os.chdir(src_path)
            for num, file in enumerate(sorted(list(glob.glob(search_depth + ptycho_config_name + '.json')))):
                if verbose:
                    print(str(num) + ': ' + os.path.join(src_path, file))
                self.tmp_list.append(os.path.join(src_path, file))
                #time_stamp_index = os.path.join(src_path, file).find('/pty_out')
                #self.timestamp_list.append(os.path.join(src_path, file)[time_stamp_index - 15:time_stamp_index])
                self.timestamp_list.append(self.tmp_list[num].split('/')[-len(search_depth.split('/'))])
            if verbose:
                print(self.timestamp_list)
            self.grpw.widget.children[5].options = self.timestamp_list

            '''find the hdf files to display in the window'''
            if Choose_timestamp != 'empty':
                indexer = self.timestamp_list.index(Choose_timestamp)
                current_file = self.tmp_list[indexer]
                current_folder = current_file.replace(f'{ptycho_config_name}.json','')
                os.chdir(current_folder)
                if verbose:
                    print(f'indexer: {indexer}')
                    print(f'current_file: {current_file}')
                    print(f'current_folder: {current_folder}')
                for num, hdfile in enumerate(sorted(list(glob.glob( '*.hdf')))):
                    if verbose:
                        print(f'{num}: {os.path.join(current_folder,hdfile)}')
                    self.recon_list.append(os.path.join(current_folder,hdfile))
                    self.recon_timestamp_list.append(os.path.join(current_folder,hdfile)[-19:-4])
                if verbose:
                    print(self.recon_timestamp_list)
                    print(self.recon_list)
                self.grpw.widget.children[6].options = self.recon_timestamp_list
                if Choose_recon_timestamp != 'empty':
                    try:
                        recon_indexer = self.recon_timestamp_list.index(Choose_recon_timestamp)
                        self.recon_path = self.recon_list[recon_indexer]
                        print(f'path of the reconstruction file to be shown: {self.recon_path}')
                        if verbose:
                            print(f'index of the reconstruction file to be shown: {recon_indexer}')
                    except:
                        if verbose:
                            print('error switch between files has occured')
                    #display the reconstruction

        


    def get_reconstruction_path_widget(self,path=None):
        st = {"description_width": "initial"}
        basedir,year,session = self.prefill_boxes(path)
        subfolder = Text(description='subfolder:', style=st)
        ptycho_config_name = Text(description='Name of the json file to process:', style=st)
        Choose_timestamp = Dropdown(options=['empty'], value='empty', description='choose a timestamp')
        Choose_recon_timestamp = Dropdown(options=['empty'], value='empty', description='choose a recon')
        verbose = Checkbox(value=False, description='Check this for debugging and error printing', style=st)


        self.grpw = ipywidgets.interact(self.get_reconstruction_path,
                            basedir = basedir,
                            year = year,
                            session = session,
                            subfolder = subfolder,
                            ptycho_config_name = ptycho_config_name,
                            Choose_timestamp = Choose_timestamp,
                            Choose_recon_timestamp = Choose_recon_timestamp,
                            verbose = verbose)

