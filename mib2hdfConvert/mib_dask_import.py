#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script imports Medipix mib files into numpy.memmap, usese dask.array for 
reshaping and reformatting and passes as a lazy signal to hyperspy.
It works on Raw and not-RAW files and can accommodate single chip and
quad chip data.
If the data is from quad chip it adds crosses of zero pixel values.
The code looks into the frame exposure times in the MIB header characters and 
finds whether all the frame times are the same or there is a flyback pixel.
This would then create a STEM_flag, resulting to further reshaping of the data. 
It only needs the mib file for import, i.e. ignores the .HDR file.
Code can accommodate unfinished 4D-STEM frames as it uses the file size to figure 
out the number of frames.

                 
"""
import os
import numpy as np
import h5py
import dask.array as da
import dask
import hyperspy.api as hs


def _manageHeader(fname):
    """Get necessary information from the header of the .mib file.

    Parameters
    ----------
    fname : str
        Filename for header file.

    Returns
    -------
    hdr : tuple
        (DataOffset,NChips,PixelDepthInFile,sensorLayout,Timestamp,shuttertime,bitdepth)

    Examples
    --------
    #Output for 6bit 256*256 data:
    #(768, 4, 'R64', '2x2', '2019-06-14 11:46:12.607836', 0.0002, 6)
    #Output for 12bit single frame nor RAW:
    #(768, 4, 'U16', '2x2', '2019-06-06 11:12:42.001309', 0.001, 12)

    """   
    Header = str()
    with open(fname,'rb') as input:
        aByte= input.read(1)
        Header += str(aByte.decode('ascii'))
        # This gets rid of the header 
        while aByte and ord(aByte) != 0: 
       
            aByte= input.read(1)
            Header += str(aByte.decode('ascii'))

    elements_in_header = Header.split(',')
    
    
    DataOffset = int(elements_in_header[2])
    
    NChips = int(elements_in_header[3])

    PixelDepthInFile= elements_in_header[6]
    sensorLayout = elements_in_header[7].strip()    
    Timestamp = elements_in_header[9]
    shuttertime = float(elements_in_header[10])
    
    if PixelDepthInFile == 'R64':
        bitdepth =int(elements_in_header[18]) # RAW
    elif PixelDepthInFile =='U16':
        bitdepth =12
    elif PixelDepthInFile =='U08':
        bitdepth =6
    elif PixelDepthInFile =='U32':
        bitdepth =24
        
    hdr = (DataOffset,NChips,PixelDepthInFile,sensorLayout,Timestamp,shuttertime,bitdepth)
 
    return hdr


def parse_hdr(fp):
    """Parse information from mib file header info from _manageHeader function.
    
    Parameters
    ----------
    fp : str
        Filepath to .mib file.
    
    Returns
    -------
    hdr_info : dict
        Dictionary containing header info extracted from .mib file.
            
    """
    hdr_info = {}
    
    read_hdr = _manageHeader(fp)

    # Set the array size of the chip 

    if read_hdr[3] == '1x1':
        hdr_info['width'] = 256
        hdr_info['height'] = 256
    elif read_hdr[3] == '2x2':
        hdr_info['width'] = 512
        hdr_info['height'] = 512
    
    hdr_info['Assembly Size'] = read_hdr[3]

    # Set mib offset
    hdr_info['offset'] = read_hdr[0]
    # Set data-type
    hdr_info['data-type'] = 'unsigned'
    # Set data-length
    if read_hdr[6] == '1':
        # Binary data recorded as 8 bit numbers
        hdr_info['data-length'] = '8'
    else:
        # Changes 6 to 8 , 12 to 16 and 24 to 32 bit
        cd_int = int(read_hdr[6])
        hdr_info['data-length'] = str(int((cd_int + cd_int/3) ))
        
    hdr_info['Counter Depth (number)'] = int(read_hdr[6])
    if read_hdr[2] =='R64':
        hdr_info['raw'] = 'R64'
    else:
        hdr_info['raw'] = 'MIB'
    # Set byte order
    hdr_info['byte-order'] = 'dont-care'
    # Set record by to stack of images
    hdr_info['record-by'] = 'image'
    
    
    # Set title to file name
    hdr_info['title'] = fp.split('.')[0]
    # Set time and date
    # Adding the try argument to accommodate the new hdr formatting as of April 2018
    try:
        year, month, day_time = read_hdr[4].split('-')
        day , time = day_time.split(' ')
        hdr_info['date'] = year + month + day
        hdr_info['time'] = time
    except:
        day, month, year_time = read_hdr[4].split('/')
        year , time = year_time.split(' ')
        hdr_info['date'] = year + month + day
        hdr_info['time'] = time
        
    hdr_info['data offset'] = read_hdr[0]

    return hdr_info


    
def add_crosses(a):
    """ Adds 3 pixel buffer cross to quad chip data.
    
    Parameters
    ----------
    a : dask.array
        Stack of raw frames, prior to dimension reshaping, to insert 
        3 pixel buffer cross into.
    
    Returns
    -------
    b : dask.array
        Stack of frames including 3 pixel buffer cross.
    """
    # Determine dimensions of raw frame data
    a_type = a.dtype
    a_shape = a.shape

    len_a_shape = len(a_shape)
    img_axes = len_a_shape-2, len_a_shape-1
    a_half = int(a_shape[img_axes[0]] / 2), int(a_shape[img_axes[1]] / 2)
    # Define 3 pixel wide cross of zeros to pad raw data
    z_array = da.zeros((a_shape[0],a_shape[1],3), dtype = a_type)
    z_array2 = da.zeros((a_shape[0],3,a_shape[img_axes[1]]+3), dtype = a_type)
    # Insert blank cross into raw data
    
    b = da.concatenate((a[:,:,:a_half[1]],z_array, a[:,:,a_half[1]:]), axis = -1)

    b = da.concatenate((b[:, :a_half[0],:], z_array2, b[:,a_half[0]:,:]), axis = -2)

    return b
    
def get_mib_depth(hdr_info,fp):
"""Determine the total number of frames based on .mib file size.
    
    Parameters
    ----------
    hdr_info : dict
        Dictionary containing header info extracted from .mib file.
    fp : filepath
        Path to .mib file.
    
    Returns
    -------
    depth : int
        Number of frames in the stack
    """
    # Define standard frame sizes for quad and single medipix chips
    if hdr_info['Assembly Size'] == '2x2':
        mib_file_size_dict = {
        '1': 33536,
        '6': 262912,
        '12': 525056,
        '24': 1049344,
        }
    if hdr_info['Assembly Size'] == '1x1':
        mib_file_size_dict = {
        '1': 8576,
        '6': 65920,
        '12': 131456,
        '24': 262528,
        }
    
    file_size = os.path.getsize(fp[:-3]+'mib')
    if hdr_info['raw'] == 'R64':

        single_frame = mib_file_size_dict.get(str(hdr_info['Counter Depth (number)']))
        depth = int(file_size / single_frame)
    elif hdr_info['raw'] == 'MIB':
        if hdr_info['Counter Depth (number)'] =='1':
            # 1 bit and 6 bit non-raw frames have the same size
            single_frame = mib_file_size_dict.get('6')  
            depth = int(file_size / single_frame)
        else:
            single_frame = mib_file_size_dict.get(str(hdr_info['Counter Depth (number)']))
            depth = int(file_size / single_frame)
    
    return depth

def read_exposures(hdr_info, fp, pct_frames_to_read = 0.1, mmap_mode='r'):
    """
    Looks into the frame times of the first 10 pct of the frames to see if they are 
    all the same (TEM) or there is a flyback (4D-STEM).
    For this to work, the tick in the Merlin softeare to print exp time into header
    must be selected!
    
    Parameters
    -------------
    hdr_info: dict
        Output from parse_hdr function
    fp: str
        MIB file name / path
    pct_frames_to_read : float
        Percentage of frames to read, default value 0.1
    mmap_mode: str
        Memmpa read mode - default is 'r'
    Returns
    ------------
    exp_time: list
        List of frame exposure times
    """
    width = hdr_info['width']
    height = hdr_info['height']
    depth = get_mib_depth(hdr_info, fp)
    offset = hdr_info['offset']
    data_length = hdr_info['data-length']
    data_type = hdr_info['data-type']
    endian = hdr_info['byte-order']
    record_by = hdr_info['record-by']
    read_offset = 0

    if data_type == 'signed':
        data_type = 'int'
    elif data_type == 'unsigned':
        data_type = 'uint'
    elif data_type == 'float':
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    # mib data always big-endian
    endian = '>'
    data_type += str(int(data_length))
    data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    hdr_multiplier = (int(data_length)/8)**-1
    hdr_bits = int(hdr_info['data offset'] * hdr_multiplier)

    data = np.memmap(fp,
                     offset=read_offset,
                     dtype=data_type,
                     mode=mmap_mode)
    data = da.from_array(data)

    if record_by == 'vector':   # spectral image
        size = (height, width, depth)
        data = data.reshape(size)
    elif record_by == 'image':  # stack of images
        width_height = width * height

        size = (depth, height, width)

        #remove headers at the beginning of each frame and reshape

        if hdr_info['raw'] == 'R64':
            try: 
                data = data.reshape(-1,  width_height + hdr_bits)[:,71:79]
                data = data [:, ]
                data_crop = data[:int(depth*pct_frames_to_read)]
                d = data_crop.compute()
                exp_time = []
                for line in range(d.shape[0]):
                    str_list = [chr(d[line][n]) for n in range(d.shape[1])]
                    exp_time.append(float(''.join(str_list)))
            except ValueError:
                print('Frame exposure times are not appearing in header!')
                

        else: 
            try: 
                data = data.reshape(-1,  width_height + hdr_bits)[:,71:79]
                data = data [:, ]
                data_crop = data[:int(depth*pct_frames_to_read)]
                d = data_crop.compute()
                exp_time = []
                for line in range(d.shape[0]):
                    str_list = [chr(d[line][n]) for n in range(d.shape[1])]
                    exp_time.append(float(''.join(str_list)))
            except ValueError:
                print('Frame exposure times are not appearing in header!')
            

    elif record_by == 'dont-care':  # stack of images
        size = (height, width)
        data = data.reshape(size)
    return exp_time


def STEM_flag_dict(exp_times_list):
    """Determines whether a .mib file contains STEM or TEM data and how many 
    frames to skip due to triggering from a list of exposure times.
    
    Parameters
    ----------
    exp_times_list : list
        List of exposure times extracted from a .mib file.
    
    Returns
    -------
    output : dict
        Dictionary containing - STEM_flag, scan_X, exposure_time,
                                number_of_frames_to_skip, flyback_times
    """
    output = {}
    times_set = set(exp_times_list)
    # If single exposure times in header, treat as TEM data.
    if len(times_set) == 1:
        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = list(times_set)
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    #In case exp times not appearing in header treat as TEM data
    elif len(times_set) ==0: 
                             
        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = None
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # Otherwise, treat as STEM data.
    else:
        STEM_flag = 1
        # Check that the smallest time is the majority of the values
        exp_time = max(times_set, key = exp_times_list.count)
        if exp_times_list.count(exp_time) < int(0.9*len(exp_times_list)):
            print('Something wrong with the triggering!')
        peaks = [i for i, e in enumerate(exp_times_list) if e != exp_time]
        # Diff between consecutive elements of the array
        lines = np.ediff1d(peaks) 

        if len(set(lines)) == 1:
            scan_X = lines[0]
            frames_to_skip = peaks[0]
        else:
        # Assuming the last element to be the line length
            scan_X = lines[-1]  
            check = np.ravel(np.where(lines == scan_X, True, False))  
            # Checking line lengths
            start_ind = np.where(check == False)[0][-1] + 2
            frames_to_skip = peaks[start_ind] 
        
        flyback_times = list(times_set)
        flyback_times.remove(exp_time)
        output['STEM_flag'] = STEM_flag
        output['scan_X'] = scan_X
        output['exposure time'] = exp_time
        output['number of frames_to_skip'] = frames_to_skip
        output['flyback_times'] = flyback_times

    return output


def read_mib(fp, hdr_info, mmap_mode='r'):
    """Read a raw .mib file using memory mapping where the array
    is stored on disk and not directly loaded, but may be treated
    like a numpy.ndarray.                                                 
                                                              
                      
 
    Parameters
    ----------
    fp: str
        Filepath of .mib file to be loaded.
                                     
    hdr_info: dict
        A dictionary containing the keywords as parsed by read_hdr
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode
        (see `numpy.memmap`).  The mode has no effect for pickled or
        zipped files.
    
    Returns
    -------
    data : numpy.memmap
        
    """ 
    
    reader_offset = 0

    width = hdr_info['width']
    height = hdr_info['height']

    offset = hdr_info['offset']
    data_length = hdr_info['data-length']
    data_type = hdr_info['data-type']
    endian = hdr_info['byte-order']
    record_by = hdr_info['record-by']

    depth = get_mib_depth(hdr_info, fp)
            

    if data_type == 'signed':
        data_type = 'int'
    elif data_type == 'unsigned':
        data_type = 'uint'
    elif data_type == 'float':
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    # mib data always big-endian 
    endian = '>'

    data_type += str(int(data_length))
    if data_type == 'uint1':
        data_type = 'uint8'
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)
    
    
    hdr_multiplier = (int(data_length)/8)**-1
    hdr_bits = int(hdr_info['data offset'] * hdr_multiplier)

    
    data = np.memmap(fp,
                     offset=reader_offset,
                     dtype=data_type,
                     mode=mmap_mode)
    data = da.from_array(data)


    if record_by == 'vector':   # spectral image
        size = (height, width, depth)
        try:
            data = data.reshape(size)
        # in case of incomplete frame:    
        except ValueError:
            if hdr_info['raw'] == 'R64':

                data = data.reshape(depth)
            
        
    elif record_by == 'image':  # stack of images
        width_height = width * height

        size = (depth, height, width)

        #remove headers at the beginning of each frame and reshape

        if hdr_info['Assembly Size'] == '2x2':

            data = data.reshape(-1, width_height + hdr_bits)[:,-width_height:].reshape(depth, width, height)

        
        
        if hdr_info['raw'] == 'R64':
            if hdr_info['Counter Depth (number)'] == 24 or  hdr_info['Counter Depth (number)'] == 12:
                COLS = 4

            if hdr_info['Counter Depth (number)'] == 1:
                COLS = 64

            if hdr_info['Counter Depth (number)'] == 6:
                COLS = 8
                

            data = data.reshape((depth*width_height))
            
            data = data.reshape(depth,height * (height//COLS) , COLS )
                
        
            data = da.flip(data,2)
            
            if hdr_info['Assembly Size'] == '2x2':
                
                try:
                    data = data.reshape((depth*width_height))
                    
                    data = data.reshape(depth,512 // 2, 512 * 2 )
                except ValueError:
                    data = data.reshape((depth*width_height))
                    data = data.reshape(depth,512 // 2, 512 * 2 )
                
                det1 = data[:, :, 0:256]
                det2 = data[:, :, 256:512]
                det3 = data[:, :, 512:512 + 256]
                det4 = data[:, :, 512+256:]
                
                det3 = da.flip(det3, 2)
                det3 = da.flip(det3, 1)
                
                det4 = da.flip(det4, 2)
                det4 = da.flip(det4, 1)
                
                data = da.concatenate((da.concatenate((det1,det3),1),da.concatenate((det2,det4),1)),2)
                
        if hdr_info['Assembly Size'] == '1x1':

            data = data.reshape(-1, width_height + hdr_bits)[:,-width_height:].reshape(depth, width, height)
            data = data.reshape(depth,256, 256 )
        
    elif record_by == 'dont-care':  # stack of images
        size = (height, width)
        data = data.reshape(size)

    return data


def mib_dask_reader(mib_filename):
    """Read a .mib file using dask and return as a lazy pyXem / hyperspy signal.
    
    Parameters
    ----------
    mib_filename : str
        The name of the .mib file to be read.
    
    Returns
    -------
    data_hs : hyperspy.signals.Signal2D
        
    data_dict : dict
    
    TODO: add data_dict as attributes to data
        
    """
    hdr_stuff = parse_hdr(mib_filename)
    data = read_mib(mib_filename, hdr_stuff)
    exp_times_list = read_exposures(hdr_stuff, mib_filename)
    data_dict = STEM_flag_dict(exp_times_list)

    if hdr_stuff['Assembly Size'] == '2x2':
        data = add_crosses(data)
        
    data_hs = hs.signals.Signal2D(data).as_lazy()
    
    return data_hs , data_dict
