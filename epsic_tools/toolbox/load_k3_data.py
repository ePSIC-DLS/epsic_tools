"""
Code below was developed by:
- Liam Spillane (Gatan): DM5 file format and initial code to load 4D-STEM data from them
- Mohsen Danaie (ePSIC): Fixed some bugs, added lazy loading and extended to cover 4D-STEM and EELS

This will ultimately be added to https://github.com/hyperspy/rosettasciio
"""

import h5py
# from tkinter import filedialog as fd
import numpy as np
import dask.array as da
import glob

import os
import sys

sys.argv.extend(['-a', ' '])


class InSitu_K3_Reader:

    def __init__(self, filePath):
        print('Simple DM5 reader for InSitu K3 datasets:')
        print('')

        # Save to class member variables
        self.filePath = filePath

        # Additional Initialisation functions
        self.OpenDirectorImageFile_DM5()
        self.GetDocumemtObjectListAttributes()
        self.GetImageSourceListAtrributes()
        self.GetImageListAtributes()
        self.GetTotalFrameNum()

        # Variable defining index of active raw file. Must be initialised as 0
        self.rawFileIndex = 0

    def OpenDirectorImageFile_DM5(self):
        print('Opening: ' + str(self.filePath))
        f = h5py.File(self.filePath, "r")

        # Save to class member variables
        self.f = f

    def GetDocumemtObjectListAttributes(self):
        group1 = self.f.get('/DocumentObjectList/[0]')

        sourceIndex = group1.attrs['ImageSource']
        imageDisplayType = group1.attrs['ImageDisplayType']

        # Save to class member variables
        self.sourceIndex = sourceIndex
        self.imageDisplayType = imageDisplayType

    def GetImageSourceListAtrributes(self):

        group2Path = '/ImageSourceList/[' + str(self.sourceIndex) + ']'

        group2 = self.f.get(group2Path)

        imageRef = group2.attrs['ImageRef']
        className = group2.attrs['ClassName'].decode('utf-8')

        # Save to class member variables
        self.imageRef = imageRef
        self.className = className

    def GetImageListAtributes(self):
        group3Path = '/ImageList/[' + str(self.imageRef) + ']'
        group3 = self.f.get(group3Path)

        # get the ImageData tagGroup
        imageData = group3.get('ImageData')

        # get data as numpy array
        data = imageData.get('Data')

        #Save to class member variables
        self.imageData = imageData
        self.data = data

    def GetDimCalibrations(self, dimIndex: int):
        #Function will print origin, scale and units for the defined dimension
        #4DSTEM data has 4 dimensions, starting at index 0

        # Check N is integer
        if not isinstance(dimIndex, int):
            raise TypeError(f"Expected an integer, got {type(dimIndex).__name__}")

        # get the calibrations of the x-axis
        group = self.imageData.get('Calibrations/Dimension/[' + str(dimIndex) + ']')

        dimOrigin = group.attrs['Origin']
        dimScale = group.attrs['Scale']
        dimUnits = group.attrs['Units'].decode('latin-1')

        print('\nDimension: ' + str(dimIndex))
        print('Origin [' + str(dimIndex) + ']:' + str(dimOrigin))
        print('Scale [' + str(dimIndex) + ']:' + str(dimScale))
        print('Units [' + str(dimIndex) + ']:' + str(dimUnits))

    def OpenDataInDM(self, dataArray):
        #Function will create DM image object from a numpy array and show this image in the active DM workspace
        #If this script is executed outside of DigtalMicrograph, the visualization code is skipped.

        if 'DigitalMicrograph' in sys.modules:
                img = DM.CreateImage(dataArray.copy())
                img.GetTagGroup().SetTagAsBoolean('Meta Data:Data Order Swapped', True)
                img.GetTagGroup().SetTagAsString('Meta Data:Format', 'Diffraction image')

                img.ShowImage()
        else:
            print("")
            print("DigitalMicrograph module not present. Array visualization code skipped.")

    def GetNthFrame(self, N: int):
        # Function will read out a timeslice from the InSitu dataset and ouput as a number array
        # Any value of N may be chosen from 0 up to the last timeslice in the timeseries
        # The rawfile index is determined automatically depending on the value of N chosen

        # Check N is integer
        if not isinstance(N, int):
            raise TypeError(f"Expected an integer, got {type(N).__name__}")

        directorImageFilePath = self.filePath

        # Find raw file from director image filePath. Assumes director and raw files are in same folder
        rawFilePath = directorImageFilePath[:-4] + ".raw"
        raw_length = os.path.getsize(rawFilePath)

        # Calculate the number of pixels and bytes per frame
        data_type = self.data.dtype
        pixels_per_frame = np.prod(self.data.shape, dtype='uint64')
        bytes_per_frame = self.data.nbytes

        # Calculate the starting byte position for the Nth frame
        read_start = N * bytes_per_frame

        # Determine the correct raw file and offset within that file
        rawFileIndex = read_start // raw_length
        read_start = read_start % raw_length

        # Adjust raw file path if necessary
        if rawFileIndex != 0:
            rawFilePath = os.path.splitext(rawFilePath)[0] + ".raw_" + str(rawFileIndex)

        # Read from raw file
        # Code supports readout from datasets containing multiple raw files
        if read_start + bytes_per_frame > raw_length:
            count_1 = int((raw_length - read_start) // data_type.itemsize)
            count_2 = int(pixels_per_frame - count_1)

            array_1 = np.fromfile(rawFilePath, dtype=data_type, count=count_1, offset=read_start)

            rawFileIndex += 1
            rawFilePath = os.path.splitext(rawFilePath)[0] + ".raw_" + str(rawFileIndex)

            array_2 = np.fromfile(rawFilePath, dtype=data_type, count=count_2, offset=0)
            array = np.concatenate([array_1, array_2])
        else:
            array = np.fromfile(rawFilePath, dtype=data_type, count=pixels_per_frame, offset=read_start)

        array = array.reshape(self.data.shape)
        print(f"pixels_per_frame: {pixels_per_frame}")
        return array

    def GetTotalFrameNum(self):
        directorImageFilePath = self.filePath

        # From attributes of the file
        with h5py.File(directorImageFilePath, 'r') as f:
            frame_num_attr = f['ImageList/[1]/ImageTags/In-situ/Recorded'].attrs['# Frames']

        # Find raw file from director image filePath. Assumes director and raw files are in same folder
        # Calculate total bytes of all raw files
        file_name = os.path.basename(directorImageFilePath).split('.')[0]
        # print(file_name)
        raw_list = glob.glob(os.path.dirname(directorImageFilePath) + f'/{file_name}.raw*')

        file_size_counter = 0
        for file in sorted(raw_list):
            file_size_counter += os.path.getsize(file)
        
        # rawFilePath = directorImageFilePath[:-4] + ".raw"
        # raw_length = os.path.getsize(rawFilePath)

        # Calculate the number of pixels and bytes per frame
        data_type = self.data.dtype
        pixels_per_frame = np.prod(self.data.shape, dtype='uint64')
        bytes_per_frame = self.data.nbytes   

        frame_num_calc = file_size_counter / bytes_per_frame

        # Check if the value from the attributes match to that of the calculated
        if round(frame_num_calc) == frame_num_attr:
            print('The number of frames in the RAW files matches to the stated Recorded number in DM5 file.')
            print(f'Total number of frames acquired: {frame_num_attr}')
        else:
            print('The number of frames in the RAW files DOES NOT match to the stated Recorded number in DM5 file!')
            print(f'Frame numbers written to RAW: {round(frame_num_calc)}')
            print(f'Frame numbers stated in DM5: {frame_num_attr}')
            print('Retuning the number in the RAW files.')
        return round(frame_num_calc)

        
    
    def GetNthFrame_lazy(self, N: int):
        # Function will read out a timeslice from the InSitu dataset and ouput as a number array
        # Any value of N may be chosen from 0 up to the last timeslice in the timeseries
        # The rawfile index is determined automatically depending on the value of N chosen

        # Check N is integer
        if not isinstance(N, int):
            raise TypeError(f"Expected an integer, got {type(N).__name__}")

        directorImageFilePath = self.filePath

        # Find raw file from director image filePath. Assumes director and raw files are in same folder
        rawFilePath = directorImageFilePath[:-4] + ".raw"
        raw_length = os.path.getsize(rawFilePath)

        # Calculate the number of pixels and bytes per frame
        data_type = self.data.dtype
        pixels_per_frame = np.prod(self.data.shape, dtype='uint64')
        bytes_per_frame = self.data.nbytes

        # Calculate the starting byte position for the Nth frame
        read_start = N * bytes_per_frame

        # Determine the correct raw file and offset within that file
        rawFileIndex = read_start // raw_length
        read_start = read_start % raw_length

        # Adjust raw file path if necessary
        if rawFileIndex != 0:
            rawFilePath = os.path.splitext(rawFilePath)[0] + ".raw_" + str(rawFileIndex)

        # Read from raw file
        # Code supports readout from datasets containing multiple raw files
        if read_start + bytes_per_frame > raw_length:
            count_1 = int((raw_length - read_start) // data_type.itemsize)
            count_2 = int(pixels_per_frame - count_1)

            # array_1 = np.fromfile(rawFilePath, dtype=data_type, count=count_1, offset=read_start)
            memmap_1 = np.memmap(rawFilePath, dtype = data_type, mode = 'r', offset = read_start, shape = (count_1,))

            rawFileIndex += 1
            rawFilePath = os.path.splitext(rawFilePath)[0] + ".raw_" + str(rawFileIndex)

            # array_2 = np.fromfile(rawFilePath, dtype=data_type, count=count_2, offset=0)
            memmap_2 = np.memmap(rawFilePath, dtype = data_type, mode = 'r', offset = 0, shape = (count_2,))
            array = np.concatenate([memmap_1, memmap_2])
        else:
            # array = np.fromfile(rawFilePath, dtype=data_type, count=pixels_per_frame, offset=read_start)
            array = np.memmap(rawFilePath, dtype = data_type, mode = 'r', offset = read_start, shape = (pixels_per_frame,))

        array = array.reshape(self.data.shape)
        # dask_arrays = [da.from_array(array, chunks=(1, height, width)) for memmap in memmaps]
        if len(self.data.shape) == 3:
            dask_array = da.from_array(array, chunks=(1, 1, self.data.shape[2])) 
        elif len(self.data.shape) == 4:
            dask_array = da.from_array(array, chunks=(1, 1, self.data.shape[2], self.data.shape[3])) 
        else:
            print("array shape not addressed here")
        # print(f"pixels_per_frame: {pixels_per_frame}")
        return dask_array


