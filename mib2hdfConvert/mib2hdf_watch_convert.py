import argparse
import os
from IdentifyPotentialConversions import check_differences
import gc
from mib_dask_import import mib_dask_reader
import time
import pprint
import reshape_4DSTEM_funcs as reshape
import hyperspy.api as hs
import numpy as np

hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()

def max_contrast8(d):
    """Rescales contrast of hyperspy Signal2D to 8-bit.
    
    Parameters
    ----------
    d : hyperspy.signals.Signal2D
        Signal2D object to be rescaled
    
    Returns
    -------
    d : hyperspy.signals.Signal2D
        Signal2D object following intensity rescaling
    """
    data = d.data
    data = data - data.min()
    if data.max() != 0:
        data = data * (255 / data.max())
    d.data = data
    return d

    
def convert(beamline, year, visit, mib_to_convert, folder):
    """Convert a set of Merlin/medipix .mib files in a set of time-stamped folders 
    into corresponding .hdf5 raw data files, and a series of standard images contained 
    within a similar folder structure in the processing folder of the same visit.
    
    The STEM / TEM is figured out by the exp times of frames.
    If STEM, then data is reshaped using the flyback frames.
    
    Parameters
    ----------
    beamline : str
    
    year : str
    
    visit : str

    mib_to_convert : list 
        List of MIB files to convert
    
    folder : 
    
    Returns
    -------
        - reshaped 4DSTEM HDF5 file
        - The above file binned by 4 in the diffraction plane
        - HSPY, TIFF file and JPG file of incoherent BF reconstruction
        - HSPY, TIFF file and JPG file of sparsed sum of the diffraction patterns
    """
    t1 = []
    t2 = []
    t3 = []
    
    t0 = time.time()
    # Define processing path in which to save outputs
        
    proc_location = os.path.join('/dls',beamline,'data', year, visit, 'processing', 'Merlin')
    if not os.path.exists(proc_location):
        os.mkdir(proc_location)
    
    # Get the raw data folders and assign as a set
    data_folders = []
    for path in mib_to_convert:
        data_folders.append(os.path.join(*path.split('/')[:-1]))
    data_folders_set = set(data_folders)
    
    # Loop over all data folders
    for mib_path in list(data_folders_set):

        print('********************************************************')
        print('Currently active in this directory: %s' % mib_path.split('/')[-1])
        
        # Determine mib files in this folder
        mib_num = 0
        mib_list = []
        for file in os.listdir('/'+ mib_path):
            if file.endswith('mib'):
                mib_num += 1
                mib_list.append(file)

        # If there is only 1 mib file in folder load it as a hyperspy Signal2D
        if mib_num == 1:
            # Load the .mib file and determine whether it contains TEM or STEM
            # data based on frame exposure times.
            try:

                dp = mib_dask_reader('/' +mib_path + '/'+ mib_list[0])
                pprint.pprint(dp.metadata)
                dp.compute() 
                t1 = time.time()
                if dp.metadata.Signal.signal_type == 'STEM':
                    STEM_flag = True
                else: 
                    STEM_flag = False
                scan_X = dp.metadata.Signal.scan_X

                
            except ValueError:
                print('file could not be read into an array!')
            # Process single .mib file identified as containing TEM data.
            # This just saves the data as an .hdf5 image stack.
            if STEM_flag is False: 
                if folder:

                    temp1 = mib_path.split('/')
                    temp2 = folder.split('/')
                    ind = temp1.index(temp2[0])
                    saving_path = proc_location +'/'+os.path.join(*mib_path.split('/')[ind:])
                else:    
                    saving_path = proc_location +'/'+ os.path.join(*mib_path.split('/')[6:])
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                print('saving here: ',saving_path)
                # Calculate summed diffraction pattern
                dp_sum = max_contrast8(dp.sum())
                # Save summed diffraction pattern
                dp_sum.save(saving_path + '/' +mib_list[0]+'_sum', extension = 'jpg')
                t2 = time.time()
                # Save raw data in .hdf5 format
                dp.save(saving_path + '/' +mib_list[0] + data_dim(dp), extension = 'hdf5')
                t3 = time.time()
            # Process single .mib file identified as containing STEM data
            # This reshapes to the correct navigation dimensions and 
            else:
                # Define save path for STEM data
                saving_path = proc_location +'/'+ os.path.join(*mib_path.split('/')[6:])
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                img_flag = 0
                
                print('Data loaded to hyperspy')
                # checks to see if it is a multi-frame data before reshaping
                if dp.axes_manager[0].size > 1:
                # Attempt to reshape the data based on exposure times
                    try:
                        dp = reshape.reshape_4DSTEM_FlyBack(dp)
                        print('Data reshaped using flyback pixel to: '+ str(dp.axes_manager.navigation_shape))
                    # If exposure times fail use data size
                    except:
                        print('Data reshape using flyback pixel failed! - Reshaping using scan size instead.')
                        num_frames = dp.axes_manager[0].size
                        dp = reshape.reshape_4DSTEM_FrameSize(dp, scan_X, int(num_frames / scan_X))
                     # Crop quad chip data to 512X512 so even numbers for binning
                    if dp.axes_manager[-1].size  == 515:
                        dp_crop = dp.isig[1:-2,1:-2]
                        print('cropped data to 512*512 in order to bin')
                    else:
                        dp_crop = dp
                    # Set img_flag hardcoded
                    # This part is the "pre-processing pipeline"
                    img_flag = 1
                    # Bin by factor of 4 in diffraction pattern
                    dp_bin = dp_crop.rebin(scale = (1,1,4,4))
                    # Calculate sum of binned data 
                   
                    ibf = dp_bin.sum(axis=dp_bin.axes_manager.signal_axes)
                    # Rescale contrast of IBF image
                    ibf = max_contrast8(ibf)
                 
                    # sum dp image of a subset dataset
                    dp_subset = dp.inav[0::int(dp.axes_manager[0].size / 50), 0::int(dp.axes_manager[0].size / 50)]
                    sum_dp_subset = dp_subset.sum()
                    sum_dp_subset = max_contrast8(sum_dp_subset)
                    
                    # save data
                
                    if img_flag == 1:
                        try: 
                            print('Saving average diffraction pattern')
                            file_dp = mib_list[0].rpartition('.')[0]+ '_subset_dp'
                            sum_dp_subset = hs.signals.Signal2D(sum_dp_subset)
                            sum_dp_subset.save(saving_path+'/'+file_dp, extension = 'tiff')
                            sum_dp_subset.save(saving_path+'/'+file_dp, extension = 'jpg')
                            sum_dp_subset.save(saving_path+'/'+file_dp)
                            print('Saving ibf image')
                            print(saving_path)
                            ibf = hs.signals.Signal2D(ibf)
                            file_ibf =  mib_list[0].rpartition('.')[0]+ '_ibf'
                            ibf.save(saving_path+'/'+file_ibf, extension = 'tiff')
                            ibf.save(saving_path+'/'+file_ibf, extension = 'jpg')
                            ibf.save(saving_path+'/'+file_ibf)

                            t2 = time.time()
                        except:
                            print('Issue with saving images!')

                        # Save binned data in .hdf5 file
                    print('Saving binned data: ' + mib_list[0].rpartition('.')[0] + '_binned.hdf5')
                    dp_bin.save(saving_path+ '/'+'binned_' + mib_list[0].rpartition('.')[0]+data_dim(dp_bin), extension = 'hdf5')
                    print('Saved binned data: binned_' + mib_list[0].rpartition('.')[0] + '.hdf5')
                    del dp_bin
                 # Save complete .hdf5 files 
                print('Saving hdf5 : ' + mib_list[0].rpartition('.')[0] +'.hdf5')
                dp.save(saving_path+'/'+mib_list[0].rpartition('.')[0]+data_dim(dp), extension = 'hdf5')
                print('Saved hdf5 : ' + mib_list[0].rpartition('.')[0] +'.hdf5')
                tmp = []
                np.savetxt(saving_path+'/'+mib_list[0].rpartition('.')[0]+data_dim(dp)+'fully_saved', tmp)
                t3 = time.time()
                
                del dp
                gc.collect()
        # If there are multiple mib files in folder, load them.
        # TODO: Assumes TEM data - needs updating to consider STEM data.
        elif mib_num > 1:

            if folder:
            # find index
                temp1 = mib_path.split('/')
                temp2 = folder.split('/')
                ind = temp1.index(temp2[0])
                saving_path = proc_location +'/'+os.path.join(*mib_path.split('/')[ind:])
            else:    
                saving_path = proc_location +'/'+ os.path.join(*mib_path.split('/')[6:])    
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            print('saving here: ', saving_path)
            for k, file in enumerate(mib_list):
                
                print(mib_path)
                print(file)
                t0 = time.time()
                dp = mib_dask_reader('/' +mib_path + '/'+ file)
                pprint.pprint(dp.metadata)
                if dp.metadata.Signal.signal_type == 'STEM':
                    STEM_flag = True
                else: 
                    STEM_flag = False
                    
                scan_X = dp.metadata.Signal.scan_X
                dp.compute() 
                
                t1 = time.time()
                
                if STEM_flag is False:

                    dp_sum = max_contrast8(dp.sum())
                    dp_sum.save(saving_path + '/' +file+'_sum', extension = 'jpg')
                    t2 = time.time()
                    dp.save(saving_path + '/' +file, extension = 'hdf5')
                    t3 = time.time()

         # Print timing information
        if t1 is not None:
            print('time to load data: ', int(t1-t0))
        if t2 is not None:
            print('time to save last image: ', int(t2-t0)) 
        if t3 is not None:
            print('time to save full hdf5: ', int(t3-t0))  
                    

                    
def watch_convert(beamline, year, visit, folder):
    
    [to_convert, mib_files, mib_to_convert] = check_differences(beamline, year, visit, folder)
    # Holder for raw data path
    if folder:
        raw_location = os.path.join('/dls',beamline,'data', year, visit, os.path.relpath(folder))
    else:
        raw_location = os.path.join('/dls',beamline,'data', year, visit, 'Merlin')
    if bool(to_convert):
        convert(beamline, year, visit, mib_to_convert, folder)
    else:

        watch_check = 'Y'
        if (watch_check == 'Y' or watch_check == 'y'):
            print(raw_location)
            path_to_watch = raw_location
            [to_convert, mib_files, mib_to_convert] = check_differences(beamline, year, visit, folder)
            before = dict ([(f, None) for f in os.listdir (path_to_watch)])
            while True:
                time.sleep (60)
                after = dict ([(f, None) for f in os.listdir (path_to_watch)])
                added = [f for f in after if not f in before]
                if added: 
                    print("Added dataset: ", ", ".join (added))

                    new_data_folder = os.listdir (path = os.path.join(path_to_watch, added[-1]))
                    for f in new_data_folder:
                        if f.endswith('mib'):
                            wait_flag = 1
                            while wait_flag == 1:
                                try:
                                    print('file name: ', f)

                                    # above give the file size from source
                                    # but this below throws an error while copy is not complete:
                                    f_size = os.path.getsize(f)
                                    # print(f_size)
                                    print('file size: ', f_size)
                                    wait_flag = 0
                                except FileNotFoundError:
                                    time.sleep(20)
                                    print('waiting for mib data to copy!!')
                                    print(os.path.isfile(os.path.join(path_to_watch, added[-1], f)))
                                    if os.path.isfile(os.path.join(path_to_watch, added[-1], f)):
                                        wait_flag = 0
                                    else:
                                        pass
           
                    [to_convert, mib_files, mib_to_convert] = check_differences(beamline, year, visit, folder)
                    convert(beamline, year, visit, mib_files, folder)
                before = after

def data_dim(data):
    """
    This function gets the data hyperspy object and outputs the dimensions as string
    to be written into file name ultimately saved.
    input:
        data
    returns:
        data_dim_str: data dimensions as string
    """ 
    dims = str(data.data.shape)[1:-1].replace(' ','').split(',')
    # 4DSTEM data
    if len(data.data.shape) == 4:
        data_dim_str = '_scan_array_'+dims[0]+'by'+dims[1]+'_diff_plane_'+dims[2]+'by'+dims[3]+'_'
    # stack of images
    if len(data.data.shape) == 3:
        data_dim_str = '_number_of_frames_' + dims[0]+'_detector_plane_'+dims[2]+'by'+dims[3]+'_'
    return data_dim_str
        
    
def main(beamline, year, visit, folder = None):
    watch_convert(beamline, year, visit, folder)


if __name__ == "__main__":
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers =20, memory_limit = 100e9)
    client = Client(cluster)
    parser = argparse.ArgumentParser()
    parser.add_argument('beamline', help='Beamline name')
    parser.add_argument('year', help='Year')
    parser.add_argument('visit', help='Session visit code')
    parser.add_argument('folder', nargs= '?',default=None, help='OPTION to add a specific folder within a visit \
                        to look for data, e.g. sample1/dataset1/. If None the assumption would be to look in Merlin folder')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()
    
    main(args.beamline, args.year, args.visit, args.folder)
