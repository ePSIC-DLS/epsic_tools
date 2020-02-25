import argparse
import os
from IdentifyPotentialConversions import check_differences
import gc
from mib_dask_import import mib_to_h5stack
from mib_dask_import import h5stack_to_hs
from mib_dask_import import parse_hdr
from mib_dask_import import mib_dask_reader
from mib_dask_import import get_mib_depth
import time
import pprint
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

def change_dtype(d):
    """
    Changes the data type of hs object d to int16
    Parameters:
    -----------
    d : hyperspy.signals.Signal2D
        Signal2D object with dtype float64

    Returns
    -------
    d : hyperspy.signals.Signal2D
        Signal2D object following dtype change to int16
    """
    d = d.data.astype('int16')
    d = hs.signals.Signal2D(d)
    
    return d

def bin_sig(d, bin_fact):
    """
    bins the reshaped 4DSTEMhs object by bin_fact on signal (diffraction) plane
    Parameters:
    ------------
    d: hyperspy.signals.Signal2D - can also be lazy
        reshaped to scanX, scanY | DetX, DetY
        This needs to be computed, i.e. not lazy, to work. If lazy, and binning 
        not aligned with dask chunks raises ValueError 
    Returns:
    -------------
    d_sigbin: binned d in the signal plane
    """
    # figuring out how many pixles to crop before binning
    # we assume the Detx and DetY dimensions are the same
    to_crop = d.axes_manager[-1].size % bin_fact
    d_crop = d.isig[to_crop:,to_crop:]
    try:
        d_sigbin = d_crop.rebin(scale=(1,1,bin_fact,bin_fact))
    except ValueError:
        print('Rebinning does not align with data dask chunks. Pass non-lazy signal before binning.')
        return
    return d_sigbin

def bin_nav(d, bin_fact):
    """
    bins the reshaped 4DSTEMhs object by bin_fact on navigation (probe scan) plane
    Parameters:
    ------------
    d: hyperspy.signals.Signal2D - can also be lazy
        reshaped to scanX, scanY | DetX, DetY
        This needs to be computed, i.e. not lazy, to work. If lazy, and binning 
        not aligned with dask chunks raises ValueError 
    Returns:
    -------------
    d_navbin: binned d in the signal plane
    """
    # figuring out how many pixles to crop before binning
    # we assume the Detx and DetY dimensions are the same
    to_cropx = d.axes_manager[0].size % bin_fact
    to_cropy = d.axes_manager[1].size % bin_fact
    d_crop = d.inav[to_cropx:,to_cropy:]
    try:
        d_navbin = d_crop.rebin(scale=(bin_fact,bin_fact,1,1))
    except ValueError:
        print('Rebinning does not align with data dask chunks. Pass non-lazy signal before binning.')
        return
    return d_navbin

def convert(beamline, year, visit, mib_to_convert, folder):
    """Convert a set of Merlin/medipix .mib files in a set of time-stamped folders
    into corresponding .hdf5 raw data files, and a series of standard images contained
    within a similar folder structure in the processing folder of the same visit.
    For 4DSTEM files with scan array larger than 300*300 it writes the stack into a h5 file
    first and then loads it lazily and reshapes it.

    Parameters
    ----------
    beamline : str

    year : str

    visit : str

    mib_to_convert : list
        List of MIB files to convert

    folder : optional - in case only a specific folder in a visit needs converting, e.g. sample1/dataset1/

    Returns
    -------
        - reshaped 4DSTEM HDF5 file
        - The above file binned by 4 in the diffraction plane
        - The above file binned by 4 in the navigation plane
        - HSPY, TIFF file and JPG file of incoherent BF reconstruction
        - HSPY, TIFF file and JPG file of sparsed sum of the diffraction patterns
        - An empty txt file is saved to show that the saving of HDF5 files is complete.

        TODO - Folder option is not working - fix!
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
                hdr_info = parse_hdr(os.path.join('/', mib_path, file))
                print(hdr_info)

        # If there is only 1 mib file in folder load it as a hyperspy Signal2D
        if mib_num == 1:
            # Load the .mib file and determine whether it contains TEM or STEM
            # data based on frame exposure times.
            try:
                print(mib_path)

                depth = get_mib_depth(hdr_info, hdr_info['title'] + '.mib')
                print(depth)
                # Only write the h5 stack for large scan arrays
                if depth > 300*300:
                    print('large file 4DSTEM file - first saving the stack into h5 file!')
                    # if folder:
                    #     h5_path = proc_location + '/' + folder + '/' + hdr_info['title'].split('/')[-2] + '/' + hdr_info['title'].split('/')[-1] + '.h5'
                    # else:
                    h5_path = proc_location + '/' + hdr_info['title'].split('/')[-2] + '/' + hdr_info['title'].split('/')[-1] + '.h5'
                    if not os.path.exists(os.path.dirname(h5_path)):
                        os.makedirs(os.path.dirname(h5_path))
                    print(h5_path)
                    mib_to_h5stack(hdr_info['title'] + '.mib', hdr_info, h5_path)
                    dp = mib_dask_reader(hdr_info['title'] + '.mib', h5_path)
                else:
                    print(hdr_info['title'] + '.mib')
                    dp = mib_dask_reader(hdr_info['title'] + '.mib')
                print(dp)
                pprint.pprint(dp.metadata)
                dp.compute(progressbar = False)
                t1 = time.time()
                if dp.metadata.Signal.signal_type == 'STEM':
                    STEM_flag = True
                else:
                    STEM_flag = False
#                scan_X = dp.metadata.Signal.scan_X

            except ValueError:
                print('file could not be read into an array!')
            # Process single .mib file identified as containing TEM data.
            # This just saves the data as an .hdf5 image stack.
            if STEM_flag is False:
                
                saving_path = proc_location +'/'+ hdr_info['title'].split('/')[-2] + '/'
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                

                print('saving here: ',saving_path)
                # Calculate summed diffraction pattern
                #dp_sum = max_contrast8(dp.sum())
                #dp_sum = change_dtype(dp_sum)
                # Save summed diffraction pattern
                #dp_sum.save(saving_path + '/' +mib_list[0]+'_sum', extension = 'jpg')
                t2 = time.time()
                # Save raw data in .hdf5 format
                dp.save(saving_path + '/' +mib_list[0] + data_dim(dp), extension = 'hdf5')
                tmp = []
                np.savetxt(saving_path+'/'+mib_list[0].rpartition('.')[0]+data_dim(dp)+'fully_saved', tmp)
                t3 = time.time()
            # Process single .mib file identified as containing STEM data
            # This reshapes to the correct navigation dimensions and
            else:
                # Define save path for STEM data
                # if folder:
                #     saving_path = proc_location + '/' + folder + '/' + hdr_info['title'].split('/')[-2] + '/'
                # else:
                saving_path = proc_location +'/'+ hdr_info['title'].split('/')[-2] + '/'
                # if not os.path.exists(saving_path):
                #     os.makedirs(saving_path)


                # Bin by factor of 4 in diffraction and navigation planes
                dp_bin_sig = bin_sig(dp,4)
                dp_bin_nav = bin_nav(dp,4)
                
                # Calculate sum of binned data

                ibf = dp_bin_sig.sum(axis=dp_bin_sig.axes_manager.signal_axes)
                # Rescale contrast of IBF image
                ibf = max_contrast8(ibf)
                ibf = change_dtype(ibf)

                # sum dp image of a subset dataset
                #dp_subset = dp.inav[0::int(dp.axes_manager[0].size / 50), 0::int(dp.axes_manager[0].size / 50)]
                #sum_dp_subset = dp_subset.sum()
                sum_dp_subset = dp_bin_sig.sum()
                sum_dp_subset = max_contrast8(sum_dp_subset)
                sum_dp_subset = change_dtype(sum_dp_subset)

                # save data

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
                print('Saving binned diffraction data: ' + mib_list[0].rpartition('.')[0] + '_binned.hdf5')
                dp_bin_sig.save(saving_path+ '/'+'binned_diff_' + mib_list[0].rpartition('.')[0]+data_dim(dp_bin_sig), extension = 'hdf5')
                print('Saved binned diffraction data: binned_' + mib_list[0].rpartition('.')[0] + '.hdf5')
                del dp_bin_sig
                print('Saving binned navigation data: ' + mib_list[0].rpartition('.')[0] + '_binned.hdf5')
                dp_bin_nav.save(saving_path+ '/'+'binned_nav_' + mib_list[0].rpartition('.')[0]+data_dim(dp_bin_nav), extension = 'hdf5')
                print('Saved binned navigation data: binned_' + mib_list[0].rpartition('.')[0] + '.hdf5')
                del dp_bin_nav
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

                dp.compute(progressbar=False)

                t1 = time.time()

                if STEM_flag is False:

                    dp_sum = max_contrast8(dp.sum())
                    dp_sum = change_dtype(dp_sum)
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
    return



def watch_convert(beamline, year, visit, folder):

    mib_dict = check_differences(beamline, year, visit, folder)
    # Holder for raw data path
    if folder:
        raw_location = os.path.join('/dls',beamline,'data', year, visit, os.path.relpath(folder))
    else:
        raw_location = os.path.join('/dls',beamline,'data', year, visit, 'Merlin')
    to_convert = mib_dict['MIB_to_convert']
    if bool(to_convert):
        convert(beamline, year, visit, to_convert, folder)


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
        data_dim_str = '_number_of_frames_' + dims[0]+'_detector_plane_'+dims[1]+'by'+dims[1]+'_'
    return data_dim_str


def main(beamline, year, visit, folder, folder_num):
    print(beamline, year, visit, folder)
    if folder=='False':
        folder = ''
        HDF5_dict= check_differences(beamline, year, visit)
    else:
        HDF5_dict= check_differences(beamline, year, visit, folder)

    # proc_path = HDF5_dict['processing_path']
    
    to_convert = HDF5_dict['MIB_to_convert']
    folder = to_convert[int(folder_num)-1].rpartition('/')[0].rpartition(visit)[2][1:]
    try:
        save_location = os.path.join('/dls',beamline,'data', year, visit, 'processing', folder)
        if os.path.exists(save_location) == False:
            os.makedirs(save_location)
        watch_convert(beamline, year, visit, folder)
        
    except Exception as e:
        print('** ERROR processing** \n ' , e)


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
    parser.add_argument('folder_num', nargs= '?', help='passed by scheduler')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()
    print(args)

    main(args.beamline, args.year, args.visit, args.folder, args.folder_num)
