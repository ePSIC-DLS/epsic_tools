import argparse
import os
from IdentifyPotentialConversions import check_differences
import gc
from mib_dask_import import mib_dask_reader
import time
import pprint
import reshape_4DSTEM_funcs as reshape
import hyperspy.api as hs

#%%
def max_contrast8(d):
    data = d.data
    data = data - data.min()
    if data.max() != 0:
        data = data * (255 / data.max())
    d.data = data
    return d

#%%
    
def convert(beamline, year, visit, mib_to_convert, folder):
    """    
    This is to convert a set of time-stamped 4DSTEM folders (mib_files) into a
    similar folder structure in the processing folder of the same visit. The
    following files get saved for each stack:
        - reshaped 4DSTEM HDF5 file
        - The above file binned by 4 in the diffraction plane
        - TIFF file and JPG file of incoherent BF reconstruction
        - TIFF file and JPG file of sparsed sum of the diffraction patterns
    The STEM / TEM is figured out by the exp times of frames
    If STEM, thed data is reshaped using the flyback frames
    """
    t1 = []
    t2 = []
    t3 = []
    
    t0 = time.time()
#    if folder:
#        raw_location = os.path.join('/dls',beamline,'data', year, visit, os.path.relpath(folder))
#    else:
#        raw_location = os.path.join('/dls',beamline,'data', year, visit, 'Merlin')  
        
    proc_location = os.path.join('/dls',beamline,'data', year, visit, 'processing', 'Merlin')
    if not os.path.exists(proc_location):
        os.mkdir(proc_location)
    
    # getting the raw data folders as a set
    data_folders = []
    for path in mib_to_convert:
        data_folders.append(os.path.join(*path.split('/')[:-1]))
    data_folders_set = set(data_folders)
    
    # main loop 
    for mib_path in list(data_folders_set):
        #time0 = time.time()
        print('********************************************************')
        print('Currently active in this directory: %s' % mib_path.split('/')[-1])
        
        # get number of mib files in this folder
        mib_num = 0
        mib_list = []
        for file in os.listdir('/'+ mib_path):
            if file.endswith('mib'):
                mib_num += 1
                mib_list.append(file)
        #print(mib_num)
        #print(mib_list)
        #print('STEM_flag: ', STEM_flag)
        
        if mib_num == 1:
            try:
                #print(mib_path)
                #print(mib_list[0])
                
                dp = mib_dask_reader('/' +mib_path + '/'+ mib_list[0])
                pprint.pprint(dp[1], width=1)
                dp[0].compute() # is replacing needed?
                dp_data = dp[0]
                t1 = time.time()
                STEM_flag = dp[1].get('STEM_flag')
                scan_X = dp[1].get('scan_X')
                t2 = time.time()
                
                #print('loaded the mib file')
                
            except ValueError:
                print('file could not be read into an array!')
                
                continue
            if (STEM_flag == 0 or STEM_flag == '0'): # if it is TEM data
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
                print('saving here: ',saving_path)
                dp_data.save(saving_path + '/' +mib_list[0], extension = 'hdf5')
                dp_sum = max_contrast8(dp_data.sum())
                dp_sum.save(saving_path + '/' +mib_list[0]+'_sum', extension = 'jpg')
            else:
                saving_path = proc_location +'/'+ os.path.join(*mib_path.split('/')[6:])
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                img_flag = 0
                
                print('Data loaded to hyperspy')
                # checks to see if it is a multi-frame data before reshaping
                if dp[0].axes_manager[0].size > 1:
                # attampt to reshape the data 
                    try:
                        dp_data = reshape.reshape_4DSTEM_FlyBack(dp)
                        print('Data reshaped using flyback pixel to: '+ str(dp_data.axes_manager.navigation_shape))
                    except:
                        print('Data reshape using flyback pixel failed! - Reshaping using scan size instead.')
                        num_frames = dp_data.axes_manager[0].size
                        dp_data = reshape.reshape_4DSTEM_FrameSize(dp[0], scan_X, int(num_frames / scan_X))
                        
                    if dp_data.axes_manager[-1].size  == 515:
                        dp_crop = dp_data.isig[1:-2,1:-2]
                        print('cropped data to 512*512 in order to bin')
                    else:
                        dp_crop = dp_data
                    img_flag = 1
#                    a_chunk=(dp_crop.axes_manager[0].size//10)
#                    b_chunk=(dp_crop.axes_manager[3].size//4)
#                    dp_da = dp_crop.data.rechunk((a_chunk, a_chunk, b_chunk, b_chunk))
#                    dp_crop = hs.signals.Signal2D(dp_da).as_lazy()
                    dp_bin = dp_crop.rebin(scale = (1,1,4,4))
               
                   
                    ibf = dp_bin.sum(axis=dp_bin.axes_manager.signal_axes)
                    # print(ibf.size)
                    #ibf = hs.signals.Signal2D(ibf)
                    ibf = max_contrast8(ibf)
                 
                    # sum dp image of a subset dataset
                    dp_subset = dp_data.inav[0::int(dp_data.axes_manager[0].size / 50), 0::int(dp_data.axes_manager[0].size / 50)]
                    sum_dp_subset = dp_subset.sum()
                    sum_dp_subset = max_contrast8(sum_dp_subset)
                    # img_flag = 1
                    
                    # save data
                
                    if img_flag == 1:
                        try: 
                            print('Saving average diffraction pattern')
                            file_dp = mib_list[0].rpartition('.')[0]+ '_subset_dp'
                            sum_dp_subset = hs.signals.Signal2D(sum_dp_subset)
                            sum_dp_subset.save(saving_path+'/'+file_dp, extension = 'tiff')
                            sum_dp_subset.save(saving_path+'/'+file_dp, extension = 'jpg')
                            print('Saving ibf image')
                            print(saving_path)
                            ibf = hs.signals.Signal2D(ibf)
                            file_ibf =  mib_list[0].rpartition('.')[0]+ '_ibf'
                            ibf.save(saving_path+'/'+file_ibf, extension = 'tiff')
                            ibf.save(saving_path+'/'+file_ibf, extension = 'jpg')
                            ibf.save(saving_path+'/'+file_ibf)
    #                        ibf.data.to_hdf5(file_ibf, saving_path, compression = 'gzip')
                            t2 = time.time()
                        except:
                            print('Issue with saving images!')
                            # continue
                                  
#                if dp_data.axes_manager[0].size > 1:
                    print('Saving binned data: ' + mib_list[0].rpartition('.')[0] + '_binned.hdf5')
                    dp_bin.save(saving_path+ '/'+'binned_' + mib_list[0], extension = 'hdf5')
                    print('Saved binned data: binned_' + mib_list[0].rpartition('.')[0] + '.hdf5')
                    del dp_bin
                
                print('Saving hdf5 : ' + mib_list[0].rpartition('.')[0] +'.hdf5')
                dp_data.save(saving_path+'/'+mib_list[0], extension = 'hdf5')
                print('Saved hdf5 : ' + mib_list[0].rpartition('.')[0] +'.hdf5')
                t3 = time.time()
                
                del dp
                gc.collect()
                        
        elif mib_num > 1:
            # to change!!!!
            # if (STEM_flag == 0 or STEM_flag == '0'):
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
                t1 = time.time()
                dp = mib_dask_reader('/' +mib_path + '/'+ file)
                pprint.pprint(dp[1], width=1)
                STEM_flag = dp[1].get('STEM_flag')
                scan_X = dp[1].get('scan_X')
                dp[0].compute() # is replacing needed?
                dp_data = dp[0]
                t1 = time.time()
                
                if (STEM_flag == 0 or STEM_flag == '0'):

                    dp_data = dp[0]
                    dp_sum = max_contrast8(dp_data.sum())
                    dp_sum.save(saving_path + '/' +file+'_sum', extension = 'jpg')
                    t2 = time.time()
                    dp_data.save(saving_path + '/' +file, extension = 'hdf5')
                    t3 = time.time()

                    
        if t1 is not None:
            print('time to load data: ', int(t1-t0))
        if t2 is not None:
            print('time to save last image: ', int(t2-t0)) 
        if t3 is not None:
            print('time to save full hdf5: ', int(t3-t0))  
                    
#%%
                    
def watch_convert(beamline, year, visit, folder):
    
    [to_convert, mib_files, mib_to_convert] = check_differences(beamline, year, visit, folder)
    #holder for raw data path
    if folder:
        raw_location = os.path.join('/dls',beamline,'data', year, visit, os.path.relpath(folder))
    else:
        raw_location = os.path.join('/dls',beamline,'data', year, visit, 'Merlin')
    if bool(to_convert):
        convert(beamline, year, visit, mib_to_convert, folder)
    else:
        #watch_check = input('Do you want to keep watching this folder? (Y/N)')
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
                    # print(added[-1])
                    new_data_folder = os.listdir (path = os.path.join(path_to_watch, added[-1]))
                    for f in new_data_folder:
                        if f.endswith('mib'):
                            wait_flag = 1
                            while wait_flag == 1:
                                try:
                                    print('file name: ', f)
                                    # f_size = os.stat(os.path.join(path_to_watch, added[-1], f)).st_size
                                    # f_size = os.path.getsize(os.path.join(path_to_watch, added[-1], f))
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
                    # print('here!')
                before = after

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
    #watch_convert(args.beamline, args.year, args.visit, args.folder, args.STEM_flag, args.scan_X)
