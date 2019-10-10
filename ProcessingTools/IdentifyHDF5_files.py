import os
import sys

def get_HDF5_files(beamline, year, visit, folder = None):
    # check processing folder for converted hdf5 and binned hdf5 files and 
    # dictionary containing files and paths
    # 
    # perhaps should cross reference with .mib files to get filenames of unbinned data....

    hdf5_dirs = []    
    hdf5_files = []
    binned_hdf5_files = []
    binned_dirs = []
    
    if folder:
        # check that the path folder exists
        proc_location = os.path.join('/dls',beamline,'data', year, visit,'processing', os.path.relpath(folder))
        print(proc_location)
        if not os.path.exists(proc_location):
            print('This folder ', proc_location,'does not exist!') 
            print('The expected format for folder is sample1/dataset1/')
            sys.exit()  # options here?
    else:
        #my local path 
        #proc_location = os.path.join('/dls/e02/data/',os.sep, year, visit,'processing', 'Merlin')
        #linux path
        proc_location = os.path.join('/dls', beamline,'data', year, visit, 'processing', 'Merlin')    
    #print(proc_location)
    #if not os.path.exists(proc_location):
    #     print('Cannot find',proc_location)
    #     sys.exit()


    #itterator for directorys and files
    path_walker = os.walk(proc_location)
    #itterate through directory, folders and files
    for p, d, files in path_walker:
        #find hdf5 files 
        for f in files:
            if f.endswith('hdf5'):
                #build binned list
                if f.startswith('binned'):
                    binned_hdf5_files.append(f)
                    binned_dirs.append(p)
                #build hdf5 list
                else:
                    hdf5_files.append(f)
                    hdf5_dirs.append(p)
    #build a dictionary of file names and paths
    hdf5_dict = {}
    hdf5_dict['processing_path'] = proc_location
    hdf5_dict['HDF5_files'] = hdf5_files
    hdf5_dict['HDF5_paths'] = hdf5_dirs
    hdf5_dict['binned_HDF5_files'] = binned_hdf5_files   
    hdf5_dict['binned_HDF5_paths'] = binned_dirs
    
    #return the dictionary
    return hdf5_dict

