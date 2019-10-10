import os
import argparse
import sys
#%%
#for testing 
beamline = 'e02'
year = '2019'
visit = 'mg22549-6'
folder = None
post_processing = True
#%%
def check_differences(beamline, year, visit, folder = None, post_processing = False):
    # print("Starting main function")
    # fist check to see which visit is active

    mib_files = []
    raw_dirs = []
    if folder:
        # check that the path folder exists
        raw_location = os.path.join('/dls',beamline,'data', year, visit, os.path.relpath(folder))
        print(raw_location)
        if not os.path.exists(raw_location):
            print('This folder ', raw_location,'does not exist!') 
            print('The expected format for folder is sample1/dataset1/')
            sys.exit()  # options here?
    else:
        #my local path 
        raw_location = os.path.join('y:',os.sep, year, visit, 'Merlin')
        #raw_location = os.path.join(os.sep,  'dls', beamline,'data', year, visit, 'Merlin')
    proc_location = os.path.join('y:',os.sep, year, visit,'processing', 'Merlin')
    #proc_location = os.path.join(os.sep,  'dls', beamline,'data', year, visit, 'processing', 'Merlin')
    #print(proc_location)
    if not os.path.exists(proc_location):
        os.mkdir(proc_location)

    
    #os.chdir(raw_location)
    raw_dirs = []    
    hdf5_files = []
    binned_hdf5_files = []
    binned_dirs = []
    if post_processing:
        #do post processing on already converted files
        #first check for text file with parameters in
        proc_path  = os.path.split(proc_location)[0]
        proc_file_path =  os.path.join(proc_path, 'processing_params.txt')
        if os.path.exists(proc_file_path) :
            proc_dict = {}
            #read value from text parameter file into dictionary
            with open(proc_file_path) as f:
                for line in f:
                    #ignore commented lines
                    if line.startswith('#'):
                        next
                    else:
                        (key, val) = line.split(sep = ':')
                        proc_dict[key] = float(val)
            
        
        else:
            #processing_params.txt doesn't exist  - jump out. 
            print('no processing_params.txt file in ', proc_path)
            exit()
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
                        raw_dirs.append(p)
            
        
    else:
        # look through all the files in that location and find any mib files
        path_walker = os.walk(raw_location)
        for p, d, files in path_walker:
            # look at the files and see if there are any mib files there
            for f in files:
                if f.endswith('mib'):
    #                if folder:
    #                    p = './'+ folder + p[1:]
                    #mib_files.append((p, f))
                    mib_files.append(os.path.join(str(p), str(f)))
                    #print(p)
                    #print(f)
                    raw_dirs.append(p)
        
        
        # print('RAW dirs:  ', raw_dirs)
        # print('MIB files: ', mib_files)
    
        # look in the processing folder and list all the directories
        converted_dirs = []
        
        hdf_files = []
        path_walker = os.walk(proc_location)
        for p, d, files in path_walker:
            # look at the files and see if there are any mib files there
            for f in files:
                if f.endswith('hdf5'):
                    if folder:
                        p = './'+ folder + p[1:]
                    hdf_files.append((p, f))
                    converted_dirs.append(p)
        
        # only using the time-stamp section of the paths to compare:
        raw_dirs_check = []
        converted_dirs_check = []
        for folder in raw_dirs:
            raw_dirs_check.append(folder.split('/')[-1]) 
        for folder in converted_dirs:
            converted_dirs_check.append(folder.split('/')[-1]) 
        # compare the directory lists, and see which has the
        converted = set(converted_dirs_check)
        to_convert = set(raw_dirs_check) - set(converted_dirs_check)
        
        mib_to_convert = []
        for mib_path in mib_files:
            if mib_path.split('/')[-2] in to_convert:
                mib_to_convert.append(mib_path)
                
    
        print('Converted Datasets: ', converted)
        print('To CONVERT:  ', to_convert)
        
        return to_convert, mib_files, mib_to_convert

#%%
def main(beamline, year, visit, folder = None):
    check_differences(beamline, year, visit, folder = None)

#%%
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('beamline', help='Beamline name')
    parser.add_argument('year', help='Year')
    parser.add_argument('visit', help='Session visit code')
    parser.add_argument('folder', nargs= '?', help='Option to add folder')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()
    
    main(args.beamline, args.year, args.visit, args.folder)