#!/usr/bin/env bash
import hyperspy.api as hs
print(f"hyperspy version: {hs.__version__}")
import pyxem as pxm
print(f"pyxem version: {pxm.__version__}")
import numpy as np
import py4DSTEM
print(f"py4DSTEM version: {py4DSTEM.__version__}")
import dask.array as da
from rsciio.quantumdetector import load_mib_data, parse_exposures
from rsciio.quantumdetector import file_reader
import h5py
import shutil
import pandas as pd

import ipywidgets
from ipywidgets.widgets import *
import sys
import os
import glob
import logging
import subprocess

formatter = logging.Formatter("%(asctime)s    %(process)5d %(processName)-12s %(threadName)-12s                   %(levelname)-8s %(pathname)s:%(lineno)d %(message)s")
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set the debug log level.
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)

# Make a logger for this module.
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

def find_metadat_file(timestamp, acquisition_path):
    metadata_file_paths = []
    mib_file_paths = []
        
    for root, folders, files in os.walk(acquisition_path):
        for file in files:
            if file.endswith('hdf'):
                metadata_file_paths.append(os.path.join(root, file))
            elif file.endswith('mib'):
                mib_file_paths.append(os.path.join(root, file))
    for path in metadata_file_paths:
        if timestamp == path.split('/')[-1].split('.')[0]:
            return path
    logger.debug('No metadata file could be matched.')
    return 

# ptyrex import
import json

def gen_config(template_path, dest_path, config_name, meta_file_path, rotation_angle, camera_length, conv_angle):
    config_file = dest_path + '/' + config_name + '.json'

    with open(template_path, 'r') as template_file:
        pty_expt = json.load(template_file)
    data_path = meta_file_path

    pty_expt['base_dir'] = dest_path
    pty_expt['process']['save_dir'] = dest_path
    pty_expt['experiment']['data']['data_path'] = data_path

    pty_expt['process']['common']['scan']['rotation'] = rotation_angle

    # pty_expt['process']['common']['scan']['N'] = scan_shape
    pty_expt['experiment']['detector']['position'] = [0, 0, camera_length]
    pty_expt['experiment']['optics']['lens']['alpha'] = conv_angle

    with h5py.File(meta_file_path, 'r') as microscope_meta:
        meta_values = microscope_meta['metadata']
        pty_expt['process']['common']['scan']['N'] = [int(meta_values['4D_shape'][:2][0]),
                                                      int(meta_values['4D_shape'][:2][1])]
        pty_expt['process']['common']['source']['energy'] = [float(np.array(meta_values['ht_value(V)']))]
        pty_expt['process']['common']['scan']['dR'] = [float(np.array(meta_values['step_size(m)'])),
                                                       float(np.array(meta_values['step_size(m)']))]
        # pty_expt['experiment']['optics']['lens']['alpha'] = 2 * float(np.array(meta_values['convergence_semi-angle(rad)']))
        pty_expt['experiment']['optics']['lens']['defocus'] = [float(np.array(meta_values['defocus(nm)']) * 1e-9),
                                                               float(np.array(meta_values['defocus(nm)']) * 1e-9)]
        pty_expt['process']['save_prefix'] = config_name

    with open(config_file, 'w') as f:
        json.dump(pty_expt, f, indent=4)


def Meta2Config(acc,nCL,aps):
    '''This function converts the meta data from the 4DSTEM data set into parameters to be used in a ptyREX json file'''

    '''The rotation angles noted here are from ptychographic reconstructions which have been successful. see the 
    following directory for example reconstruction from which these values are derived:
     /dls/science/groups/imaging/ePSIC_ptychography/experimental_data'''
    if acc == 80e3:
        rot_angle = 238.5
        print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 41.65e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 31.74e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 24.80e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =15.44e-3
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    elif acc == 200e3:
        rot_angle = -77.585
        print('Rotation angle = ' + str(rot_angle) +' Warning: This rotation angle need further calibration')
        if aps == 1:
            conv_angle = 37.7e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 28.8e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 22.4e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle = 14.0
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 5:
            conv_angle = 6.4
            print('Condenser aperture size is 10um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
    elif acc == 300e3:
        rot_angle = -85.5
        print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 44.7e-3
            print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 34.1e-3
            print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 26.7e-3
            print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =16.7e-3
            print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    else:
        print('Rotation angle for this acceleration voltage is unknown, please collect calibration data. Rotation angle being set to zero')
        rot_angle = 0

    '''this is incorrect way of calucating the actual camera length but okay for this prototype code'''
    '''TODO: add py4DSTEM workflow which automatic determines the camera length from a small amount of reference data and the known convergence angle'''
    camera_length = 1.5*nCL
    print('camera length estimated to be ' + str(camera_length))

    return rot_angle,camera_length,conv_angle


# Widgets
class convert_info_widget():

    meta_keys = ['filename', 'A1_value_(kV)', 'A2_value_(kV)', 'aperture_size',
       'convergence_semi-angle(rad)', 'current_OLfine', 'deflector_values',
       'defocus(nm)', 'defocus_per_bit(nm)', 'field_of_view(m)', 'ht_value(V)',
       'lens_values', 'magnification', 'merlin_camera_length(m)',
       'nominal_camera_length(m)', 'nominal_scan_rotation', 'set_bit_depth',
       'set_dwell_time(usec)', 'set_scan_px', 'spot_size', 'step_size(m)',
       'x_pos(m)', 'x_tilt(deg)', 'y_pos(m)', 'y_tilt(deg)', 'z_pos(m)',
       'zero_OLfine']

    meta_keys_after = ['filename', '4D_shape', 'A1_value_(kV)', 'A2_value_(kV)',
       'aperture_size', 'convergence_semi-angle(rad)', 'current_OLfine',
       'deflector_values', 'defocus(nm)', 'defocus_per_bit(nm)',
       'field_of_view(m)', 'ht_value(V)', 'lens_values', 'magnification',
       'merlin_camera_length(m)', 'nominal_camera_length(m)',
       'nominal_scan_rotation', 'set_bit_depth', 'set_dwell_time(usec)',
       'set_scan_px', 'spot_size', 'step_size(m)', 'x_pos(m)', 'x_tilt(deg)',
       'y_pos(m)', 'y_tilt(deg)', 'z_pos(m)', 'zero_OLfine']
    
    def __init__(self, only_ptyrex=False, only_virtual=False, ptyrex_submit=False):
        if only_ptyrex:
            self._ptyrex_json()
        elif only_virtual:
            self._virtual_images()
        elif ptyrex_submit:
            self._ptyrex_submit()
        else:
            self._activate()

    def _paths(self, year, session, subfolder_check, subfolder):

        if subfolder == '' and subfolder_check:
            print("**************************************************")
            print("'subfolder' is not speicified")
            print("All MIB files in 'Merlin' folder will be converted")
            print("**************************************************")
        
        self.src_path = f'/dls/e02/data/{year}/{session}/Merlin/{subfolder}'
        print("source_path: ", self.src_path)
        if os.path.exists(self.src_path):
            mib_files = []
            for p, d, files in os.walk(self.src_path):
                # look at the files and see if there are any mib files there
                for f in files:
                    if f.endswith('mib'):
                        mib_files.append(os.path.join(str(p), str(f)))
            print('%d MIB files exist in the source directory'%len(mib_files))
            #print(*mib_files, sep="\n")
            src_path_flag = True
        else:
            print('Path specified does not exist!')
            src_path_flag = False

        self.dest_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/{subfolder}'
        print("destination_path: "+self.dest_path)
        if not os.path.exists(self.dest_path) and src_path_flag:
            os.makedirs(self.dest_path)
            print("created: "+self.dest_path)

        if subfolder == '':
            self.script_save_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/scripts'
        else:
            self.script_save_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/{subfolder}/scripts'
        print("script_save_path: "+self.script_save_path)
        if not os.path.exists(self.script_save_path) and src_path_flag:
            os.makedirs(self.script_save_path)
            print("created: "+self.script_save_path)
            
        self.to_convert = []
        if subfolder != '' or subfolder_check==True:
            self.to_convert = self._check_differences(self.src_path, self.dest_path)
            
        self.to_convert.sort()

    def _verbose(self, path_verbose):
        if path_verbose:
            meta_show = {}
            meta_show["filename"] = []
            for key in self.meta_keys:
                meta_show[key] = []

            for path in self.to_convert:
                src_path = path[:-40]
                time_stamp = path.split('/')[-1][:15]
                meta_path = find_metadat_file(time_stamp, src_path)
                
                with h5py.File(meta_path, 'r') as microscope_meta:
                    meta_show['filename'].append(path.split('/')[-1][:15])
                    for meta_key, meta_val in microscope_meta['metadata'].items():
                        try:
                            meta_show[meta_key].append(meta_val[()])
                        except:
                            meta_show[meta_key].append('NA')

            self.df = pd.DataFrame(meta_show)
            self.keys_show = self.df[["filename", 'ht_value(V)', 'aperture_size', "convergence_semi-angle(rad)",  'defocus(nm)', 'magnification', 'step_size(m)', 'nominal_camera_length(m)']]

            return self.keys_show
                
    def _organize(self, reshaping,Scan_X, Scan_Y,bin_nav_widget,bin_sig_widget,
                  node_check,n_jobs,create_virtual_image,mask_path,disk_lower_thresh,
                  disk_upper_thresh,DPC_check,parallax_check,
                  create_batch_check,create_info_check):
        
        self.python_script_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/MIB_convert_submit.py'
        
        self.bash_script_path = os.path.join(self.script_save_path, 'cluster_submit.sh')
        self.info_path = os.path.join(self.script_save_path, 'convert_info.txt')
        if create_batch_check:            
            with open (self.bash_script_path, 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition %s\n'%node_check)
                f.write('#SBATCH --job-name mib_convert\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node 1\n')
                f.write('#SBATCH --cpus-per-task 1\n')
                f.write('#SBATCH --time 05:00:00\n')
                if create_virtual_image:
                    f.write('#SBATCH --mem 192G\n\n')
                else:
                    f.write('#SBATCH --mem 64G\n\n')

                f.write(f"#SBATCH --array=0-{len(self.to_convert)-1}%{n_jobs}\n")
                f.write(f"#SBATCH --error={self.script_save_path}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={self.script_save_path}{os.sep}%j_output.out\n")

                f.write('echo "I am running the array job with task ID $SLURM_ARRAY_TASK_ID"\n')
                f.write('module load python/epsic3.10\n\n')  
                f.write('export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}\n')
                f.write('export BLOSC_NTHREADS=$((SLURM_CPUS_PER_TASK * 2))\n')                
                f.write('sleep 10\n')
                f.write(f"python {self.python_script_path} {self.info_path} $SLURM_ARRAY_TASK_ID\n")

            print("sbatch file created: "+self.bash_script_path)
            print("submission python file: "+self.python_script_path)

        if create_info_check:
            self.info_path = os.path.join(self.script_save_path, 'convert_info.txt')
                
            if bin_sig_widget != 1:
                bin_sig_flag = 1
                bin_sig_factor = bin_sig_widget
            else:
                bin_sig_flag = 0
                bin_sig_factor = bin_sig_widget

            if bin_nav_widget != 1:
                bin_nav_flag = 1
                bin_nav_factor = bin_nav_widget
            else:
                bin_nav_flag = 0
                bin_nav_factor = bin_nav_widget

            if reshaping == "Auto reshape":
                auto_reshape = True
                no_reshaping = False
                use_fly_back = False
                known_shape = False
            elif reshaping == "Flyback":
                auto_reshape = False
                no_reshaping = False
                use_fly_back = True
                known_shape = False
            elif reshaping == "Known shape":
                auto_reshape = False
                no_reshaping = False
                use_fly_back = False
                known_shape = True
            else:
                auto_reshape = False
                no_reshaping = True
                use_fly_back = False
                known_shape = False
                
            iBF = True
            
            if mask_path == '':
                mask_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/29042024_12bitmask.h5'
                
            with open (self.info_path, 'w') as f:
                f.write(
                    f"to_convert_paths = {self.to_convert}\n"
                    f"auto_reshape = {auto_reshape}\n"
                    f"no_reshaping = {no_reshaping}\n"
                    f"use_fly_back = {use_fly_back}\n"
                    f"known_shape = {known_shape}\n"
                    f"Scan_X = {Scan_X}\n"
                    f"Scan_Y = {Scan_Y}\n"
                    f"iBF = {iBF}\n"
                    f"bin_sig_flag = {bin_sig_flag}\n"
                    f"bin_sig_factor = {bin_sig_factor}\n"
                    f"bin_nav_flag = {bin_nav_flag}\n"
                    f"bin_nav_factor = {bin_nav_factor}\n"
                    f"reshaping = {reshaping}\n"
                    f"create_virtual_image = {create_virtual_image}\n"
                    f"mask_path = {mask_path}\n"
                    f"disk_lower_thresh = {disk_lower_thresh}\n"
                    f"disk_upper_thresh = {disk_upper_thresh}\n"
                    f"DPC = {DPC_check}\n"
                    f"parallax = {parallax_check}\n"
                        )
                
            print("conversion info file created: "+self.info_path)

    def _meta_show(self, meta_check, sort_key, search_key, search_value):
        if meta_check:
            meta_show = {}
            meta_show['filename'] = []
            for key in self.meta_keys_after:
                meta_show[key] = []  
                
            for path, directories, files in os.walk(self.dest_path):
                for f in files:
                    if f.endswith('_data.hdf5'):
                        folder_name = f.replace('_data.hdf5','')
                        meta_path = self.dest_path + '/' + folder_name + '/' + folder_name + '.hdf'
                        if not os.path.exists(meta_path):
                            meta_path = self.dest_path + '/' + folder_name + '/' + folder_name + '.hdf5'

                        with h5py.File(meta_path, 'r') as microscope_meta:
                            meta_show['filename'].append(path.split('/')[-1][:15])
                            for meta_key, meta_val in microscope_meta['metadata'].items():
                                try:
                                    meta_show[meta_key].append(meta_val[()])
                                except:
                                    meta_show[meta_key].append('NA')
        
            self.df_after = pd.DataFrame(meta_show)
            self.keys_show_after = self.df_after[["filename", '4D_shape', 'ht_value(V)', 'aperture_size', "convergence_semi-angle(rad)",  'defocus(nm)', 'magnification', 'step_size(m)', 'field_of_view(m)', 'nominal_camera_length(m)', 'set_bit_depth']]

            if sort_key != '':
                try:
                    return self.keys_show_after.sort_values(sort_key)
                except:
                    print("Wrong metadata key!")

            elif search_key != '':
                try:
                    return self.keys_show_after.loc[self.keys_show_after[search_key] == eval(search_value)]
                except:
                    print("Wrong metadata key! or empty search value!")

            else:
                return self.keys_show_after
                

    def _ptycho(self, create_ptycho_folder, ptycho_config_name, ptycho_template_path):
        '''This part of the code is designed to automatic generate Ptyrex Json and associated folders, its uses os.walk 
           and the inputted year, session and subfolder varibles to find already converted data files, using the 
           converted data meta file and known dictionary of parameters it is possible fill out the json file
           frederick allars 09-05-2024'''
        if create_ptycho_folder:
            files_tmp = []
            '''use os.walk to find the date time when the data was collected which corresponds to its folder and file names within the subfolder'''
            for path, directories, files in os.walk(self.dest_path):
                if files != []:
                    files_tmp.extend(files)
            files_tmp.sort()
            for f in files_tmp:
                if f.endswith('_data.hdf5'):
                    folder_name = f.replace('_data.hdf5','')

                    '''create folders with standard names and skip if they already exist otherwise an error is incurred'''
                    pty_dest = self.dest_path + '/' + folder_name + '/' + 'pty_out'
                    pty_dest_2 = self.dest_path + '/' + folder_name + '/' + 'pty_out/initial_recon'
                    print(pty_dest)
                    try:
                        os.makedirs(pty_dest)
                    except:
                        print('skipping this folder as it already has pty_out folder')
                    try:
                        os.makedirs(pty_dest_2)
                    except:
                        print('skipping this folder as it already has pty_out/initial folder')

                    '''now the objective to get all the data required to fill the Json, we use the folder name to 
                    create the path to the meta data file'''
                    meta_file = self.dest_path + '/' + folder_name + '/' + folder_name + '.hdf'

                    '''we can now open the meta data file itself to check the energy which will give us the rotation angle,
                     the size of the aperture which will tell us the convergence angle, and the camera length which 
                     we can guess from the nomial camera length with approximate k factor in this case 1.5'''

                    '''TODO add py4DSTEM code which automatic guess the camera length from the subset of the collected diffraction patterns'''
                    with h5py.File(meta_file, 'r') as microscope_meta:
                        meta_values = microscope_meta['metadata']
                        print(meta_values['aperture_size'][()])
                        print(meta_values['nominal_camera_length(m)'][()])
                        print(meta_values['ht_value(V)'][()])
                        acc = meta_values['ht_value(V)'][()]
                        nCL = meta_values['nominal_camera_length(m)'][()]
                        aps = meta_values['aperture_size'][()]
                    rot_angle,camera_length,conv_angle = Meta2Config(acc, nCL, aps)

                    '''check that the config_name parameter has been filled if not give it a default name'''
                    if ptycho_config_name == '':
                        config_name = 'pty_recon'
                    else:
                        config_name = ptycho_config_name

                    '''TODO: set up some standard ptyREX config files to reference at different energies'''
                    if ptycho_template_path == '':
                        template_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/UserExampleJson.json'
                    else:
                        template_path = ptycho_template_path


                    gen_config(template_path, pty_dest_2, config_name, meta_file, rot_angle, camera_length, 2*conv_angle)
        
      
    def _submit(self, submit_check):
        if submit_check:
            sshProcess = subprocess.Popen(['ssh',
                               '-tt',
                               'wilson'],
                               stdin=subprocess.PIPE, 
                               stdout = subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=0)
            sshProcess.stdin.write("ls .\n")
            sshProcess.stdin.write("echo END\n")
            sshProcess.stdin.write(f"sbatch {self.bash_script_path}\n")
            sshProcess.stdin.write("uptime\n")
            sshProcess.stdin.write("logout\n")
            sshProcess.stdin.close()
            
            
            for line in sshProcess.stdout:
                if line == "END\n":
                    break
                print(line,end="")
            
            #to catch the lines up to logout
            for line in  sshProcess.stdout: 
                print(line,end="")
        
    
    def _activate(self):
        print('*********************************************************************************')
        print('Make sure that <Submit checkbox> is unchecked before changing any other variables')
        print('*********************************************************************************')
        
        st = {"description_width": "initial"}
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        
        subfolder_check = Checkbox(value=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)
        
        path_verbose = Checkbox(value=False, description="Show the metadata of each MIB file", style=st)

        
        reshaping = Select(options=['Auto reshape', 'Flyback', 'Known shape', 'No reshaping'],
                            value='Auto reshape',
                            rows=4,
                            description='Choose a reshaping option',
                            disabled=False, style=st)
        
        Scan_X = IntText(description='Scan_X: (avaiable for Known_shape)', style=st)
        Scan_Y = IntText(description='Scan_Y: (avaiable for Known_shape)', style=st)

        bin_nav_widget = IntSlider(
                                value=2,
                                min=1,
                                max=8,
                                step=1,
                                description='Bin_nav:',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='d', style=st
                                        )

        bin_sig_widget = IntSlider(
                                value=2,
                                min=1,
                                max=8,
                                step=1,
                                description='Bin_sig:',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='d', style=st
                                        )

        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        create_info_check = Checkbox(value=False, description='Create conversion info file', style=st)
        submit_check = Checkbox(value=False, description='Submit a slurm job', style=st)

        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        n_jobs = IntSlider(value=3, min=1, max=12, step=1,
                            description='Number of multiple slurm jobs:',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            readout_format='d', style=st)
        
        create_virtual_image = Checkbox(value=False, description='Create virtual images', style=st)
        disk_lower_thresh = FloatText(description='Lower threshold value to detect the disk', value=0.01, style=st)
        disk_upper_thresh = FloatText(description='Upper threshold value to detect the disk', value=0.15, style=st)
        mask_path = Text(description='Enter the mask file path (optional) :', style=st)
        DPC_check = Checkbox(value=False, description='DPC', style=st)
        parallax_check = Checkbox(value=False, description='Parallax', style=st)

        self.path = ipywidgets.interact(self._paths, 
                                          year=year, 
                                          session=session,
                                          subfolder_check=subfolder_check,
                                          subfolder=subfolder)
        
        self.verbose = ipywidgets.interact(self._verbose, path_verbose=path_verbose)
        
        self.values = ipywidgets.interact(self._organize,
                                          reshaping=reshaping,
                                          Scan_X=Scan_X, 
                                          Scan_Y=Scan_Y,
                                        bin_nav_widget=bin_nav_widget, 
                                          bin_sig_widget=bin_sig_widget,
                                          node_check=node_check,
                                          n_jobs=n_jobs,
                                        create_virtual_image=create_virtual_image,
                                         mask_path=mask_path,
                                          disk_lower_thresh=disk_lower_thresh,
                                          disk_upper_thresh=disk_upper_thresh,
                                         DPC_check=DPC_check,
                                         parallax_check=parallax_check,
                                        create_batch_check=create_batch_check, 
                                          create_info_check=create_info_check)
        
        self.submit = ipywidgets.interact(self._submit, submit_check=submit_check)


    def _ptyrex_json(self):
        st = {"description_width": "initial"}
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder_check = Checkbox(value=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)

        meta_check = Checkbox(value=False, description="Show the metadata of converted data", style=st)
        sort_key = Text(description='Metadata key to sort:', style=st)
        search_key = Text(description='Metadata key to search:', style=st)
        search_value = Text(description='Value to search:', style=st)
        
        create_ptycho_folder = Checkbox(value=False, description='Create a ptychography subfolder', style=st)
        ptycho_config_name = Text(description='Enter config name (optional) :', style=st)
        ptycho_template_path = Text(description='Enter template config path (optional) :', style=st)

        self.path = ipywidgets.interact(self._paths, 
                                          year=year, 
                                          session=session,
                                          subfolder_check=subfolder_check,
                                          subfolder=subfolder)

        self.meta_show = ipywidgets.interact(self._meta_show,
                                            meta_check=meta_check,
                                            sort_key=sort_key,
                                            search_key=search_key,
                                            search_value=search_value)
                
        self.ptycho = ipywidgets.interact(self._ptycho, 
                                      create_ptycho_folder=create_ptycho_folder, 
                                      ptycho_config_name=ptycho_config_name, 
                                      ptycho_template_path=ptycho_template_path)
        
    def _virtual_images(self):
        st = {"description_width": "initial"}
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder_check = Checkbox(value=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)

        meta_check = Checkbox(value=False, description="Show the metadata of converted data", style=st)
        sort_key = Text(description='Metadata key to sort:', style=st)
        search_key = Text(description='Metadata key to search:', style=st)
        search_value = Text(description='Value to search:', style=st)
        
        create_virtual_image = Checkbox(value=False, description='Create virtual images', style=st)
        disk_lower_thresh = FloatText(description='Lower threshold value to detect the disk', value=0.01, style=st)
        disk_upper_thresh = FloatText(description='Upper threshold value to detect the disk', value=0.15, style=st)
        mask_path = Text(description='Enter the mask file path (optional) :', style=st)
        DPC_check = Checkbox(value=False, description='DPC', style=st)
        dpc_lpass = FloatText(description='DPC low pass', value=0.00, style=st)
        dpc_hpass = FloatText(description='DPC high pass', value=0.00, style=st)     
        parallax_check = Checkbox(value=False, description='Parallax', style=st)
        
        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        
        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        create_info_check = Checkbox(value=False, description='Create conversion info file', style=st)
        submit_check = Checkbox(value=False, description='Submit a slurm job', style=st)

        
        self.path = ipywidgets.interact(self._paths, 
                                          year=year, 
                                          session=session,
                                          subfolder_check=subfolder_check,
                                          subfolder=subfolder)

        self.meta_show = ipywidgets.interact(self._meta_show,
                                            meta_check=meta_check,
                                            sort_key=sort_key,
                                            search_key=search_key,
                                            search_value=search_value)
        
        self.virtual_values = ipywidgets.interact(self._virtual,
                                                     mask_path=mask_path,
                                                      disk_lower_thresh=disk_lower_thresh,
                                                      disk_upper_thresh=disk_upper_thresh,
                                                     DPC_check=DPC_check,
                                                     dpc_lpass=dpc_lpass,
                                                     dpc_hpass=dpc_hpass,
                                                     parallax_check=parallax_check,
                                                     node_check=node_check,
                                                     create_batch_check=create_batch_check,
                                                     create_info_check=create_info_check,
                                                     submit_check=submit_check)
        

        
    def _virtual(self, mask_path, disk_lower_thresh,
                        disk_upper_thresh, DPC_check,
                        dpc_lpass, dpc_hpass, parallax_check,
                        create_batch_check, node_check,
                        create_info_check, submit_check):
        
        converted_files = []
        for path, directories, files in os.walk(self.dest_path):
            for f in files:
                if f.endswith('_data.hdf5'):
                    folder_name = f.replace('_data.hdf5','')
                    converted_path = self.dest_path + '/' + folder_name + '/' + f
                    converted_files.append(converted_path)
        
        python_script_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/py4DSTEM_virtual_image.py'
        bash_script_path = os.path.join(self.script_save_path, 'virtual_submit.sh')
        info_path = os.path.join(self.script_save_path, 'py4DSTEM_info.txt')
        
        if create_batch_check:            
            with open (bash_script_path, 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition %s\n'%node_check)
                f.write('#SBATCH --job-name mib_convert\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node 1\n')
                f.write('#SBATCH --cpus-per-task 1\n')
                f.write('#SBATCH --time 05:00:00\n')
                f.write('#SBATCH --mem 192G\n\n')

                f.write(f"#SBATCH --array=0-{len(converted_files)-1}%3\n")
                f.write(f"#SBATCH --error={self.script_save_path}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={self.script_save_path}{os.sep}%j_output.out\n")

                f.write('module load python/epsic3.10\n\n')            
                f.write(f"python {python_script_path} {info_path} $SLURM_ARRAY_TASK_ID\n")

            print("sbatch file created: "+bash_script_path)
            print("submission python file: "+python_script_path)        

        if mask_path == '':
            mask_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/29042024_12bitmask.h5'
            
        if create_info_check:
            with open (info_path, 'w') as f:
                f.write(
                    f"to_convert_paths = {converted_files}\n"
                    f"mask_path = {mask_path}\n"
                    f"disk_lower_thresh = {disk_lower_thresh}\n"
                    f"disk_upper_thresh = {disk_upper_thresh}\n"
                    f"DPC = {DPC_check}\n"
                    f"dpc_lpass = {dpc_lpass}\n"
                    f"dpc_hpass = {dpc_hpass}\n"
                    f"parallax = {parallax_check}\n"
                    f"device = cpu\n"
                        )

            print("conversion info file created: "+info_path)    
            
        if submit_check:
            sshProcess = subprocess.Popen(['ssh',
                               '-tt',
                               'wilson'],
                               stdin=subprocess.PIPE, 
                               stdout = subprocess.PIPE,
                               universal_newlines=True,
                               bufsize=0)
            sshProcess.stdin.write("ls .\n")
            sshProcess.stdin.write("echo END\n")
            sshProcess.stdin.write(f"sbatch {bash_script_path}\n")
            sshProcess.stdin.write("uptime\n")
            sshProcess.stdin.write("logout\n")
            sshProcess.stdin.close()
            
            
            for line in sshProcess.stdout:
                if line == "END\n":
                    break
                print(line,end="")
            
            #to catch the lines up to logout
            for line in  sshProcess.stdout: 
                print(line,end="")

    def _check_differences(self, source_path, destination_path):
        """Checks for .mib files associated with a specified session that have
        not yet been converted to .hdf5.
    
        Parameters
        ----------
    
    
        Returns
        -------
        a dictionary with the following keys:
        to_convert_folder : list
            List of directories that is the difference between those converted
            and those to be converted. NB: These are just the folder names and
            not complete paths, e.g. '20190830 112739'.
        mib_paths : list
            List of ALL the mib files including the entire paths that are found
            in the experimental session ,e.g. '/dls/e02/data/2019/em20198-8/Merlin
            /Merlin/Calibrations/AuXgrating/20190830 111907/
            AuX_100kx_10umAp_20cmCL_3p55A2.mib'
        mib_to_convert : list
            List of unconverted mib files including the complete path
        """
        print("*****************************************************")
        print("Duplicate check begins")
        print("If there are many files in the destination directory,")
        print("it will take a bit long time")
        print("*****************************************************")
        mib_paths = []
        raw_dirs = []
        for p, d, files in os.walk(source_path):
            # look at the files and see if there are any mib files there
            for f in files:
                if f.endswith('mib'):
                    mib_paths.append(os.path.join(str(p), str(f)))
        # look in the processing folder and list all the directories
        print("Total MIB files found: %d"%len(mib_paths))
        converted_dirs = []
    
        hdf_files = []
        for p, d, files in os.walk(destination_path):
            # look at the files and see if there are any mib files there
            for f in files:
                if f.endswith('_data.hdf5'):
                    #if folder:
                    #    p = './'+ folder + p[1:]
                    hdf_files.append(f)
        # only using the time-stamp section of the paths to compare:
        print("Total converted files found: %d"%len(hdf_files))
        
        raw_dirs_check = []
        converted_dirs_check = []
    
        for folder in mib_paths:
            raw_dirs_check.append(os.path.basename(folder).split('.')[0])
        for f in hdf_files:
            converted_dirs_check.append(f.split('.')[0])
            
        # compare the directory lists, and see which have not been converted.
        to_convert_folder = set(raw_dirs_check) - set(converted_dirs_check)
        print("Duplicate check finished")
        print("%d MIB files will be converted:"%len(to_convert_folder))
        mib_to_convert = []
        for mib_path in mib_paths:
            if os.path.basename(mib_path).split('.')[0] in to_convert_folder:
                mib_to_convert.append(mib_path)
    
        return mib_to_convert
    
    def _ptyrex_paths(self, year, session, subfolder_check, subfolder):
        self.json_files = []
        if subfolder == '':
            self.script_save_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/scripts'
        else:
            self.script_save_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/{subfolder}/scripts'

        if subfolder == '' and subfolder_check:
            print("**************************************************")
            print("'subfolder' is not speicified")
            print("All MIB files in 'Merlin' folder will be converted")
            print("**************************************************")
        
        self.json_sub_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/{subfolder}'
        print("source_path: ", self.json_sub_path)
        if os.path.exists(self.json_sub_path):
            test_string = 'autoptycho_is_done.txt'
            cur_string_length = 1000   
            for p, d, files in os.walk(self.json_sub_path):
                # look at the files and see if there are any json files there
                for f in files:
                    if f.endswith('json'):
                        tmp_string = os.path.join(p, f)
                        #print(tmp_string)
                        tmp_string_length = len(tmp_string)
                        if tmp_string_length <= cur_string_length:
                            cur_string_length = tmp_string_length
                            #print(os.path.isfile(os.path.join(str(p), test_string)))
                            if os.path.isfile(os.path.join(str(p), test_string)) == False:
                                self.json_files.append(os.path.join(str(p), str(f)))
            print(f'\n{self.json_files}\n')
            self._create_ptyrex_bash_submit(session)#,create_batch_check)
            #self._ptyrex_ssh_submit()
        else:
            print('Path specified does not exist!')
            src_path_flag = False


    def _create_ptyrex_bash_submit(self,session):#,create_batch_check):
        self.bash_ptyrex_path = []
        print(f'\n{self.json_files}\n')
        print(len(self.json_files))
        counter = 0
        for x in self.json_files:
            print(f'\n{x}\n')
            index = x.find('/pty_out')
            tmp_string = x[index-6:index]
            self.bash_ptyrex_path.append(os.path.join(self.script_save_path, f'{tmp_string}_ptyrex_submit.sh'))
            print(self.bash_ptyrex_path[counter])
            if 1:            
                with open (self.bash_ptyrex_path[counter], 'w') as f:
                    f.write('#!/usr/bin/env bash\n')
                    f.write('#SBATCH --partition=cs05r\n')
                    f.write('#SBATCH --job-name=ptyrex_recon\n')
                    f.write('#SBATCH --nodes 1\n')
                    f.write('#SBATCH --tasks-per-node=4\n')
                    f.write('#SBATCH --cpus-per-task 1\n')
                    f.write('#SBATCH --gpus-per-node=4\n')
                    f.write('#SBATCH --time 00:30:00\n')
                    f.write('#SBATCH --mem 0G\n\n')
                    f.write('#SBATCH --constraint=NVIDIA_Pascal\n')
                    
                    f.write(f"#SBATCH --error={self.script_save_path}{os.sep}%j_error.err\n")
                    f.write(f"#SBATCH --output={self.script_save_path}{os.sep}%j_output.out\n")

                    f.write(f"cd /home/ejr78941/ptyrex_temp_5/PtyREX")
                     
                    f.write('\n\nmodule load python/cuda11.7\n\n')  
                    f.write('module load hdf5-plugin/1.12\n\n')  
                    
                    f.write(f"mpirun -np 4 ptyrex_recon -c {self.json_files[counter]}\n")
                
            print("sbatch file created: "+ self.bash_ptyrex_path[counter])
            counter = counter + 1
    
    def _create_flagging_text_files(self,submit_ptyrex_job):
        if submit_ptyrex_job:    
            #assuming the paths given are the json_paths
            test_string = 'autoptycho_is_done.txt'
            default_string = 'this data set has been auto reconstructed already and therefore will skipped in all proceeding auto recons, please detele to restore auto functionality'
	    #flagging_files = []
            for x in self.json_files:
                a = x.find('initial_recon/') + len('initial_recon/')
                flagging_files = (x[:a]+test_string)
                with open (flagging_files, 'w') as f:
            	    f.write(default_string)
                f.close()
            print(flagging_files)
     
    def _ptyrex_ssh_submit(self, submit_ptyrex_job):
        counter = 0
        if submit_ptyrex_job:
            for x in self.bash_ptyrex_path:
                sshProcess = subprocess.Popen(['ssh',
                                   '-tt',
                                   'wilson'],
                                   stdin=subprocess.PIPE, 
                                   stdout = subprocess.PIPE,
                                   universal_newlines=True,
                                   bufsize=0)
                sshProcess.stdin.write("ls .\n")
                sshProcess.stdin.write("echo END\n")
                sshProcess.stdin.write(f"sbatch {x}\n")
                sshProcess.stdin.write("uptime\n")
                sshProcess.stdin.write("logout\n")
                sshProcess.stdin.close()
            
            
                for line in sshProcess.stdout:
                    if line == "END\n":
                        break
                    print(line,end="")
            
                #to catch the lines up to logout
                for line in  sshProcess.stdout: 
                    print(line,end="")
        self._create_flagging_text_files(submit_ptyrex_job)            

    def _ptyrex_submit(self):
        st = {"description_width": "initial"}
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder_check = Checkbox(value=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)

        create_batch_check = Checkbox(value=False, description="Create ptyrex submission script", style=st)
        sort_key = Text(description='Metadata key to sort:', style=st)
        search_key = Text(description='Metadata key to search:', style=st)
        search_value = Text(description='Value to search:', style=st)

        submit_ptyrex_job = Checkbox(value=False, description="Submit ptyrex jobs", style=st)

        self.ptyrex_paths = ipywidgets.interact(self._ptyrex_paths, 
                                          year=year, 
                                          session=session,
                                          subfolder_check=subfolder_check,
                                          subfolder=subfolder)

        #self.create_ptyrex_bash_submit = ipywidgets.interact(self._create_ptyrex_bash_submit,
        #                                    session=session,create_batch_check=create_batch_check)

        self.ptyrex_ssh_submit = ipywidgets.interact(self._ptyrex_ssh_submit,
                                            session=session,submit_ptyrex_job=submit_ptyrex_job)