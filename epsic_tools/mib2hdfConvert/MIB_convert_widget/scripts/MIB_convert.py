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
        rot_angle = 90
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
    def __init__(self):
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


    def _verbose(self, path_verbose):
        if path_verbose:
            print(*self.to_convert, sep="\n")        


    def _organize(self, no_reshaping, use_fly_back, known_shape, 
                  Scan_X, Scan_Y, ADF_check, iBF_check, DPC_check,
                bin_nav_widget, bin_sig_widget, node_check,
                create_batch_check, create_info_check,
                  create_json, ptycho_config, ptycho_template):

        #self.python_script_path = '/dls/science/groups/e02/Ryu/epsic_code/MIB_convert/test/scripts/MIB_convert_submit.py'
        self.python_script_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/MIB_convert_submit.py'
        if create_batch_check:
            self.bash_script_path = os.path.join(self.script_save_path, 'cluster_submit.sh')
            self.info_path = os.path.join(self.script_save_path, 'convert_info.txt')
            
            with open (self.bash_script_path, 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition %s\n'%node_check)
                f.write('#SBATCH --job-name mib_convert\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node 1\n')
                f.write('#SBATCH --cpus-per-task 1\n')
                f.write('#SBATCH --time 05:00:00\n')
                f.write('#SBATCH --mem 100G\n\n')

                f.write(f"#SBATCH --array=0-{len(self.to_convert)-1}%3\n")
                f.write(f"#SBATCH --error={self.script_save_path}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={self.script_save_path}{os.sep}%j_output.out\n")

                f.write('echo "I am running the array job with task ID $SLURM_ARRAY_TASK_ID"\n')
                f.write('module load python/epsic3.10\n\n')              
                f.write('sleep 10\n')
                f.write(f"python {self.python_script_path} {self.info_path} $SLURM_ARRAY_TASK_ID\n")

            print("sbatch file created: "+self.bash_script_path)
            print("submission python file: "+self.python_script_path)

        if create_info_check:
            self.info_path = os.path.join(self.script_save_path, 'convert_info.txt')
            if iBF_check:
                iBF = 1
            else:
                iBF = 0

            if ADF_check:
                ADF = 1
            else:
                ADF = 0

            if DPC_check:
                DPC = 1
            else:
                DPC = 0
                
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

            if no_reshaping:
                reshape = 0
            else:
                reshape = 1

            if create_json:
                json = 1
            else:
                json = 0
            
            with open (self.info_path, 'w') as f:
                f.write(
                    f"to_convert_paths = {self.to_convert}\n"
                    f"no_reshaping = {no_reshaping}\n"
                    f"use_fly_back = {use_fly_back}\n"
                    f"known_shape = {known_shape}\n"
                    f"Scan_X = {Scan_X}\n"
                    f"Scan_Y = {Scan_Y}\n"
                    f"iBF = {iBF}\n"
                    f"ADF = {ADF}\n"
                    f"DPC = {DPC}\n"
                    f"bin_sig_flag = {bin_sig_flag}\n"
                    f"bin_sig_factor = {bin_sig_factor}\n"
                    f"bin_nav_flag = {bin_nav_flag}\n"
                    f"bin_nav_factor = {bin_nav_factor}\n"
                    f"reshape = {reshape}\n"
                    f"create_json = {json}\n"
                    f"ptycho_config = {ptycho_config}\n"
                    f"ptycho_template = {ptycho_template}\n"
                        )
                
            print("conversion info file created: "+self.info_path)

    def _ptycho(self, create_ptycho_folder, ptycho_config_name, ptycho_template_path):
        '''This part of the code is designed to automatic generate Ptyrex Json and associated folders, its uses os.walk 
           and the inputted year, session and subfolder varibles to find already converted data files, using the 
           converted data meta file and known dictionary of parameters it is possible fill out the json file
           frederick allars 09-05-2024'''
        if create_ptycho_folder:
            hdf_files = []
            '''use os.walk to find the date time when the data was collected which corresponds to its folder and file names within the subfolder'''
            for path, directories, files in os.walk(self.dest_path):
                for f in files:
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
                        print(meta_file)

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
                            template_path = '/dls/science/groups/imaging/ePSIC_ptychography/experimental_data/User_example/UserExampleJson.json'
                        else:
                            template_path = ptycho_template_path


                        gen_config(template_path, pty_dest_2, config_name, meta_file, rot_angle, camera_length, conv_angle)        

                
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
        
        path_verbose = Checkbox(value=False, description="Show the path of each MIB file", style=st)

        no_reshaping = Checkbox(value=False, description='No reshaping', style=st)
        use_fly_back = Checkbox(value=True, description='Use Fly-back', style=st)
        known_shape = Checkbox(value=False, description='Known_shape', style=st)
        Scan_X = IntText(description='Scan_X: (avaiable for Known_shape)', style=st)
        Scan_Y = IntText(description='Scan_Y: (avaiable for Known_shape)', style=st)

        ADF_check = Checkbox(value=False, description='ADF (Not available yet)', style=st)
        iBF_check = Checkbox(value=True, description='iBF', style=st)
        DPC_check = Checkbox(value=False, description='DPC (Not available yet)', style=st)

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
        create_json = Checkbox(value=False, description='Create a ptychography subfolder', style=st)
        ptycho_config = Text(description='Enter config name (optional) :', style=st)
        ptycho_template = Text(description='Enter template config path (optional) :', style=st)
        
        create_ptycho_folder = Checkbox(value=False, description='Create a ptychography subfolder', style=st)
        ptycho_config_name = Text(description='Enter config name (optional) :', style=st)
        ptycho_template_path = Text(description='Enter template config path (optional) :', style=st)

        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        
        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        create_info_check = Checkbox(value=False, description='Create conversion info file', style=st)
        submit_check = Checkbox(value=False, description='Submit a slurm job', style=st)

        self.path = ipywidgets.interact(self._paths, 
                                          year=year, 
                                          session=session,
                                          subfolder_check=subfolder_check,
                                          subfolder=subfolder)
        
        self.verbose = ipywidgets.interact(self._verbose, path_verbose=path_verbose)
        
        self.values = ipywidgets.interact(self._organize,
                                          no_reshaping=no_reshaping, 
                                        use_fly_back=use_fly_back, 
                                          known_shape=known_shape, 
                                          Scan_X=Scan_X, 
                                          Scan_Y=Scan_Y,
                                        ADF_check=ADF_check, 
                                          iBF_check=iBF_check, 
                                          DPC_check=DPC_check,
                                        bin_nav_widget=bin_nav_widget, 
                                          bin_sig_widget=bin_sig_widget,
                                          node_check=node_check,
                                        create_batch_check=create_batch_check, 
                                          create_info_check=create_info_check, 
                                          create_json=create_json, 
                                          ptycho_config=ptycho_config, 
                                          ptycho_template=ptycho_template)
        
        self.submit = ipywidgets.interact(self._submit, submit_check=submit_check)

        print("********************************************************************************")
        print("********************************************************************************")
        print("The widgets below are to generate PtyREX JSON files for the converted MIB files.")
        print("********************************************************************************")
        print("********************************************************************************")
        self.ptycho = ipywidgets.interact(self._ptycho, 
                                      create_ptycho_folder=create_ptycho_folder, 
                                      ptycho_config_name=ptycho_config_name, 
                                      ptycho_template_path=ptycho_template_path)

    
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
