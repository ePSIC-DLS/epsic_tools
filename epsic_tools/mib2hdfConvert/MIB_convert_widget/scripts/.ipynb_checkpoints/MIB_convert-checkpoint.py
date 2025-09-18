#!/usr/bin/env bash
import sys
import pprint
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
import datetime

import ipywidgets
from ipywidgets.widgets import *
import os
import glob
import logging
import subprocess
import nbformat
import yaml
import json
import re

import traceback


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

def gen_config(template_path, dest_path, config_name, meta_file_path, factor=1.7, overwrite = False, verbose=False):
    '''
    A function to generate json file for PtyREX recontrsution for data collected on E02 using the meta data .hdf5
    files in the /dls/e02/data directory. Is one of many functions designed allow automated ptyrex reconstruction
    via notebook see _ptycho, Meta2Config.
    :param template_path: path of the prexisting json which is being copied and being used as template
    :param dest_path: the destination of the outputted json file
    :param config_name: name of the json file being outputted
    :param meta_file_path: hdf5 meta data file which cointains the information needed to fill out the json file
    :param factor: a scaling factor used for manualy setting the camera length
    :param overwrite: if true overwrites any existing json file which exist in the pty_out/inital folder with
    the same config_name
    :param verbose: If true prints out most of the text, while if false gives minial printout
    :return: None
    frederick allars 18-08-2025
    '''


    config_file = dest_path + '/' + config_name + '.json'

    with open(template_path, 'r') as template_file:
        pty_expt = json.load(template_file)
    data_path = meta_file_path

    pty_expt['base_dir'] = dest_path
    pty_expt['process']['save_dir'] = dest_path
    pty_expt['experiment']['data']['data_path'] = data_path

    with h5py.File(meta_file_path, 'r') as microscope_meta:
        meta_values = microscope_meta['metadata']
        acc = meta_values['ht_value(V)'][()]
        nCL = meta_values['nominal_camera_length(m)'][()]
        aps = meta_values['aperture_size'][()]

        '''detemine the rotation angle and camera length from the HT value (accerlation voltage) and nomial camera length,
        also determine the convergence angle from the aperture size used'''
        rotation_angle, camera_length, conv_angle = Meta2Config(acc, nCL, aps, factor, verbose)

        pty_expt['process']['common']['scan']['rotation'] = rotation_angle
        pty_expt['experiment']['detector']['position'] = [0, 0, camera_length]
        pty_expt['experiment']['optics']['lens']['alpha'] = conv_angle * 2
        pty_expt['process']['common']['scan']['N'] = [int(meta_values['4D_shape'][:2][0]),
                                                          int(meta_values['4D_shape'][:2][1])]
        pty_expt['process']['common']['source']['energy'] = [float(np.array(meta_values['ht_value(V)']))]
        pty_expt['process']['common']['scan']['dR'] = [float(np.array(meta_values['step_size(m)'])),
                                                           float(np.array(meta_values['step_size(m)']))]
        # pty_expt['experiment']['optics']['lens']['alpha'] = 2 * float(np.array(meta_values['convergence_semi-angle(rad)']))
        pty_expt['experiment']['optics']['lens']['defocus'] = [float(np.array(meta_values['defocus(nm)']) * 1e-9),
                                                                   float(np.array(meta_values['defocus(nm)']) * 1e-9)]
        pty_expt['process']['save_prefix'] = config_name

    if os.path.exists(config_file) & ~overwrite:
        if verbose:
            print('\nskipping this file as json file already exists, instead edit the exitsing json file or name it'
                  ' something different\n')
    else:
        if overwrite & verbose:
            print('\noverwriting existing File...\n')
        with open(config_file, 'w') as f:
            json.dump(pty_expt, f, indent=4)



def Meta2Config(acc,nCL,aps,factor=1.7,verbose=False):


    '''This function converts the meta data from the 4DSTEM data set into parameters to be used in a ptyREX json file'''

    '''The rotation angles noted here are from ptychographic reconstructions which have been successful. see the 
    following directory for example reconstruction from which these values are derived:
     /dls/science/groups/imaging/ePSIC_ptychography/experimental_data'''
    if acc == 80e3:
        rot_angle = 238.5
        if verbose:
            print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 41.65e-3
            if verbose:
                print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 31.74e-3
            if verbose:
                print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 24.80e-3
            if verbose:
                print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =15.44e-3
            if verbose:
                print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    elif acc == 200e3:
        rot_angle = -77.585
        if verbose:
            print('Rotation angle = ' + str(rot_angle) +' Warning: This rotation angle need further calibration')
        if aps == 1:
            conv_angle = 37.7e-3
            if verbose:
                print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 28.8e-3
            if verbose:
                print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 22.4e-3
            if verbose:
                print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle = 14.0
            if verbose:
                print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 5:
            conv_angle = 6.4
            if verbose:
                print('Condenser aperture size is 10um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
    elif acc == 300e3:
        rot_angle = -85.5
        if verbose:
            print('Rotation angle = ' + str(rot_angle))
        if aps == 1:
            conv_angle = 44.7e-3
            if verbose:
                print('Condenser aperture size is 50um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 2:
            conv_angle = 34.1e-3
            if verbose:
                print('Condenser aperture size is 40um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 3:
            conv_angle = 26.7e-3
            if verbose:
                print('Condenser aperture size is 30um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        elif aps == 4:
            conv_angle =16.7e-3
            if verbose:
                print('Condenser aperture size is 20um has corresponding convergence semi angle of ' + str(conv_angle * 1e3) + 'mrad')
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    else:
        print('Rotation angle for this acceleration voltage is unknown, please collect calibration data. Rotation angle being set to zero')
        rot_angle = 0

    '''this is incorrect way of calucating the actual camera length but okay for this prototype code'''
    '''TODO: add py4DSTEM workflow which automatic determines the camera length from a small amount of reference data and the known convergence angle'''
    camera_length = factor*nCL
    if verbose:
        print('camera length estimated to be ' + str(camera_length))

    return rot_angle,camera_length,conv_angle

def _create_ptyrex_bash_submit(json_files, script_folser, node_type, ptycho_time, verbose=False):
    bash_ptyrex_path = []

    for num, x in enumerate(json_files):
        if verbose:
            print(f'\n{str(num)}: {x}\n')
        time_stamp_index = x.find('/pty_out')
        time_stamp = x[time_stamp_index - 6:time_stamp_index]
        if verbose:
            print(f'\ntime_stamp = {time_stamp}\n')
        '''rememeber to self.script_save_path to save these files to the script folder'''
        bash_ptyrex_path.append(os.path.join(script_folser, f'{time_stamp}_ptyrex_submit.sh'))
        if verbose:
            print(f'{num}: {bash_ptyrex_path[num]}')
        if True:
            with open(bash_ptyrex_path[num], 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition=cs05r\n')
                f.write('#SBATCH --job-name=ptyrex_recon\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node=4\n')
                f.write('#SBATCH --cpus-per-task 1\n')
                f.write('#SBATCH --gpus-per-node=4\n')
                f.write(f'#SBATCH --time {ptycho_time}\n')
                f.write('#SBATCH --mem 0G\n\n')
                f.write(f'#SBATCH --constraint=NVIDIA_{node_type}\n')

                f.write(f"#SBATCH --error={script_folser}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={script_folser}{os.sep}%j_output.out\n")

                f.write(f"cd /home/ejr78941/ptyrex_temp_5/PtyREX")

                f.write('\n\nmodule load python/cuda11.7\n\n')
                f.write('module load hdf5-plugin/1.12\n\n')

                f.write(f"mpirun -np 4 ptyrex_recon -c {json_files[num]}\n")
    return bash_ptyrex_path

def _create_flagging_text_files(submit_ptyrex_job, tmp_list, verbose=False):
    '''
    :param submit_ptyrex_job: a test whether ptyrex jobs should be submitted, used here as addiotnional test
    to check whether this function should run or not
    :param tmp_list: a list of json files, this used to determine the directories where should auto ptycho flagging
    should be generated
    :param verbose: this is used to determine whether full or minial printout should be used
    :return:
    '''
    if submit_ptyrex_job:
        # assuming the paths given are the json_paths
        test_string = 'autoptycho_is_done.txt'
        default_string = 'this data set has been auto reconstructed already and therefore will skipped in all proceeding auto recons, please detele to restore auto functionality'
        for x in tmp_list:
            a = x.find('initial_recon/') + len('initial_recon/')
            flagging_files = (x[:a] + test_string)
            if verbose:
                print(f'\nflag file name: {flagging_files}\n')
            with open(flagging_files, 'w') as f:
                f.write(default_string)
            f.close()

def _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, tmp_list, verbose=False):
    '''
    :param bash_list:
    :param submit_ptyrex_job:
    :param tmp_list:
    :param verbose:
    :return:
    '''
    if submit_ptyrex_job:
        sshProcess = subprocess.Popen(['ssh',
                                       '-tt',
                                       'wilson'],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      universal_newlines=True,
                                      bufsize=0)
        sshProcess.stdin.write("ls .\n")
        sshProcess.stdin.write("echo END\n")
        for num, x in enumerate(bash_list):
            sshProcess.stdin.write(f"sbatch {x}\n")
        sshProcess.stdin.write("uptime\n")
        sshProcess.stdin.write("logout\n")
        sshProcess.stdin.close()

        for line in sshProcess.stdout:
            if line == "END\n":
                break
            print(line, end="")

        # to catch the lines up to logout
        for line in sshProcess.stdout:
            print(line, end="")
    _create_flagging_text_files(submit_ptyrex_job, tmp_list, verbose)

# ----------------------------------------------------------------------------------------
class NotebookHelper:
    """
    This workflow takes a notebook. 
    """

    def __init__(self, notebook_paths, notebook_name):
        self.__notebook_paths = notebook_paths
        self.__notebook_name = notebook_name

    # ------------------------------------------------------------------
    # Method to get settings from notebook cell.
    def get_settings(self, cell_index):
        """
        Get settings from a notebook cell. 
        """

        ipynb_filename = os.path.join(self.__notebook_paths , f"{self.__notebook_name}.ipynb")


        # Read the notebook into memory.
        with open(ipynb_filename, "r") as stream:
            notebook = nbformat.read(stream, as_version=4)

        source = notebook["cells"][cell_index]["source"].strip()

        if len(source) == 0:
            raise RuntimeError(
                f"notebook {self.__notebook_name} cell {cell_index} is blank"
            )

        # Replace some markdown things that might be in there.
        source = re.sub(r"^(```yaml)(\n)?", "", source)
        source = re.sub(r"^(```json)(\n)?", "", source)
        source = re.sub(r"^(```)(\n)?", "", source)
        source = re.sub(r"(\n)?(```)$", "", source)

        if source[0] == "{":
            try:
                settings_dicts = json.loads(source)
            except Exception:
                raise RuntimeError(
                    f"notebook {self.__notebook_name} cell {cell_index} failed to parse as json"
                )
        else:
            try:
                settings_dicts = yaml.safe_load(source)
            except Exception:
                raise RuntimeError(
                    f"notebook {self.__notebook_name} cell {cell_index} failed to parse as yaml"
                )

        return settings_dicts


    def set_settings(self, new_settings, save_path, blank_cell_index=2):
        """
        Set settings to new values and saves a new version of notebook.
        """
#to do: here the loading of notebook is repeated - maybe have it as a separate function?
        ipynb_filename = os.path.join(self.__notebook_paths , f"{self.__notebook_name}.ipynb")

        # Read the notebook into memory.
        with open(ipynb_filename, "r") as stream:
            notebook = nbformat.read(stream, as_version=4)
#to do: Check here not to overwrite the dictionary

        source = []
        for key, value in new_settings.items():
            if type(value) is dict:
                source.append(f"{key}='{value['value']}'\n")
            else:
                source.append(f"{key}='{value}'\n") 
        source = ''.join(source)
        notebook["cells"][blank_cell_index]["source"] = source
        nbformat.write(notebook, save_path)
        return
#to do: Bring submit option / code here


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
    
    def __init__(self, ptyrex_json=False, 
                       virtual_image=False, 
                       ptyrex_submit=False, 
                       au_calibration_submit=False,
                       radial_transformation_submit=False,
                       software_basedir=None):
        if ptyrex_json:
            self._ptyrex_json()
        elif virtual_image:
            self._virtual_images()
        elif ptyrex_submit:
            self._ptyrex_submit()
        elif au_calibration_submit:
            self._au_calibration_submit()
        elif radial_transformation_submit:
            self._radial_transformation_submit()
        else:
            self._activate()
            
        if sw_basedir != None:
            self.software_basedir = '/'+software_basedir
        else:
            self.software_basedir = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/'

    def _paths(self, basedir, year, session, subfolder_check, subfolder):

        if subfolder == '' and subfolder_check:
            print("**************************************************")
            print("'subfolder' is not speicified")
            print("All MIB files in 'Merlin' folder will be converted")
            print("**************************************************")
        
        self.src_path = f'/{basedir}/{year}/{session}/Merlin/{subfolder}'
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

        self.dest_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
        print("destination_path: "+self.dest_path)
        if not os.path.exists(self.dest_path) and src_path_flag:
            os.makedirs(self.dest_path)
            print("created: "+self.dest_path)

        if subfolder == '':
            self.script_save_path = f'/{basedir}/{year}/{session}/processing/Merlin/scripts'
        else:
            self.script_save_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}/scripts'
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

        self.python_script_path = self.software_basedir + 'MIB_convert_submit.py'
        
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

            if reshaping == "Auto_reshape":
                auto_reshape = True
                no_reshaping = False
                use_fly_back = False
                known_shape = False
            elif reshaping == "Flyback":
                auto_reshape = False
                no_reshaping = False
                use_fly_back = True
                known_shape = False
            elif reshaping == "Known_shape":
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
                mask_path = self.software_basedir + '29042024_12bitmask.h5'
                
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
                

    def _ptycho(self,basedir, year, session, subfolder, create_ptycho_folder, ptycho_config_name, ptycho_template_path,
                overwrite, verbose):
        '''
        This part of the code is designed to automatic generate Ptyrex Json and associated folders, its uses glob.glob
        and the inputted year, session and subfolder varibles to find already converted data files, using the
        converted data meta file and known dictionary of parameters it is possible fill out the json file
        frederick allars 18-08-2025
        '''

        if create_ptycho_folder == True:

            '''define the source path'''
            src_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'

            '''use glob and os to find the meta data files'''
            try:
                os.chdir(src_path)

                for num, file in enumerate(sorted(list(glob.glob('*/*.hdf')))):
                    '''debug statement to check file paths'''
                    if verbose == True:
                        print(str(num) + ': ' + os.path.join(src_path, file))
                    folder_pty_out = os.path.join(src_path, file[:file.find('/')]) + '/pty_out'
                    folder_inital = os.path.join(src_path, file[:file.find('/')]) + '/pty_out/initial_recon'

                    if verbose:
                        print('attempting to create ptychography folder in ' + file[:file.find('/')] + ' ...\n')
                    try:
                        os.makedirs(folder_pty_out)
                    except:
                        if verbose == True:
                            print('skipping pty_out folder creation as it already has pty_out folder')
                    try:
                        os.makedirs(folder_inital)
                    except:
                        if verbose == True:
                            print('skipping initial folder creation as it already has pty_out/initial folder\n')

                    '''determine the meta data path from the time stamp of the folder'''
                    meta_file = os.path.join(src_path, file[:file.find('/')]) + '/' + file[:file.find('/')] + '.hdf'
                    '''debug print below'''
                    if verbose == True:
                        print('\nCurrent meatdata file: ' + meta_file + '...\n')

                    '''TODO add py4DSTEM code which automatic guess the camera length from the subset of the collected diffraction patterns'''

                    '''check that the config_name parameter has been filled if not give it a default name'''
                    if ptycho_config_name == '':
                        config_name = 'pty_recon'
                    else:
                        config_name = ptycho_config_name

                    '''TODO: set up some standard ptyREX config files to reference at different energies'''
                    if ptycho_template_path == '':
                        template_path = self.software_basedir + 'UserExampleJson.json'
                    else:
                        template_path = ptycho_template_path

                    try:
                        gen_config(template_path, folder_inital, config_name, meta_file, overwrite = overwrite,
                                   verbose = verbose)
                        if verbose:
                            print('ptychography folder suscessfully created.\n')
                    except:
                        print(traceback.format_exc())
                        print('The metadata for this dataset seems to be missing or might be mid-conversion, '
                              'please check the file \nskipping for now...')

            except:
                print('\n' + str(sorted(list(glob.glob('*/*.hdf'))) + '\n'))
                print('the enetred path does not register as a available  path to any E02 data')
        if create_ptycho_folder == True:
            print('Ptychography folders and files generated...')

      
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
        basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        
        subfolder_check = Checkbox(value=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)
        
        path_verbose = Checkbox(value=False, description="Show the metadata of each MIB file", style=st)

        
        reshaping = Select(options=['Auto_reshape', 'Flyback', 'Known_shape', 'No_reshaping'],
                            value='Auto_reshape',
                            rows=4,
                            description='Choose a reshaping option',
                            disabled=False, style=st)
        
        Scan_X = IntText(description='Scan_X: (available for Known_shape)', style=st)
        Scan_Y = IntText(description='Scan_Y: (available for Known_shape)', style=st)

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
        n_jobs = IntSlider(value=3, min=1, max=9, step=1,
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
                                          basedir=basedir,
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
        basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder = Text(description='Subfolder:', style=st)
        ptycho_config_name = Text(value="ptycho",description='Enter config name (optional) :', style=st)
        ptycho_template_path = Text(value="/dls_sw/e02/PtyREX_templates/80KeV_template.json",description='Enter template config path (optional) :', style=st)
        verbose = Checkbox(value=False, description='Use this to report/check for errors', style=st)
        overwrite = Checkbox(value=False, description='Overwrite existing ptycho json files', style=st)
        create_ptycho_folder = Checkbox(value=False, description='Create a ptychography subfolder', style=st)
        #detele_ptycho_flags  = Checkbox(value=False, description='Detele a auto processing flags', style=st)
                
        self.ptycho = ipywidgets.interact(self._ptycho,
                                        basedir=basedir,
                                        year=year,
                                        session=session,
                                        subfolder=subfolder,
                                        ptycho_config_name=ptycho_config_name,
                                        ptycho_template_path=ptycho_template_path,
                                        verbose = verbose,
                                        overwrite = overwrite,
                                        create_ptycho_folder=create_ptycho_folder)
        
    def _virtual_images(self):
        st = {"description_width": "initial"}
        basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
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
        
        n_jobs = IntSlider(value=3, min=1, max=9, step=1,
                    description='Number of multiple slurm jobs:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d', style=st)
        
        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        
        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        create_info_check = Checkbox(value=False, description='Create conversion info file', style=st)
        submit_check = Checkbox(value=False, description='Submit a slurm job', style=st)

        
        self.path = ipywidgets.interact(self._paths,
                                          basedir=basedir,
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
                                                     n_jobs=n_jobs,
                                                     node_check=node_check,
                                                     create_batch_check=create_batch_check,
                                                     create_info_check=create_info_check,
                                                     submit_check=submit_check)
        

        
    def _virtual(self, mask_path, disk_lower_thresh,
                        disk_upper_thresh, DPC_check,
                        dpc_lpass, dpc_hpass, parallax_check,
                        create_batch_check, n_jobs, node_check,
                        create_info_check, submit_check):
        
        converted_files = []
        for path, directories, files in os.walk(self.dest_path):
            for f in files:
                if f.endswith('_data.hdf5'):
                    folder_name = f.replace('_data.hdf5','')
                    converted_path = self.dest_path + '/' + folder_name + '/' + f
                    converted_files.append(converted_path)
        
        python_script_path = self.software_basedir + 'py4DSTEM_virtual_image.py'
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

                f.write(f"#SBATCH --array=0-{len(converted_files)-1}%{n_jobs}\n")
                f.write(f"#SBATCH --error={self.script_save_path}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={self.script_save_path}{os.sep}%j_output.out\n")

                f.write('module load python/epsic3.10\n\n')            
                f.write(f"python {python_script_path} {info_path} $SLURM_ARRAY_TASK_ID\n")

            print("sbatch file created: "+bash_script_path)
            print("submission python file: "+python_script_path)        

        if mask_path == '':
            mask_path = self.software_basedir + '29042024_12bitmask.h5'
            
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
                
                
    def _au_calibration_submit(self):
        st = {"description_width": "initial"}
        basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        au_cal_folder = Text(description='Au data folder name:', style=st)
        
        pixel_size_factor = FloatText(description='Multi. factor for the initial pixel size guess', value=1.0, style=st)
        ring_det_range = IntText(description='Pixel range to detect a diffraction ring', value=8, style=st)
        
        n_jobs = IntSlider(value=3, min=1, max=9, step=1,
                            description='Number of multiple slurm jobs:',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            readout_format='d', style=st)
        
        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        
        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        submit_check = Checkbox(value=False, description='Submit a slurm job', style=st)        
        
        self.au_calibration_values = ipywidgets.interact(self._au_calibration,
                                                        basedir=basedir,
                                                        year=year,
                                                        session=session,
                                                        au_cal_folder=au_cal_folder,
                                                        pixel_size_factor=pixel_size_factor,
                                                        ring_det_range=ring_det_range,
                                                        n_jobs=n_jobs,
                                                        node_check=node_check,
                                                        create_batch_check=create_batch_check,
                                                        submit_check=submit_check)
        
        
    def _au_calibration(self, basedir, year, session, au_cal_folder, 
                        pixel_size_factor, ring_det_range, n_jobs,
                       node_check, create_batch_check, submit_check):
        
        current = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        starting_notebook_path = self.software_basedir
        starting_notebook_name = 'au_xgrating_cal_submit'
        nb = NotebookHelper(starting_notebook_path, starting_notebook_name)

        old_settings = nb.get_settings(1) # settings should be cell index 1
        old_settings = old_settings.split(' ')
        old_keys = [i.split('=')[0] for i in old_settings]
        old_vals = [i.split('=')[1] for i in old_settings]
        old_dict = dict(zip(old_keys, old_vals))

        # Specify the root directory for the Merlin folders
        merlin_root = basedir+ '/' + year + '/' + session + '/processing/Merlin/' + au_cal_folder
        hdf5_file_paths = glob.glob(merlin_root+ '/*/*.hdf5', recursive=True)

        # Output the paths
        hdf5_file_paths.sort()
        print(len(hdf5_file_paths))
        print(*hdf5_file_paths, sep="\n")

        if create_batch_check:
            code_path = merlin_root + '/cluster_logs'
            if not os.path.exists(code_path):
                os.mkdir(code_path)
                
            # make some changes in new setting
            # log files from the cluster jobs and the bash script will be saved here:
            new_notebook_paths_list = []
            for file in hdf5_file_paths:
                # update the settings
                new_setting = old_dict.copy()
                new_setting['file_path'] = file
                new_setting['save_path_name'] = 'automatic_Au_calibration'
                new_setting['pixel_size_factor'] = pixel_size_factor
                new_setting['ring_det_range'] = ring_det_range

                save_path = os.path.join(os.path.dirname(file), new_setting['save_path_name'])
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                new_notebook_path = os.path.join(save_path, 'submitted_notebook.ipynb')
                nb.set_settings(new_setting, new_notebook_path)
                new_notebook_paths_list.append(new_notebook_path)


            note_book_path_file = os.path.join(code_path, 'notebook_list.txt')
            with open (note_book_path_file, 'w') as f:
                f.write('\n'.join(new_notebook_paths_list))
            
            bash_script_path = os.path.join(code_path, 'cluster_submit.sh')
            with open (bash_script_path, 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition %s\n'%node_check)
                f.write('#SBATCH --job-name epsic_notebook\n')
                f.write('#SBATCH --time 02:00:00\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node 1\n')
                f.write('#SBATCH --mem 0G\n')

                f.write(f"#SBATCH --array=0-{len(new_notebook_paths_list)-1}%{n_jobs}\n")
                f.write(f"#SBATCH --error={code_path}{os.sep}logs_{current}{os.sep}error_%j.out\n")
                f.write(f"#SBATCH --output={code_path}{os.sep}logs_{current}{os.sep}output_%j.out\n")
                f.write(f"module load python/epsic3.10\n")
                f.write(f"mapfile -t paths_array < {note_book_path_file}\n")

                f.write('echo ${paths_array[$SLURM_ARRAY_TASK_ID]}\n')
                f.write('jupyter nbconvert --to notebook --inplace --ClearMetadataPreprocessor.enabled=True ${paths_array[$SLURM_ARRAY_TASK_ID]}\n')
                f.write('jupyter nbconvert --to notebook --allow-errors --execute ${paths_array[$SLURM_ARRAY_TASK_ID]}')
            
            print("sbatch file created: "+bash_script_path)
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
        
        
    def _radial_transformation_submit(self):
        st = {"description_width": "initial"}
        basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder = Text(description='Subfolder:', style=st)
        au_cal_folder = Text(description='Au data folder name:', style=st)
        
        R_Q_ROTATION = FloatText(description='Angle between R and Q (optional)', value=0.0, style=st)
        
        also_rpl = Checkbox(value=False, description='Save the results also as .rpl (optional)', style=st)
        mask_path = Text(description='Enter the mask file path (optional) :', style=st)
        
        fast_origin = Checkbox(value=True, description='Perform a fast centre finding', style=st)
        
        n_jobs = IntSlider(value=3, min=1, max=9, step=1,
                            description='Number of multiple slurm jobs:',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            readout_format='d', style=st)
        
        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        
        create_info_check = Checkbox(value=False, description='Create conversion info file', style=st)
        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        submit_check = Checkbox(value=False, description='Submit a slurm job', style=st)        
        
        self.radial_transformation_values = ipywidgets.interact(self._radial_transformation,
                                                        basedir=basedir,
                                                        year=year,
                                                        session=session,
                                                        subfolder=subfolder,
                                                        au_cal_folder=au_cal_folder,
                                                        R_Q_ROTATION=R_Q_ROTATION,
                                                        also_rpl=also_rpl,
                                                        mask_path=mask_path,
                                                        fast_origin=fast_origin, 
                                                        n_jobs=n_jobs,
                                                        node_check=node_check,
                                                        create_info_check=create_info_check,        
                                                        create_batch_check=create_batch_check,
                                                        submit_check=submit_check)
    
    
    
    def _radial_transformation(self, basedir, year, session, subfolder, au_cal_folder,
                              R_Q_ROTATION, also_rpl, mask_path, fast_origin,
                              n_jobs, node_check, create_info_check, 
                              create_batch_check, submit_check):

        script_path = self.software_basedir + 'apply_elliptical_correction_polardatacube.py'
        YEAR = year
        VISIT = session
        sub = subfolder
        au_calib_name = au_cal_folder
        base_dir = f'{basedir}/{YEAR}/{VISIT}/processing/Merlin'
        au_calib_dir = f'{basedir}/{YEAR}/{VISIT}/processing/Merlin/{au_calib_name}/' # The whole path can be manually specified

        au_calib_list = glob.glob(au_calib_dir+'/*/*.json', recursive=True)
        if au_calib_list == []:
            print("No calibration data exists")
            print("Please check the directory path again")
        else:
            print("Calibration data list")
            print(*au_calib_list, sep='\n')

        # (optional) Angle between the real space dimensions and the reciprocal space dimensions
        # R_Q_ROTATION = eval(R_Q_ROTATION) 

        also_rpl = also_rpl # if 'True', the results will also be saved in '.rpl' format

        # mask_path = mask_path # if you would like to apply a certain mask to the diffraction patterns

        fast_origin = fast_origin # if not 'True', the process includes the Bragg peak finding (the centre positions could be more accurate, but it needs more time)
        
        file_adrs = glob.glob(base_dir+'/'+sub+'/*/*/*_data.hdf5', recursive=True)
        if file_adrs == []:
            file_adrs = glob.glob(base_dir+'/'+sub+'/*/*_data.hdf5', recursive=True)
            if file_adrs == []:
                file_adrs = glob.glob(base_dir+'/'+sub+'/*_data.hdf5', recursive=True)
                if file_adrs == []:
                    print("Please make sure that the base directory and subfolder name are correct.")

        print(len(file_adrs))
        print(*file_adrs, sep='\n')
        
        data_labels = []
        for adr in file_adrs:
            datetime = adr.split('/')[-2]
            if os.path.exists(os.path.dirname(adr) + "/" + datetime + "_azimuthal_data_centre.png"):
                continue
            else:
                data_labels.append(sub+'/'+adr.split('/')[-2])

        # print(len(data_labels))
        # print(*data_labels, sep='\n')

        if create_info_check:
            code_path = base_dir + '/' + sub + '/cluster_logs'
            if not os.path.exists(code_path):
                os.mkdir(code_path)
                
            info_path = os.path.join(code_path, 'transformation_info.txt')
            with open (info_path, 'w') as f:
                f.write(
                    f"basedir = {basedir}\n"
                    f"YEAR = {YEAR}\n"
                    f"VISIT = {VISIT}\n"
                    f"sub = {sub}\n"
                    f"data_labels = {data_labels}\n"
                    f"au_calib_dir = {au_calib_dir}\n"
                    f"R_Q_ROTATION = {R_Q_ROTATION}\n"
                    f"also_rpl = {also_rpl}\n"
                    f"mask_path = {mask_path}\n"
                    f"fast_origin = {fast_origin}\n"
                        )
            print("conversion info file created: "+info_path)

        if create_batch_check:
            bash_script_path = os.path.join(code_path, 'cluster_submit.sh')
            with open(bash_script_path, 'w') as f:
                f.write("#!/usr/bin/env bash\n")
                f.write('#SBATCH --partition %s\n'%node_check)
                f.write("#SBATCH --job-name=rad_trans\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=4\n")
                f.write("#SBATCH --cpus-per-task=1\n")
                f.write("#SBATCH --time=2:00:00\n")
                f.write("#SBATCH --mem=128G\n")
                f.write("#SBATCH --output=%s/%%j.out\n"%code_path)
                f.write("#SBATCH --error=%s/%%j.error\n\n"%code_path)
                f.write(f"#SBATCH --array=0-{len(data_labels)-1}%{n_jobs}\n")

                f.write("module load python/epsic3.10\n")
                f.write(f'python {script_path} {info_path} $SLURM_ARRAY_TASK_ID\n')
            print("sbatch file created: "+bash_script_path)
            
        
        if submit_check:
            sshProcess = subprocess.Popen(['ssh',
                                           '-tt',
                                           'wilson'],
                                           stdin=subprocess.PIPE, 
                                           stdout = subprocess.PIPE,
                                           universal_newlines=True,
                                           bufsize=0)
            sshProcess.stdin.write("echo END\n")
            sshProcess.stdin.write("sbatch "+bash_script_path+"\n")
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



    def _ptyrex_paths(self,basedir, year, session, subfolder, ptycho_config_name, create_ptycho_bash_script_check,
                      node_type, ptycho_time, submit_ptyrex_job, verbose=False):
        '''define the source path'''
        src_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
        script_folder = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}/scripts'
        tmp_list = []
        test_string = 'autoptycho_is_done.txt'

        '''use glob and os to find the meta data files'''
        if basedir == '' or year == '' or session == '' or subfolder == '':
            print('\nwaiting for the folowing inputs: basedir, year, session, subfolder\n')
        else:
            os.chdir(src_path)
            for num, file in enumerate(sorted(list(glob.glob('*/*/*/*' + ptycho_config_name + '.json')))):
                '''debug statement to check file paths'''
                if verbose == True:
                    print(str(num) + ': ' + os.path.join(src_path, file))
                    # print(str(num) + ': ' + os.path.join(src_path,file.replace(ptycho_config_name + '.json', test_string)))

                '''check whether the data has been processed before - if it has then it should have autoptycho_is_done text file'''
                if os.path.exists(os.path.join(src_path, file.replace(ptycho_config_name + '.json', test_string))):
                    if verbose:
                        print(
                            '\nskipping json as it already been processed, to process this again please clear autoptycho flag\n')
                else:
                    tmp_list.append(os.path.join(src_path, file))

            '''Create bash scripts for each of the json files'''
            if create_ptycho_bash_script_check:
                print('\nfound json files and creating bash scripts...\n')
                bash_list = _create_ptyrex_bash_submit(tmp_list, script_folder, node_type, ptycho_time, verbose)
            '''submit the bash scripts to the cluster'''
            if submit_ptyrex_job:
                _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, tmp_list)
                if verbose:
                    print(f'submited jobs to wilson')
            if create_ptycho_bash_script_check and submit_ptyrex_job:
                return tmp_list, bash_list



    def _ptyrex_submit(self):
        st = {"description_width": "initial"}
        basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder = Text(description='Subfolder:', style=st)
        ptycho_config_name = Text(description='Name of the json file to process:', style=st)
        create_ptycho_bash_script_check = Checkbox(value=False, description='Create PtyREX bash script', style=st)
        node_type = Dropdown(options=['Volta', 'Pascal'], value='Pascal', description='type of gpu to use')
        ptycho_time = Dropdown(options=['00:30:00', '01:00:00', '02:00:00', '04:00:00', '08:00:00'],value='00:30:00',description='Processing time: (HH:MM:SS)')
        submit_ptyrex_job = Checkbox(value=False, description='Submit PtyREX job', style=st)
        verbose = Checkbox(value=False, description='Check this for debugging and error printing', style=st)

        self.ptyrex_paths = ipywidgets.interact(self._ptyrex_paths,
                                          basedir=basedir,
                                          year=year, 
                                          session=session,
                                          subfolder=subfolder,
                                          ptycho_config_name=ptycho_config_name,
                                          create_ptycho_bash_script_check=create_ptycho_bash_script_check,
                                          node_type=node_type,
                                          ptycho_time=ptycho_time,
                                          submit_ptyrex_job=submit_ptyrex_job,
                                          verbose=verbose)



