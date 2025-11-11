#!/usr/bin/env bash
import sys
import pprint
import time
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
from IPython.display import display
import os
import glob
import logging
import subprocess
import nbformat
import yaml
import json
import re

#error tracking imports
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
# ----------------------------------------------------------------------------------------

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
                       ptyrex_single=False,
                       au_calibration_submit=False,
                       radial_transformation_submit=False,
                       software_basedir=None):

        if software_basedir != None:
            self.software_basedir = '/'+software_basedir+'/'
        else:
            self.software_basedir = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/'


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
        elif ptyrex_single:
            self._ptyrex_single_recon()
        else:
            self._activate()
            
    def prefill_boxes(self):
        '''
        This function is used to prefill the basedir, year and session boxes in 
        the ipython widgets such that users do not have repeatly enter in the 
        same information if they are running different cells within the same notebook
        this only works if the folder which the note book is in is associated with a 
        particular user session and this can work with staged data as well. so is based
        strongly on os.chdir function.
        '''
        #ToDo make these varibles be stored in self such that once it has been filled out one

        st = {"description_width": "initial"}
        current_dir = os.getcwd()
        if current_dir[:13] == '/dls/e02/data':
            basedir = Text(value=current_dir[0:13], description='Base data directory path:', style=st)
            year = Text(value=current_dir.split('/')[4],description='Year:', style=st)
            session = Text(value=current_dir.split('/')[5],description='Session:', style=st)
        elif current_dir == '/dls/staging/dls/e02/data':
            basedir = Text(value=current_dir[0:25], description='Base data directory path:', style=st)
            year = Text(value=current_dir.split('/')[6],description='Year:', style=st)
            session = Text(value=current_dir.split('/')[7],description='Session:', style=st)
        else:
            basedir = Text(value="/dls/e02/data", description='Base data directory path:', style=st)
            year = Text(description='Year:', style=st)
            session = Text(description='Session:', style=st)
        return basedir, year, session

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

            display(self.keys_show)
                
    def _organize(self, reshaping,Scan_X, Scan_Y,bin_nav_widget,bin_sig_widget,
                  node_check,n_jobs,create_virtual_image,mask_path,disk_lower_thresh,
                  disk_upper_thresh,DPC_check,parallax_check):

        self.reshaping = reshaping
        self.Scan_X = Scan_X
        self.Scan_Y = Scan_Y
        self.bin_nav_widget = bin_nav_widget
        self.bin_sig_widget = bin_sig_widget
        self.node_check = node_check
        self.n_jobs = n_jobs
        self.create_virtual_image = create_virtual_image
        self.mask_path = mask_path
        self.disk_lower_thresh = disk_lower_thresh
        self.disk_upper_thresh = disk_upper_thresh
        self.DPC_check = DPC_check
        self.parallax_check = parallax_check

    def _create_info(self):
        self.info_path = os.path.join(self.script_save_path, 'convert_info.txt')
            
        if self.bin_sig_widget != 1:
            bin_sig_flag = 1
            bin_sig_factor = self.bin_sig_widget
        else:
            bin_sig_flag = 0
            bin_sig_factor = self.bin_sig_widget

        if self.bin_nav_widget != 1:
            bin_nav_flag = 1
            bin_nav_factor = self.bin_nav_widget
        else:
            bin_nav_flag = 0
            bin_nav_factor = self.bin_nav_widget

        if self.reshaping == "Auto_reshape":
            auto_reshape = True
            no_reshaping = False
            use_fly_back = False
            known_shape = False
        elif self.reshaping == "Flyback":
            auto_reshape = False
            no_reshaping = False
            use_fly_back = True
            known_shape = False
        elif self.reshaping == "Known_shape":
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
        
        if self.mask_path == '':
            self.mask_path = self.software_basedir + '29042024_12bitmask.h5'
            
        with open (self.info_path, 'w') as f:
            f.write(
                f"to_convert_paths = {self.to_convert}\n"
                f"auto_reshape = {auto_reshape}\n"
                f"no_reshaping = {no_reshaping}\n"
                f"use_fly_back = {use_fly_back}\n"
                f"known_shape = {known_shape}\n"
                f"Scan_X = {self.Scan_X}\n"
                f"Scan_Y = {self.Scan_Y}\n"
                f"iBF = {iBF}\n"
                f"bin_sig_flag = {bin_sig_flag}\n"
                f"bin_sig_factor = {bin_sig_factor}\n"
                f"bin_nav_flag = {bin_nav_flag}\n"
                f"bin_nav_factor = {bin_nav_factor}\n"
                f"reshaping = {self.reshaping}\n"
                f"create_virtual_image = {self.create_virtual_image}\n"
                f"mask_path = {self.mask_path}\n"
                f"disk_lower_thresh = {self.disk_lower_thresh}\n"
                f"disk_upper_thresh = {self.disk_upper_thresh}\n"
                f"DPC = {self.DPC_check}\n"
                f"parallax = {self.parallax_check}\n"
                    )
            
        print("conversion info file created: "+self.info_path)

    def _create_batch_script(self):
        self.python_script_path = self.software_basedir + '/MIB_convert_submit.py'
        self.bash_script_path = os.path.join(self.script_save_path, 'cluster_submit.sh')
        with open (self.bash_script_path, 'w') as f:
            f.write('#!/usr/bin/env bash\n')
            f.write('#SBATCH --partition %s\n'%self.node_check)
            f.write('#SBATCH --job-name mib_convert\n')
            f.write('#SBATCH --nodes 1\n')
            f.write('#SBATCH --tasks-per-node 1\n')
            f.write('#SBATCH --cpus-per-task 1\n')
            f.write('#SBATCH --time 05:00:00\n')
            if self.create_virtual_image:
                f.write('#SBATCH --mem 192G\n\n')
            else:
                f.write('#SBATCH --mem 64G\n\n')

            f.write(f"#SBATCH --array=0-{len(self.to_convert)-1}%{self.n_jobs}\n")
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
        
    def _create_info_on_click(self, b):
        with self.info_out:
            self.info_out.clear_output(wait=True)
            self._create_info()

    def _create_batch_on_click(self, b):
        with self.batch_out:
            self.batch_out.clear_output(wait=True)
            self._create_batch_script()

    def _submit_on_click(self, b):
        with self.submit_out:
            self.submit_out.clear_output(wait=True)
            self._submit()
                
    def _submit(self):
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
        st = {"description_width": "initial"}
        basedir, year, session = self.prefill_boxes()
        subfolder_check = Checkbox(value=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)
        path_verbose = Checkbox(value=False, description="Show the metadata of each MIB file", style=st)
        reshaping = Select(options=['Auto_reshape', 'Flyback', 'Known_shape', 'No_reshaping'],
                           value='Auto_reshape', rows=4, description='Choose a reshaping option',
                           disabled=False, style=st, layout=Layout(width='70%', height='auto'))
        Scan_X = IntText(description='Scan_X: (Known_shape)', style=st)
        Scan_Y = IntText(description='Scan_Y: (Known_shape)', style=st)
        bin_nav_widget = IntSlider(value=2, min=1, max=8, step=1, description='Bin the scan region:', style=st)
        bin_sig_widget = IntSlider(value=2, min=1, max=8, step=1, description='Bin the diffraction pattern:', style=st)
        
        node_check = RadioButtons(options=['cs04r', 'cs05r'], description='Select the cluster node (cs04r recommended)', disabled=False)
        n_jobs = IntSlider(value=3, min=1, max=9, step=1, description='Number of multiple slurm jobs:', style=st)

        create_batch_button = Button(description='Create a slurm batch file', button_style='info', layout=Layout(width='auto'))
        create_info_button = Button(description='Create a conversion info file', button_style='info', layout=Layout(width='auto'))

        create_virtual_image = Checkbox(value=False, description='Create virtual images', style=st)
        disk_lower_thresh = FloatText(description='Lower disk threshold', value=0.01, style=st)
        disk_upper_thresh = FloatText(description='Upper disk threshold', value=0.15, style=st)
        mask_path = Text(description='Mask file path (optional) :', style=st)
        DPC_check = Checkbox(value=False, description='DPC', style=st)
        parallax_check = Checkbox(value=False, description='Parallax', style=st)

        submit_check = Button(description='Submit a slurm job',
                              button_style='primary', # Main button is primary
                              layout=Layout(width='auto'))

        grid = GridspecLayout(3, 3, height='400px', layout=Layout(width='70%'))
        
        grid[0, 0] = VBox([basedir, year, session])
        grid[1, 0] = VBox([subfolder_check, subfolder, path_verbose])
        
        grid[0, 1] = reshaping
        grid[1, 1] = VBox([Scan_X, Scan_Y])
        grid[2, 1] = VBox([bin_nav_widget, bin_sig_widget])
        
        grid[0, 2] = VBox([n_jobs, node_check])
        grid[1, 2] = VBox([create_virtual_image, mask_path])
        grid[2, 2] = VBox([disk_lower_thresh, disk_upper_thresh, DPC_check, parallax_check])

        self.submit_out = ipywidgets.Output()
        self.batch_out = ipywidgets.Output()
        self.info_out = ipywidgets.Output()

        controls_paths = {'basedir': basedir, 'year': year, 'session': session,
                          'subfolder_check': subfolder_check, 'subfolder': subfolder}
        controls_verbose = {'path_verbose': path_verbose}

        controls_organize = {'reshaping': reshaping, 'Scan_X': Scan_X, 'Scan_Y': Scan_Y,
                             'bin_nav_widget': bin_nav_widget, 'bin_sig_widget': bin_sig_widget,
                             'node_check': node_check, 'n_jobs': n_jobs,
                             'create_virtual_image': create_virtual_image, 'mask_path': mask_path,
                             'disk_lower_thresh': disk_lower_thresh, 'disk_upper_thresh': disk_upper_thresh,
                             'DPC_check': DPC_check, 'parallax_check': parallax_check}

        
        self.path_out = interactive_output(self._paths, controls_paths)
        self.verbose_out = interactive_output(self._verbose, controls_verbose)
        self.values_out = interactive_output(self._organize, controls_organize)
        
        submit_check.on_click(self._submit_on_click)
        create_batch_button.on_click(self._create_batch_on_click)
        create_info_button.on_click(self._create_info_on_click)
        button_box = HBox([create_info_button, create_batch_button, submit_check])

        final_layout = VBox([
            grid,
            button_box,
            self.path_out,
            self.values_out,
            self.submit_out,
            self.batch_out,
            self.info_out,
            self.verbose_out
        ])
        display(final_layout)

    # Virtual Image Section Start---------------------------------------------------------
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
                    
    def _virtual_images(self):

        basedir, year, session = self.prefill_boxes()
        st = {"description_width": "initial"}
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
        
        python_script_path = self.software_basedir + '/py4DSTEM_virtual_image.py'
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
            mask_path = self.software_basedir + '/29042024_12bitmask.h5'
            
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
    # Virtual Image Section End---------------------------------------------------------

    # Calibration Section Start-----------------------------------------------------------
    def _au_calibration_submit(self):
        st = {"description_width": "initial"}
        basedir,year,session = self.prefill_boxes()
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
        hdf5_file_paths = glob.glob(merlin_root+ '/*/*_data.hdf5', recursive=True)

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
    # Calibration Section End-----------------------------------------------------------

    # Radial Transformation Section Start-------------------------------------------------        
    def _radial_transformation_submit(self):
        st = {"description_width": "initial"}
        basedir,year,session = self.prefill_boxes()
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
        overwrite_check = Checkbox(value=True, description='Ignore the previously transformed data', style=st)
        
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
                                                        overwrite_check=overwrite_check,
                                                        create_info_check=create_info_check,        
                                                        create_batch_check=create_batch_check,
                                                        submit_check=submit_check)
    
    def _radial_transformation(self, basedir, year, session, subfolder, au_cal_folder,
                              R_Q_ROTATION, also_rpl, mask_path, fast_origin,
                              n_jobs, node_check, overwrite_check, create_info_check, 
                              create_batch_check, submit_check):

        script_path = self.software_basedir + '/apply_elliptical_correction_polardatacube.py'
        print(script_path)
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

        # print(len(file_adrs))
        # print(*file_adrs, sep='\n')
        
        data_labels = []
        for adr in file_adrs:
            datetime = adr.split('/')[-2]
            if overwrite_check:
                if os.path.exists(os.path.dirname(adr) + "/" + datetime + "_azimuthal_data_centre.png"):
                    continue
                else:
                    data_labels.append(sub+'/'+adr.split('/')[-2])
                    
            else:
                data_labels.append(sub+'/'+adr.split('/')[-2])

        print(len(data_labels))
        print(*data_labels, sep='\n')

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
                f.write("#SBATCH --time=4:00:00\n")
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
    # Radial Transformation Section End---------------------------------------------------        

    # Ptycho Method Section Start---------------------------------------------------------
    def _ptyrex_json(self):
        basedir,year,session = self.prefill_boxes()
        st = {"description_width": "initial"}
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
            if verbose:
                print(f'looking at this path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')

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
               
    def _ptyrex_paths(self,basedir, year, session, subfolder, ptycho_config_name, create_ptycho_bash_script_check,
                      node_type, ptycho_time, submit_ptyrex_job, delete_flaggin_files, verbose=False):
        '''define the source path'''
        src_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
        script_folder = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}/scripts'

        tmp_list = []
        test_string = 'autoptycho_is_done.txt'

        '''use glob and os to find the meta data files'''
        if basedir == '' or year == '' or session == '' or subfolder == '' or ptycho_config_name == '':
            print('\nwaiting for the folowing inputs: basedir, year, session, subfolder and config_name\n')
            print(f'current path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')
        else:
            '''try statement is used here to catch the case where all parameters have inputs but are still not correct mid type'''
            try:
                os.chdir(src_path)
                print(f'current path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}\n')
                if verbose:
                    print(f'json files found in the current path are listed below:\n')
                for num, file in enumerate(sorted(list(glob.glob('*/*/*/*' + ptycho_config_name + '.json')))):
                    '''debug statement to check file paths'''
                    if verbose == True:
                        print(str(num) + ': ' + os.path.join(src_path, file))
                        # print(str(num) + ': ' + os.path.join(src_path,file.replace(ptycho_config_name + '.json', test_string)))

                    '''check whether the data has been processed before - if it has then it should have autoptycho_is_done text file'''
                    #ToDo fix this to test whether even in the case where only a subsection of the code matches (it must be a whole match)
                    #for example ptycho_Sample and 2_ptycho_Sample will be brought up by ptycho_Sample as they share strings
                    if os.path.exists(os.path.join(src_path, file.replace(ptycho_config_name + '.json', test_string))):
                        if verbose:
                            print(
                                '\nskipping json as it already been processed, to process this again please delete flagging files\n')
                    else:
                        tmp_list.append(os.path.join(src_path, file))

                '''check whether the scripting folder exists, if not create the folder'''
                if os.path.exists(script_folder) == False:
                    os.mkdir(script_folder)

                '''delete auto ptycho flagging  files if there are any'''
                if delete_flaggin_files:
                    delete_ptycho_flagging_files(basedir,year,session,subfolder,verbose)
                
                '''Create bash scripts for each of the json files'''
                if create_ptycho_bash_script_check:
                    print('\n***\nFound json files and creating bash scripts...\n***\n')
                    bash_list = _create_ptyrex_bash_submit(tmp_list, script_folder, node_type, ptycho_time, verbose)
                '''submit the bash scripts to the cluster'''
                if submit_ptyrex_job:
                    _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, tmp_list)
                    if verbose:
                        print(f'submited jobs to wilson')
                if create_ptycho_bash_script_check and submit_ptyrex_job:
                    return tmp_list, bash_list
            
            except:
                print(f'current path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')
                print(f'\nEither the entered path does not exist, Path searched: {src_path}, or a different error has occured\n')

    def _ptyrex_submit(self):
        basedir,year,session = self.prefill_boxes()
        st = {"description_width": "initial"}
        subfolder = Text(description='Subfolder:', style=st)
        ptycho_config_name = Text(value='ptycho',description='Name of the json file to process:', style=st)
        create_ptycho_bash_script_check = Checkbox(value=False, description='Create PtyREX bash script', style=st)
        node_type = Dropdown(options=['Volta', 'Pascal'], value='Pascal', description='type of gpu to use')
        ptycho_time = Dropdown(options=['00:30:00', '01:00:00', '02:00:00', '04:00:00', '08:00:00'],value='00:30:00',description='Processing time: (HH:MM:SS)')
        submit_ptyrex_job = Checkbox(value=False, description='Submit PtyREX job', style=st)
        delete_flaggin_files = ToggleButton(value=False,description='delete flagging files', disabled=False, button_style='')
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
                                          delete_flaggin_files = delete_flaggin_files,
                                          verbose=verbose)
        
    def _single_recon(self,basedir, year, session, subfolder, ptycho_config_name,
                      Choose_timestamp, Choose_recon_parameter, current_value,
                      new_value, update_json, submit_ptyrex_job, verbose=False):
        self.tmp_list = []
        self.timestamp_list = []
        self.recon_timestamp_list = []
        self.recon_list = []

        '''use glob and os to find the meta data files'''
        if basedir == '' or year == '' or session == '' or subfolder == '' or ptycho_config_name == '':
            print('\nwaiting for the folowing inputs: basedir, year, session, subfolder and config_name\n')
            print(f'current path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')
        else:
            if verbose:
                print(f'current path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')
            '''this part of the code finds all of the timestamps to populate timestamp_list'''
            src_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
            os.chdir(src_path)
            for num, file in enumerate(sorted(list(glob.glob('*/*/*/*' + ptycho_config_name + '.json')))):
                if verbose:
                    print(str(num) + ': ' + os.path.join(src_path, file))
                self.tmp_list.append(os.path.join(src_path, file))
                time_stamp_index = os.path.join(src_path, file).find('/pty_out')
                self.timestamp_list.append(os.path.join(src_path, file)[time_stamp_index - 15:time_stamp_index])
            if verbose:
                print(self.timestamp_list)
            self.single_recon.widget.children[5].options = self.timestamp_list
            #self.single_recon.widget.children[5].value = self.timestamp_list[0]
            
            try:
                indexer = self.timestamp_list.index(Choose_timestamp)
                current_file = self.tmp_list[indexer]
                current_folder = current_file.replace(f'{ptycho_config_name}.json','')

                '''
                get the current value of the paramter being indexed and update the value so that the user has
                some understanding want the unit of the value should be? maybe we should implment units later?
                '''
                self.single_recon.widget.children[7].value = get_current_json_val(current_file,Choose_recon_parameter)
                #for i in self.single_recon.widget.children:
                #    print(i.value)

                if self.single_recon.widget.children[5].value != 'empty':
                    print('timestamp chosen')
                    #print(dir(self.single_recon.widget.children[7]))
                    self.single_recon.widget.children[9].disabled = False
                    self.single_recon.widget.children[10].disabled = False
                else:
                    print('cannot edit file or submit jobs with incomplete recon section')

                '''
                edit_ptyrex_json_parameter changes the json immendiately this might be a problem the current soultion is 
                gate the function with a toogle button. the issue with this, is that this requires multiple presses of 
                the toggle button to mutiple different values. maybe we store changes and then apply all at once later?
                '''
                print(f'The currently selected json is: {current_file}')
                if update_json:
                    edit_ptyrex_json_parameter(current_file, Choose_recon_parameter, new_value, verbose)
                    '''There is a delay in saving the file so I think we have sleep and the update the value in the new_value box'''
                    self.single_recon.widget.children[7].value = get_current_json_val(current_file,Choose_recon_parameter)
                    self.single_recon.widget.children[9].value = False
                    #self.single_recon.widget.children[7].value = new_value
                    #time.sleep(1)

                if submit_ptyrex_job:
                    script_folder = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}/scripts/'
                    if os.path.exists(script_folder) != True:
                        os.mkdir(script_folder)
                    bash_list = _create_ptyrex_bash_submit([current_file], script_folder, 'Volta', '04:00:00', verbose=False)
                    #bash_list = [f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}/scripts/{Choose_timestamp[-6:]}_ptyrex_submit.sh']
                    if verbose:
                        print(bash_list)
                    
                    _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, verbose=False)
                    self.single_recon.widget.children[10].value = False
            except:
                print('Please select a timestamp, if the timestamp option is unreponsive and the path is correct an error has\n' \
                'occured. please click the debugging checkbox for detials')
                
    def _ptyrex_single_recon(self):
        st = {"description_width": "initial           a"}
        basedir,year,session = self.prefill_boxes()
        subfolder = Text(description='Subfolder:', style=st)
        ptycho_config_name = Text(description='Name of the json file to process:', style=st)
        Choose_timestamp = Dropdown(options=['empty'], value='empty', description='choose a timestamp', style=st)
        Choose_recon_parameter = Dropdown(options=['camera length','scan rotation','defocus','object update',
                                                   'probe update','number of slices', 'slice thickness'], 
                                                   value='defocus', description='choose a parameter to change', style=st)
        current_value = FloatText(value=1e-9,description = 'The current value of the chosen parameter', style=st,disabled=True)
        new_value = FloatText(value=1e-9,description = 'enter a new value for the chosen parameter', style=st)
        update_json = ToggleButton(value=False,description='update the current json', disabled=True, button_style='', style=st)
        submit_ptyrex_job = ToggleButton(value=False,description='submit the ptyrex json to the cluster for processing', disabled=True, style=st)
        verbose = Checkbox(value=False, description='Check this for debugging and error printing', style=st)

        self.single_recon = ipywidgets.interact(self._single_recon,
                                            basedir=basedir,
                                            year=year, 
                                            session=session,
                                            subfolder=subfolder,
                                            ptycho_config_name=ptycho_config_name,
                                            Choose_timestamp = Choose_timestamp,
                                            Choose_recon_parameter = Choose_recon_parameter,
                                            current_value=current_value,
                                            new_value = new_value,
                                            update_json = update_json,
                                            submit_ptyrex_job = submit_ptyrex_job,
                                            verbose = verbose)
    # Ptycho Method Section End-----------------------------------------------------------        

# Function Zone Start---------------------------------------------------------------------
# Ptycho Funtion Start--------------------------------------------------------------------
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
        rot = meta_values['nominal_scan_rotation'][()]

        '''detemine the rotation angle and camera length from the HT value (accerlation voltage) and nomial camera length,
        also determine the convergence angle from the aperture size used'''
        rotation_angle, camera_length, conv_angle = Meta2Config(acc, nCL, aps, rot, factor, verbose)

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

def Meta2Config(acc,nCL,aps,rot,factor=1.7, verbose=False):


    '''This function converts the meta data from the 4DSTEM data set into parameters to be used in a ptyREX json file'''

    '''The rotation angles noted here are from ptychographic reconstructions which have been successful. see the 
    following directory for example reconstruction from which these values are derived:
     /dls/science/groups/imaging/ePSIC_ptychography/experimental_data'''
    if acc == 80e3:
        rot_angle = 238.5 - rot
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
        rot_angle = -77.585 - rot
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
        rot_angle = -85.5 - rot
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

def _create_ptyrex_bash_submit(json_files, script_folder, node_type, ptycho_time, verbose=False):
    bash_ptyrex_path = []

    for num, x in enumerate(json_files):
        if verbose:
            print(f'\n{str(num)}: {x}\n')
        time_stamp_index = x.find('/pty_out')
        time_stamp = x[time_stamp_index - 6:time_stamp_index]
        if verbose:
            print(f'\ntime_stamp = {time_stamp}\n')
        '''rememeber to self.script_save_path to save these files to the script folder'''
        bash_ptyrex_path.append(os.path.join(script_folder, f'{time_stamp}_ptyrex_submit.sh'))
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

                f.write(f"#SBATCH --error={script_folder}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={script_folder}{os.sep}%j_output.out\n")

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

def delete_ptycho_flagging_files(basedir,year,session,subfolder,verbose=False):
    '''This function is for deleting the autoptycho_is_done.txt files which prevent recontructions
       from being run more than once'''

    '''determine path path being searched and change directory to that path for glob'''
    dest_path = f'/{basedir}/{year}/{session}/processing/Merlin/{subfolder}'
    os.chdir(dest_path)
    print('File being searched: ' + dest_path)


    '''use glob and os to find and then remove the files'''
    for num,file in enumerate(sorted(list(glob.glob('*/*/*/*autoptycho_is_done.txt')))):

        '''debug: print out all of the files before deleting them'''
        os.remove(os.path.join(dest_path,file))
        if verbose:
            print(str(num) + ' deleting: ' + os.path.join(dest_path,file))

def _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, tmp_list=None, verbose=False):
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
    if tmp_list != None:
        _create_flagging_text_files(submit_ptyrex_job, tmp_list, verbose)

def ptyrex_json_helper(json_name,val):
    '''camera length = ['experiment']['detector']['position']
       scan rotation = ['process']['common']['scan']['rotation']
       defocus       = ['experiment']['optics']['lens']['defocus']
       object update = ['process']['PIE']['object']['alpha']
       probe_update  = ['process']['PIE']['object']['alpha']
       slices        = ['process']['PIE']['MultiSlice']['slices']
       thickness     = ['process']['PIE']['MultiSlice']['S_distance']
       
    '''
    if json_name == 'camera length':
        keys = ['experiment','detector','position']
        val = [0,0,val]
    elif json_name == 'scan rotation':
        keys = ['process','common','scan','rotation']
    elif json_name == 'defocus':
        keys = ['experiment','optics','lens','defocus']
        val = [val,val]
    elif json_name == 'object update':
        keys = ['process','PIE','object','alpha']
    elif json_name == 'probe update':
        keys = ['process','PIE','probe','alpha']
    elif json_name == 'number of slices':
        keys = ['process','PIE','MultiSlice','slices']
    elif json_name == 'slice thickness':
        keys = ['process','PIE','MultiSlice','S_distance']
    return keys, val

def get_current_json_val(json_path,key_name):
    '''
    this function is to obtain the current value in the json for a particular key
    such that it can be displayed before it is changed
    '''
    with open(json_path, 'r') as f:
        pty_expt = json.load(f)

    keys,new_val = ptyrex_json_helper(key_name,0)

    if len(keys) == 3:
        current_val = pty_expt[keys[0]][keys[1]][keys[2]]
    elif len(keys) == 4:
        current_val = pty_expt[keys[0]][keys[1]][keys[2]][keys[3]]
    else:
        print('error: unkown keys')
    if isinstance(current_val,list):
        if len(current_val) == 2:
            current_val = current_val[0]
        elif len(current_val) == 3:
            current_val = current_val[2]
    return current_val

def edit_ptyrex_json_parameter(json_path,key_name, new_val,verbose=False):
    
    with open(json_path, 'r') as f:
        pty_expt = json.load(f)

        keys,new_val = ptyrex_json_helper(key_name,new_val)

        '''in the case of changing the number of slices make sure that this a int and not a float
        to avoid issues during the reading of the json within ptyrex'''
        if keys[-1] == 'slices':
            new_val = int(new_val)

        if len(keys) == 3:
            pty_expt[keys[0]][keys[1]][keys[2]] = new_val
        elif len(keys) == 4:
            pty_expt[keys[0]][keys[1]][keys[2]][keys[3]] = new_val
        else:
            print('error: unkown keys')

        if verbose:
            print(f'editing the following json: {json_path}...')
                
        with open(json_path, 'w') as f:
            json.dump(pty_expt, f, indent=4)

def edit_all_ptycho_json_paramter(year,session,subfolder,json_name,key_name, new_val, verbose=False):
    
    '''determine path path being searched and change directory to that path for glob'''
    dest_path = f'/dls/e02/data/{year}/{session}/processing/Merlin/{subfolder}'
    os.chdir(dest_path)
    print('File being searched: ' + dest_path)

    '''use glob and os to find and then edit the files'''
    for num,file in enumerate(sorted(list(glob.glob('*/*/*/*' + json_name + '.json')))):

        '''debug: print out all of the files before editing them'''
        if verbose:
            print(str(num) + ': ' + os.path.join(dest_path,file))

        edit_ptyrex_json_parameter(os.path.join(dest_path,file),key_name, new_val, verbose)
# Ptycho Funtion End----------------------------------------------------------------------
# Function Zone End-----------------------------------------------------------------------