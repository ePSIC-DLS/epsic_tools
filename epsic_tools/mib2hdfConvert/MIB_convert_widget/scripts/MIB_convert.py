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


# Widgets
class convert_info_widget():

    def __init__(self):
        self._activate()

    def _paths(self, year, session, subfolder, subfolder_check):

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

        if subfolder != '' or subfolder_check==True:
            self.to_convert = self._check_differences(self.src_path, self.dest_path)
            print(*self.to_convert, sep="\n")

        #save_dir = []
        #for adr in to_convert:
        #    adr_split = adr.split('/')
        #    tmp_save = []
        #    tmp_save.append('/')
        #    tmp_save.extend(test[1:6])
        #    tmp_save.append('processing')
        #    tmp_save.extend(test[6:8])
        #    save_dir.append(os.path.join(*tmp_save))
    
    def _organize(self, year, session, subfolder_check, subfolder, no_reshaping, 
                 use_fly_back, known_shape, Scan_X, Scan_Y, add_cross_check,
                ADF_check, iBF_check, DPC_check,
                bin_nav_widget, bin_sig_widget,
                create_batch_check, create_info_check,
                submit_check):

        self._paths(year, session, subfolder, subfolder_check)

        if create_batch_check:
            bash_script_path = os.path.join(self.script_save_path, 'cluster_submit.sh')
            info_path = os.path.join(self.script_save_path, 'convert_info.txt')
            python_script_path = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts/MIB_convert_submit.py'
            
            with open (bash_script_path, 'w') as f:
                f.write('''#!/usr/bin/env bash
#SBATCH --partition cs04r
#SBATCH --job-name epsic_mib_convert
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --time 05:00:00
#SBATCH --mem 100G
'''
f"#SBATCH --array=0-{len(self.to_convert)-1}%3\n"
f"#SBATCH --error={self.script_save_path}{os.sep}error_%j.out\n"
f"#SBATCH --output={self.script_save_path}{os.sep}output_%j.out\n"
'''
echo "I am running the array job with task ID $SLURM_ARRAY_TASK_ID"
module load python/epsic3.10
            
sleep 10
'''
f"python {python_script_path} {info_path} $SLURM_ARRAY_TASK_ID\n"
                       )
            print("sbatch file created: "+bash_script_path)
            print("submission python file: "+python_script_path)

        if create_info_check:
            info_path = os.path.join(self.script_save_path, 'convert_info.txt')
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
                
            if add_cross_check:
                add_cross = 1
            else:
                add_cross = 0     
            
            with open (info_path, 'w') as f:
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
                    f"add_cross = {add_cross}\n"
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
       
        
    def _activate(self):
        print('*********************************************************************************')
        print('Make sure that <Submit checkbox> is unchecked before changing any other variables')
        print('*********************************************************************************')

        
        st = {"description_width": "initial"}
        year = Text(description='Year:', style=st)
        session = Text(description='Session:', style=st)
        subfolder_check = Checkbox(values=False, description="All MIB files in 'Merlin' folder", style=st)
        subfolder = Text(description='Subfolder:', style=st)

        no_reshaping = Checkbox(values=False, description='No reshaping', style=st)
        use_fly_back = Checkbox(value=True, description='Use Fly-back', style=st)
        known_shape = Checkbox(value=False, description='Known_shape', style=st)
        Scan_X = IntText(description='Scan_X: (avaiable for known_shape)', style=st)
        Scan_Y = IntText(description='Scan_Y: (avaiable for known_shape)', style=st)

        add_cross_check = Checkbox(value=True, description='add_cross', style=st)

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
        
        create_batch_check = Checkbox(value=False, description='Create slurm batch file', style=st)
        create_info_check = Checkbox(value=False, description='Create conversion info file', style=st)
        submit_check = Checkbox(value=False, description='Submit the job using slurm', style=st)
        
        self.values = ipywidgets.interact(self._organize, 
                                          year=year, 
                                          session=session,
                                          subfolder_check=subfolder_check,
                                          subfolder=subfolder, 
                                          no_reshaping=no_reshaping, 
                                        use_fly_back=use_fly_back, 
                                          known_shape=known_shape, 
                                          Scan_X=Scan_X, 
                                          Scan_Y=Scan_Y,
                                          add_cross_check=add_cross_check,
                                        ADF_check=ADF_check, 
                                          iBF_check=iBF_check, 
                                          DPC_check=DPC_check,
                                        bin_nav_widget=bin_nav_widget, 
                                          bin_sig_widget=bin_sig_widget,
                                        create_batch_check=create_batch_check, 
                                          create_info_check=create_info_check,
                                        submit_check=submit_check)



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