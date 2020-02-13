#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:19:23 2020

@author: eha56862
This script gets the folder path with a sim matrix and does the following:
     - checks which folders have seen no action at all, i.e. only contain two files - .h5 data file and .txt param file
     - runs the data_input_prep_ptypy_and_pycho.py on the folder above to prepare them for recons
     - submits ptypy and ptyREX jobs from these folders 
"""

import os
import sys
import numpy as np
import data_input_prep_ptypy_and_pycho
import argparse
import scandir


def get_raw_dir_list(sim_matrix_path, get_all = False):
    '''
    checks for the folders with only two files and identify them as raw
    get_all set to True returns all the folders 
    '''
    raw_dirs = []
    it =  scandir.scandir(sim_matrix_path)
    if get_all:
        for entry in it:
            if entry.is_dir():
                raw_dirs.append(entry.path)
    else:
        for entry in it:
            if entry.is_dir():
     #           if 'recons' not in os.listdir(os.path.dirname(entry.path)): 
                if len(os.listdir(entry.path)) == 2:
                    raw_dirs.append(entry.path)
    return raw_dirs

def get_ptypy_ready(sim_matrix_path):
    '''
    checks for the folders that have ptypy_recon script 
    '''
    ptypy_dirs = []
    it = scandir.scandir(sim_matrix_path)
    for entry in it:
        if entry.is_dir():
            it2 = scandir.scandir(entry.path)
            for entry2 in it2:
                if entry2.is_file():
                    if entry2.name.startswith('ptypy_recon_'):
                        ptypy_dirs.append(entry.path)
    return ptypy_dirs
def get_ptyREX_ready(sim_matrix_path):
    '''
    checks for the folders that have ptypy_recon script 
    '''
    ptyREX_dirs = []
    it =  scandir.scandir(sim_matrix_path)
    for entry in it:
        if entry.is_dir():
            it2 = scandir.scandir(entry.path)
            for entry2 in it2:
                if entry2.is_file():
                    if entry2.name.startswith('ptyREX_'):
                        ptyREX_dirs.append(entry.path)
    return ptyREX_dirs
    

def main(sim_matrix_path):
    script_path = '/dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/prep_run_ptypy_ptyREX'
#    dirs_to_prep = get_raw_dir_list(sim_matrix_path, get_all = True)
#    for path in dirs_to_prep:
#        it = scandir.scandir(path)
#        for entry in it:
#            if entry.is_file():
#                if entry.name.startswith('params'):
#                    os.system('\n module load global/cluster \n qsub '+ script_path + 'data_prep_submit.sh '+ entry.path + ' ptypy=True ptyrex=False')
    dirs_to_run_ptypy = get_ptypy_ready(sim_matrix_path)
    for path in dirs_to_run_ptypy:
        it = scandir.scandir(path)
        for entry in it:
            if entry.is_file():
                if entry.name.startswith('ptypy_recon_'):
                    # avoid the folders that have run ptypy already
                    if 'recons' not in os.listdir(os.path.dirname(entry.path)): 
                        output_folder = os.path.dirname(entry.path)
                        os.system('\n cd '+ output_folder + '\n module load global/cluster \n qsub '+ script_path + 'ptypy_batch_submit.sh '+ entry.path)
    dirs_to_run_ptyREX = get_ptyREX_ready(sim_matrix_path) 
    for path in dirs_to_run_ptyREX:
        it = scandir.scandir(path)
        for entry in it:
            if entry.is_file():
                if entry.name.startswith('ptyREX'):
                    output_folder = os.path.dirname(entry.path)
                    json_file = os.path.splitext(entry.name)[0]
                    os.system('\n cd '+ output_folder + '\n module load global/cluster \n qsub '+ script_path + 'ptyREX_batch_submit.sh '+ output_folder + ' ' + json_file + ' 12022020')
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_matrix_path', help='path containing all the simulated sets')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()

    main(args.sim_matrix_path)
