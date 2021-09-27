#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:48:52 2021

@author: eha56862
"""
import json
# import h5py
# import pyprismatic as pr
import numpy as np
import os
import argparse

class Sim_4DSTEM:
    """
    Class for 4DSTEM simulation objects.
    This populates a parameters dict with the standard keys and adds a method
    to write to json file 
    """
    def __init__(self, params_dict={}):
        params_dict = {
            "algorithm": "m",
            "atomic_model": "/dls/science/groups/e02/Mohsen/code/Git_Repos/Staff-notebooks/ptyREX_sim_matrix/xyz_files/Graphene_defect.xyz",
            "output_path": "",
            "pixel-size-x": 0.0850488e-10,
            "pixel-size-y": 0.0850488e-10,
            "num-FP": 8,
            "slice-thickness": 8e-10,
            "energy": 80e3,
            "alpha-max": 45e-3,
            "probe-step-x": 4.34739e-10,
            "probe-step-y": 4.34739e-10,
            "tile-uc": [3, 3, 1],
            "probe-defocus": 0.0,
            "C3": 0.0,
            "C5": 0.0,
            "aberrations_file": "",
            "probe-semiangle": 25e-3,
            "probe-xtilt": 0,
            "probe-ytilt": 0,
            "scan-window-x": [0.3333, 0.6666],
            "scan-window-y": [0.3333, 0.6666],
            "random_seed": 25212,
            "thermal-effects": 1,
            "occupancy": 1,
            "save-3D-output": 0,
            "save-4D-output": 1,
            "4D-crop": 0,
            "save-DPC-CoM": 0,
            "save-potential-slices": 1,
            "save-smatrix": 0,
            "import-potential": 0,
            "import-smatrix": 0,
            "nyquist-sampling": 0,
            "also-do-cpu-work": 1,
            "script_path": "/dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin-Medipix/epsic_tools/toolbox/"
            }
        self.params_dict = params_dict
        
    def write_json(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.params_dict, f, indent = 4)
            

    def _get_cell_dims(self):
        '''
        returns the cell dimensions of an xyz atomic coordination file.
        Parameters
        ___________
        
        xyz: str
            full path of the xyz coordination file
        Returns
        ____________
        
        cell_dims: numpy array
            cell dimensions in (A)
        '''
        if os.path.exists(self.params_dict['atomic_model']):
            file = self.params_dict['atomic_model']
        else:
            return FileNotFoundError('The xyz file provided does not exist!')
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if i == 1:
                    data = np.asarray(line.split(), dtype = float)
        return data
        
        
    def run_sim(self):

        # Changing units from SI to A and mrad and kV
        xyz = self.params_dict['atomic_model']
        pixelSize = self.params_dict['pixel-size-x'] * 1e10
        num_FP = self.params_dict['num-FP']
        slice_thickness = self.params_dict['slice-thickness'] * 1e10
        E0 = self.params_dict['energy'] * 1e-3
        alpha_max = self.params_dict['alpha-max'] * 1e3
        step_size = self.params_dict['probe-step-x'] * 1e10
        cell_tiling = self.params_dict['tile-uc']
        def_val = self.params_dict['probe-defocus'] * 1e10
        conv_semiangle = self.params_dict['probe-semiangle'] * 1e3
        scan_win_x = self.params_dict['scan-window-x']
        scan_win_y = self.params_dict['scan-window-y']
        probe_tilt_x = self.params_dict['probe-xtilt'] * 1e3
        probe_tilt_y = self.params_dict['probe-ytilt'] * 1e3
        c3 = self.params_dict['C3'] * 1e10
        c5 = self.params_dict['C5'] * 1e10
        
        
        # meta = pr.Metadata(filenameAtoms = xyz)
        # meta.algorithm = 'multislice'
        # meta.filenameOutput = self.params_dict['output_path']
        # meta.realspacePixelSizeX = pixelSize
        # meta.realspacePixelSizeY = pixelSize 
        # meta.potBound = 2
        # meta.numFP = num_FP
        # meta.sliceThickness = slice_thickness
        # meta.E0 = E0
        # meta.alphaBeamMax = alpha_max
        # meta.batchSizeCPU = 1
        # meta.probeStepX = float(step_size) 
        # meta.probeStepY = float(step_size) 
        
        # cell_dims = self._get_cell_dims()
        # meta.cellDimX = cell_dims[0]
        # meta.cellDimY = cell_dims[1]
        # meta.cellDimZ = cell_dims[2]
        
        # meta.tileX = cell_tiling[0]
        # meta.tileY = cell_tiling[1]
        # meta.tileZ = cell_tiling[2]
        # meta.probeDefocus = float(def_val)
        # meta.C3 = c3
        # meta.C5 = c5
        # if 'aberrations_file' in self.params_dict:
        #     if self.params_dict['aberrations_file'] is not None:
        #         meta.aberrations_file = self.params_dict['aberrations_file']
        # meta.probeSemiangle = float(conv_semiangle) 
        # meta.detectorAngleStep = 1
        # meta.probeXtilt = probe_tilt_x
        # meta.probeYtilt = probe_tilt_y
        # meta.scanWindowXMin = scan_win_x[0]
        # meta.scanWindowXMax = scan_win_x[1]
        # meta.scanWindowYMin = scan_win_y[0]
        # meta.scanWindowYMax = scan_win_y[1]
        # meta.randomSeed = self.params_dict['random_seed']
        # meta.includeThermalEffects = self.params_dict['thermal-effects']
        # meta.save2DOutput = 0
        # meta.save3DOutput = 0
        # meta.save4DOutput = 1
        # meta.nyquistSampling = 0
        # meta.saveDPC_CoM = 0
        # meta.savePotentialSlices = self.params_dict['save-potential-slices']
        # meta.alsoDoCPUWork = 1
        # meta.batchSizeGPU = 1
        # meta.numGPUs = 2
        # meta.numStreamsPerGPU = 3
        # meta.saveProbe = 1
        # meta.maxFileSize = 5*10**9
        
        # meta.go()  
        
        
    def submit_dls_cluster(self, json_path):
        self.write_json(json_path)
        bash_path = os.path.join(self.params_dict['script_path'], 'pyprismatic_cluster.sh')
        os.system('\n module load global/cluster \n qsub ' + bash_path + ' ' + json_path)
        
    def submit_dls_hamilton(self, json_path):
        self.write_json(json_path)
        bash_path = os.path.join(self.params_dict['script_path'], 'pyprismatic_hamilton.sh')
        os.system('\n module load hamilton \n qsub ' + bash_path + ' ' + json_path)
            

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', help='path for the input parameters as json file')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                       default=False)

    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        params_dict = json.load(f)
    sim = Sim_4DSTEM()
    sim.params_dict = params_dict
    
    sim.run_sim()
