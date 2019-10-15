#!/usr/bin/env python
'''
perform ePSIC 4DSTEM analysis on the DLS cluster
'''

import os
import sys
sys.path.append('/dls_sw/e02/scripts/Merlin-Medipix/ProcessingTools')
import IdentifyHDF5_files as IH5
#maximum number of jobs to run concurrently
max_c = 10

beamline = sys.argv[1]
year = sys.argv[2]
visit = sys.argv[3]
#folder = sys.argv[4]

hdf5_dict = IH5.get_HDF5_files(beamline, year, visit, folder = None)

expt_path = hdf5_dict['processing_path']
print(expt_path)
n_files = len(hdf5_dict['HDF5_files'])
print('number of files : ', n_files)
processing_dir = os.path.join(expt_path, 'logs')
print(processing_dir)
if os.path.exists(processing_dir) == False:
    os.makedirs(processing_dir)

os.system('echo /dls_sw/e02/scripts/Merlin-Medipix/ProcessingTools/batch_process_STEM.sh ' + beamline + ' ' + year + ' ' + visit)
os.system('cd ' + processing_dir + '\n module load global/cluster \n qsub -t 1-' + str(n_files) +  ' -tc ' + str(mac_c) + ' /dls_sw/e02/scripts/Merlin-Medipix/ProcessingTools/batch_process_STEM.sh ' + beamline + ' ' + year + ' ' + visit)
