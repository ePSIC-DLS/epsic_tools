"""
This script is to perform batch parallel mib to hdf5 conversion on the DLS cluster

Example of use in DLS Linux terminal:
    python batch_convert_cluster.py e02 2019 mg25124-2
or with a specific folder option:
    python batch_convert_cluster.py e02 2019 cm22979-6 Merlin/20191022_hot_graphene

The user just needs to have this python script.
A logs is created in the saving path , the processing folder of the visit, with the 
outputs of the cluster jobs and also a list of the files to convert.

In order to monitor the status of jobs, user needs to run:
    module load global/cluster
    watch qstat

"""

import os
import sys
import numpy as np
import argparse
#sys.path.append('/dls_sw/e02/scripts/batch_mib_convert')
from IdentifyPotentialConversions import check_differences
#maximum number of jobs to run concurrently
#max_c = 10
#
#beamline = sys.argv[1]
#year = sys.argv[2]
#visit = sys.argv[3]
#
#if len(sys.argv) < 5:
#    folder = None
#else:
#    folder = sys.argv[4]
#
#mib_data_dict = check_differences(beamline, year, visit, folder)
#
#saving_path = mib_data_dict['processing_path']
#
#n_files = len(mib_data_dict['MIB_to_convert'])
#print('number of MIB files to convert to HDF5 : ', n_files)
#outputs_dir = os.path.join(saving_path, 'logs')
#
#if os.path.exists(outputs_dir) == False:
#    os.makedirs(outputs_dir)
#    
#with open(outputs_dir + '/file_numbers.txt', 'w') as f:
#    for i in np.arange(n_files):
#        f.write(str(i) + ' : ' + mib_data_dict['MIB_to_convert'][i] + '\n')
#        f.write('\n')
#
#if len(sys.argv) < 5:
#    folder = 'False'
#else:
#    folder = sys.argv[4]
#
#os.system('\n cd ' + outputs_dir + '\n module load global/cluster \n qsub -t 1-' + str(n_files) +  ' -tc ' + str(max_c) + ' /dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin-Medipix/epsic_tools/mib2hdfConvert/batch_mib_convert.sh ' + beamline + ' ' + year + ' ' + visit+ ' '+ folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('beamline', help='Beamline name')
    parser.add_argument('year', help='Year')
    parser.add_argument('visit', help='Session visit code')
    parser.add_argument('-folder', default=None, help='Option to add folder')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()

    max_c = 10



    mib_data_dict = check_differences(args.beamline, args.year, args.visit, args.folder)
    saving_path = mib_data_dict['processing_path']

    n_files = len(mib_data_dict['MIB_to_convert'])
    print('number of MIB files to convert to HDF5 : ', n_files)
    outputs_dir = os.path.join(saving_path, 'logs')

    if os.path.exists(outputs_dir) == False:
        os.makedirs(outputs_dir)
    
    with open(outputs_dir + '/file_numbers.txt', 'w') as f:
        for i in np.arange(n_files):
            f.write(str(i) + ' : ' + mib_data_dict['MIB_to_convert'][i] + '\n')
            f.write('\n')

#if len(sys.argv) < 5:
#    folder = 'False'
#else:
#    folder = sys.argv[4]
    if args.folder is None: 
        os.system('\n cd ' + outputs_dir + '\n module load global/cluster \n qsub -t 1-' + str(n_files) +  ' -tc ' + str(max_c) + ' /dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin-Medipix/epsic_tools/mib2hdfConvert/batch_mib_convert.sh ' + args.beamline + ' ' + args.year + ' ' + args.visit)
    else:
        os.system('\n cd ' + outputs_dir + '\n module load global/cluster \n qsub -t 1-' + str(n_files) +  ' -tc ' + str(max_c) + ' /dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin-Medipix/epsic_tools/mib2hdfConvert/batch_mib_convert.sh ' + args.beamline + ' ' + args.year + ' ' + args.visit+ ' ' + args.folder)

