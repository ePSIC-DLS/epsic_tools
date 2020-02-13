#!/bin/bash
#$ -l h_rt=30:00:00
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=100G
#$ -o /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_prep
#$ -e /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_prep

module load ptypy/latest

python data_input_prep_ptypy_and_pycho.py $1 $2 $3
