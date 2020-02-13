#!/bin/bash
#$ -l h_rt=30:00:00
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=10G
#$ -o /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_savefig
#$ -o /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_savefig

module load ptypy/latest

python save_recon_output.py $1