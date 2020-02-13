#!/bin/bash
#$ -l h_rt=30:00:00
#$ -q high.q@@com13
#$ -l redhat_release=rhel7
#$ -l m_mem_free=100G
#$ -o /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_ptypy
#$ -e /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs/logs_ptypy
#$ -N epsic_ptypy

module load ptypy/latest

mpiexec -n 20 python $1
