#!/bin/bash
#$ -l h_rt=30:00:00
#$ -q high.q@@com13
#$ -l redhat_release=rhel7
#$ -l m_mem_free=100G
#$ -o /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_ptyREX
#$ -e /dls/science/groups/e02/Mohsen/code/Git_Repos/My_Repository/logs/logs_ptyREX
#$ -N epsic_ptypy

module load pycho

pycho_recon_single_cluster $1 $2 $3