#!/bin/bash

#$ -P tomography
#$ -N pyprismatic_epsic
#$ -l gpu_arch=Pascal
#$ -o /dls/science/groups/e02/Mohsen/code/sim_4DSTEM/ptypy_pycho_sim_matrix/logs/logs_pyprismatic
#$ -e /dls/science/groups/e02/Mohsen/code/sim_4DSTEM/ptypy_pycho_sim_matrix/logs/logs_pyprismatic
#$ -l exclusive
#$ -q high.q
#$ -l gpu=2
#$ -cwd

module load python/epsic3.7


python sim_matrix.py
