#!/bin/bash
#$ -P e02
#$ -N pyprismatic_epsic
#$ -l gpu_arch=Pascal
#$ -l exclusive
#$ -l m_mem_free=2G
#$ -q high.q
#$ -l gpu=2
#$ -cwd

module load python/3.8
conda activate /dls/science/groups/e02/Mohsen/code/dev_prismatic_28042021/dev_env_3.8

python /dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin-Medipix/epsic_tools/toolbox/sim_class.py $1
