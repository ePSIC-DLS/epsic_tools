#!/bin/bash
#$ -N pyprismatic_epsic
#$ -l m_mem_free=1G,h_rt=24:00:00,gpu=4,exclusive
#$ -pe openmpi 40
#$ -P e02
#$ -cwd


module load python/3.8
conda activate /dls/science/groups/e02/Mohsen/code/dev_prismatic_28042021/dev_env_3.8

python /dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin-Medipix/epsic_tools/toolbox/sim_class.py $1
