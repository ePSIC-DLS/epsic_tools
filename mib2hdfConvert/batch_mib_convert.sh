#!/bin/bash
#$ -l h_rt=48:00:00
#$ -cwd
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=200G

echo "I am task $SGE_TASK_ID"

module load python/epsic3.7
python /dls/science/groups/e02/Mohsen/code/Git_Repos/Merlin_Medipix/mib2hdfConvert/mib2hdf_watch_convert.py $1 $2 $3 $4 "$SGE_TASK_ID"
