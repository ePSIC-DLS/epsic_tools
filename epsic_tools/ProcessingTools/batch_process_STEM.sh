#!/bin/bash
#$ -l h_rt=48:00:00
#$ -cwd
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=200G
echo "I am task $SGE_TASK_ID"

module load python/3.7
python /home/gys37319/code/Merlin-Medipix/ProcessingTools/batch_post_processing.py e02 2019 mg24610-2 "$SGE_TASK_ID" "test"
