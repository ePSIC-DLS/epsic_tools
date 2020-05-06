#!/bin/bash
#$ -l h_rt=2:00:00
#$ -cwd
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=200G

module load python/3.7
python /home/gys37319/code/Merlin-Medipix/ProcessingTools/post_processing.py e02 2019 mg24610-2
