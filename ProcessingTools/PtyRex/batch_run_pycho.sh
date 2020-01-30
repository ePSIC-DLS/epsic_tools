#!/bin/bash
#$ -l h_rt=48:00:00
#$ -cwd
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=200G

echo "I am task $SGE_TASK_ID"


module load pycho
python /dls_sw/e02/scripts/pycho_cluster/pycho_recon_array_cluster $1 "$SGE_TASK_ID" 
