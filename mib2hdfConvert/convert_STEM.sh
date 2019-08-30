#!/bin/bash
#$ -l h_rt=0:20:00
#$ -cwd
#$ -q high.q
#$ -l redhat_release=rhel7
#$ -l m_mem_free=200G

module load python/3.7
python /home/eha56862/code/DLS_cluster_conversion_dask/mib2hdf_watch_convert.py e02 2019 cm22979-3 'Merlin/20190703_300kV_PencilBeam_MD'
