#!/usr/bin/env bash
#SBATCH --partition=cs05r
#SBATCH --job-name=data_reformat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=0G
#SBATCH --output=/$3/%j.out
#SBATCH --error=/$3/%j.error

module load python/epsic3.10

python /dls/science/groups/e02/Frederick/epsic_tools_2025/epsic_tools/epsic_tools/mib2hdfConvert/E01/Convert_data_inital.py $1 $2


