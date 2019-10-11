# Merlin-Medipix
Code for processing to be performed on converted Merlin-Medipix datasets

The process_STEM.sh file should be modified to point to the local code location and session number.

This can then be run on the cluster by navigating to the location of the .sh file and running:

	module load global/cluster
	qsub process_STEM.sh
	
The processing_params.txt sets what processing to perform and must exist in the session processing directory 
To determine the parameters for finding the bright field disk diameter, first run the get_bdf.ipynb notebook. 
