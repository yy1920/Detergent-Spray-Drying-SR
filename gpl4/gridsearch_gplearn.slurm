#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name="GPL4"   
#SBATCH -o gridsearch.%J.out                                                                                                                  
#SBATCH -e gridsearch.%J.err                                                                                                                  
#SBATCH -t 04:00:00
#SBATCH -p shared
#SBATCH -c 32
module load python
source //nobackup/dcjk57/feyn/bin/activate
python3 gridsearch_gplearn.py 
deactivate
