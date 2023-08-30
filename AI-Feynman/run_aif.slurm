#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name="AI Feynmann"                                                                                                               
#SBATCH -o aif.%J.out                                                                                                                  
#SBATCH -e aif.%J.err                                                                                                                  
#SBATCH -t 04:00:00
#SBATCH -p shared
#SBATCH -c 2
module load python
source //nobackup/dcjk57/feyn/bin/activate
python3 run_aif.py -d //nobackup/dcjk57/Test_Data_Final/pg_data_reduced/ -f pg_reduced_train.txt
deactivate
