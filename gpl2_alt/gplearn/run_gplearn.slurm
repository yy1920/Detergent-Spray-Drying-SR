#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name="AI Feynmann"                                                                                                               
#SBATCH -o aif.%J.out                                                                                                                  
#SBATCH -e aif.%J.err                                                                                                                  
#SBATCH -t 04:00:00
#SBATCH -p shared
#SBATCH -c 32
module load python
source //nobackup/dcjk57/feyn/bin/activate
#python3 gplearn_ex.py //nobackup/dcjk57/test-sets/I_13_12_train.txt
python3 run_gplearn.py
deactivate
