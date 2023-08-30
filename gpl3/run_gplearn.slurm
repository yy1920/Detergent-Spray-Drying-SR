#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name="GPL3"   
#SBATCH -o gpl3.%J.out                                                                                                                  
#SBATCH -e gpl3.%J.err                                                                                                                  
#SBATCH -t 04:00:00
#SBATCH -p shared
#SBATCH -c 32
module load python
source //nobackup/dcjk57/feyn/bin/activate
python3 run_gplearn.py 
deactivate
