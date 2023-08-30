#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name="SyRBo"
#SBATCH -o syrbo.%J.out
#SBATCH -e syrbo.%J.err
# #SBATCH -t 04:00:00
#SBATCH -p test
#SBATCH -c 32
module load python
source //nobackup/dcjk57/feyn/bin/activate
#python3 syrbo.py -d //nobackup/dcjk57/test-sets/ -f I_13_12_train.txt
python3 syrbo.py
deactivate
