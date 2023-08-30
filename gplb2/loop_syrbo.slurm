#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name="AI Feynmann"                                                                                                               
#SBATCH -o gp.%J.out                                                                                                                  
#SBATCH -e gp.%J.err                                                                                                                  
#SBATCH -t 04:00:00
#SBATCH -p shared
#SBATCH -c 1
module load python
source //nobackup/dcjk57/feyn/bin/activate
#basedir=/home/dcjk57/benchmark-data/
#basedir=//nobackup/dcjk57/test-sets/
basedir=//nobackup/dcjk57/Test_Data/*/
for file in ${basedir}*_train*.txt; do
    srun -N 1 -t 04:00:00 -p shared -o output/$(basename ${file}).%J.out python3 syrbo.py -d $(dirname ${file}) -f $(basename ${file})
done

deactivate
