#!/bin/bash
#BSUB -J run_smm_cv
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 04:00
#BSUB -o hpc/fred_false_cv_smm_%J.out
#BSUB -e hpc/fred_false_cv_smm_%J.err

# Change this to the directory where you copied the repo on DTU HPC.
cd /zhome/8c/6/163231/Algorithms_in_bioinformatics_project

module load python3/3.10.13

python3 run_cv.py --data_dir Data --alleles all \
        --lambdas 0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30 \
        --epsilons 0.01,0.05,0.1 --epochs 200 --jobs 4
