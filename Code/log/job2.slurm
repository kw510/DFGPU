#!/bin/bash

# set the account to be used for the job
#SBATCH --account=free

## batch-acc partition contains all the accelerator nodes
#SBATCH --partition=batch-acc
 
## Requesting NVIDIA k20x nodes with n cards
#SBATCH --constraint=p100 # Could also use 'p100'
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err
 
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
 
# set max wallclock time
#SBATCH --time=00:00:30

module purge
module load cudnn

nvcc tests.cu

# run the application
./a.out