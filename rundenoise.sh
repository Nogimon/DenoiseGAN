#!/bin/bash

#SBATCH --partition=imlab-gpu

#SBATCH --time=30:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1

#SBATCH --job-name=gantry

#SBATCH --output="denoise.%j.%N.out"

#SBATCH --mail-type=ALL

#SBATCH --mail-user=lid315@lehigh.edu

cd /home/lid315/superresolution/denoise

module load python/cse498

python ./main_ocmtry.py > output_denoise.out