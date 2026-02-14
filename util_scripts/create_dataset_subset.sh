#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=create_dataset_subset
#SBATCH --partition long-cpu
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/scratch/git/open_clip/util_scripts/%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/scratch/git/open_clip/util_scripts/%j_%t.err


source ~/.bashrc
mamba activate openclip
module load cuda/11.8


python create_dataset_subset.py
