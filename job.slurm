#!/bin/bash
#SBATCH --job-name=transform_normalisation_lr_adjust        # name your job
#SBATCH --output logs/result/%x-%j.out    # output
#SBATCH --error logs/result/%x-%j.err    # error
#SBATCH --nodes=1                   # how many nodes you want ?
#SBATCH --time=24:00:00             # how long your task will last
#SBATCH --partition=gpu             # which queue you want to get in
#SBATCH --mem=16G                   # how many RAM you will need
#SBATCH --gres=gpu:v100:1           # specific filtering
source venv/bin/activate
python test_subset.py

# Job submission command:
# sbatch job.slurm

# Check job status command:
# squeue -u fwei

# Cancel job command:
# scancel <job_id>