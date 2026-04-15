#!/bin/bash
#SBATCH --job-name=hmds_fish
#SBATCH --partition=debug
#SBATCH --array=1-7
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/fish%a_%j.out
#SBATCH --error=logs/fish%a_%j.err

# --- Environment ---
source /home/wenlab/miniconda3/etc/profile.d/conda.sh
conda activate /home/wenlab/miniconda3/envs/pytorch1.9.1/hmds

# --- Run ---
cd /home/wenlab/wenquan/BayesianHMDS
mkdir -p logs results

python run_fish_trials.py --fish "$SLURM_ARRAY_TASK_ID" --dim 6 --trials 10
