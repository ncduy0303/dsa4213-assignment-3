#!/bin/bash

#SBATCH --job-name=JOB_NAME_HERE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --output=./logs/JOB_NAME_HERE_%j.out.log
#SBATCH --error=./logs/JOB_NAME_HERE_%j.err.log

# --- Instructions ---
# 1. Get API key from https://wandb.ai/authorize
# 2. Initialize a sweep: `uv run wandb sweep <path_to_sweep_config.yaml>`
# 3. Paste the SWEEP ID given by the command below.
# 4. REMEMBER TO CHANGE THE JOB NAME AND OUTPUT FILES ABOVE!
# 5. Submit the job using `sbatch train.sh`

# Get API key from https://wandb.ai/authorize
export WANDB_API_KEY="WANDB_API_KEY_HERE"

export WANDB_PROJECT="dsa4213-assignment-3"

# Get SWEEP ID from the wandb sweep command
SWEEP_ID="ncduy0303/dsa4213-assignment-3/<SWEEP_ID_HERE>"

# --- Environment Setup ---
echo "Job is running on $(hostname), started at $(date)"
nvidia-smi 

# --- Run the W&B Agent ---
# The agent will automatically connect to the sweep controller,
# get a set of hyperparameters, and run the training script.
echo "Starting W&B agent for sweep: ${SWEEP_ID}"
uv run wandb agent "${SWEEP_ID}"
