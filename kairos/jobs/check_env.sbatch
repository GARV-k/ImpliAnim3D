#!/bin/bash
#SBATCH --job-name=ia3d-check-env
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
# Check the built environment on a GPU node: every third-party package the pipeline uses, then the repo's model code.
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
hostname; nvidia-smi -L
"$IA3D_REPO/kairos/in_env.sh" python kairos/env/check_env.py
