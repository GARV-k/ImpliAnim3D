#!/bin/bash
#SBATCH --job-name=ia3d-run
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Run any command of the repo in the project environment on a GPU node, from the repo root.
# Usage: kairos/sbatch.sh [sbatch options] kairos/jobs/run.sbatch <command> [arguments]
#   kairos/sbatch.sh --time=12:00:00 --job-name=shared-dec kairos/jobs/run.sbatch \
#     python -u train_GT2.py --data-root "$IA3D_DATA/sets/shapenet_rr60/obj_meshes" --workspace "$IA3D_RUNS/exp1"
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
hostname; nvidia-smi -L; date
echo "+ $*"
"$IA3D_REPO/kairos/in_env.sh" "$@"
status=$?
date
exit $status
