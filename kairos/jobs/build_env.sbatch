#!/bin/bash
#SBATCH --job-name=ia3d-build-env
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Build the project venv on top of the container (aarch64, offline) and check it on the GPU.
# Before: kairos/sbatch.sh kairos/jobs/probe.sbatch, then kairos/env/fetch.sh on the login node.
# Set IA3D_REBUILD=1 to delete the venv first.
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
hostname; nvidia-smi -L; date
apptainer exec --bind /tmpdir,/work --nv "$IA3D_CONTAINER" bash "$IA3D_REPO/kairos/env/build_in_container.sh"
status=$?
date
exit $status
