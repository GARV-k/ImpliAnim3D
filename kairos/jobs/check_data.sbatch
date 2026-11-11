#!/bin/bash
#SBATCH --job-name=ia3d-check-data
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# Data checks that need the project environment (see kairos/data/README.md):
#   1. the mesh set: watertight meshes inside the unit ball, exact rotations, SDF of the rotated mesh = rotated SDF
#   2. the regeneration recipe: one model preprocessed again from raw ShapeNet and compared with the linked data
# Usage: kairos/sbatch.sh kairos/jobs/check_data.sbatch [set name]        (default: shapenet_rr60)
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
set -euo pipefail
SET=$IA3D_DATA/sets/${1:-shapenet_rr60}
SYNSET=02691156
SCRATCH=$IA3D_STAGE/tmp/regen-check-${SLURM_JOB_ID:-manual}
hostname; date

echo "### 1. mesh set $SET"
"$IA3D_REPO/kairos/in_env.sh" python -u kairos/data/check_set.py "$SET"

echo; echo "### 2. regenerate one model of $SYNSET and compare with the linked data"
mkdir -p "$SCRATCH"
"$IA3D_REPO/kairos/in_env.sh" python -u kairos/data/regen/preprocess_sdf.py --all --limit 1 --workers 1 \
  --shapenet-dir "$IA3D_DATA/raw/ShapeNetCore.v1/extracted_200/ShapeNetCore.v1/$SYNSET" --out "$SCRATCH/$SYNSET"
"$IA3D_REPO/kairos/in_env.sh" python -u kairos/data/regen/compare_regen.py "$SCRATCH/$SYNSET" "$IA3D_DATA/shapenet/$SYNSET"
rm -rf "$SCRATCH"
date
echo CHECK_DATA_OK
