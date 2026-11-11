#!/bin/bash
#SBATCH --job-name=ia3d-pp
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# Regenerate the preprocessed ShapeNet subset (watertight shell meshes + SDF samples) for a list of synsets.
# CPU-only work, but it runs on a GPU node: the container and the venv are aarch64, the CPU partitions x86_64.
# Usage: kairos/sbatch.sh kairos/data/regen/preprocess_subset.sbatch "<synset> [<synset> ...]"     ("all" = every extracted synset)
# Input:  $IA3D_DATA/raw/ShapeNetCore.v1/extracted_200/ShapeNetCore.v1/<synset>/<id>/model.obj
# Output: $IA3D_DATA/shapenet/<synset>/{samples,meshes}/. A synset that is a symlink (linked data) is skipped,
#         and shapes that already have samples are not recomputed.
# The renders of the MeshSDF setup are not produced: only its single-view reconstruction used them.
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
SYNSETS=${1:?synset list, or all}
EXT=${IA3D_SHAPENET_EXTRACTED:-$IA3D_DATA/raw/ShapeNetCore.v1/extracted_200/ShapeNetCore.v1}
OUT=${IA3D_SHAPENET_OUT:-$IA3D_DATA/shapenet}
[ "$SYNSETS" = all ] && SYNSETS=$(ls "$EXT")
hostname; nproc; date
export OMP_NUM_THREADS=5
for syn in $SYNSETS; do
  if [ -L "$OUT/$syn" ]; then echo "### synset $syn: $OUT/$syn is a symlink to existing data, skipped"; continue; fi
  echo "### synset $syn: $(ls "$EXT/$syn" | wc -l) models; shells + SDF samples ($(date))"
  "$IA3D_REPO/kairos/in_env.sh" python -u kairos/data/regen/preprocess_sdf.py --shapenet-dir "$EXT/$syn" --all \
    --out "$OUT/$syn" --workers 12 2>&1 | grep -vE "^\[.*\] [0-9a-f]+ ok " | tail -n 100
  echo "SYNSET_SUMMARY $syn samples=$(ls "$OUT/$syn/samples" | wc -l) meshes=$(ls "$OUT/$syn/meshes" | wc -l)"
done
date; echo PP_DONE
