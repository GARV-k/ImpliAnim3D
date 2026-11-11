#!/bin/bash
#SBATCH --job-name=ia3d-smoke
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# End-to-end smoke test of the pipeline on a few ShapeNet meshes. Small and short on purpose: it shows that
# every stage runs on this stack and produces sane output, not what the method achieves.
#   1. train_GT2.py      shared decoder + one hash table per mesh (each mesh unrotated and rotated by 90 degrees)
#   2. test.py           hash table of an unseen mesh, decoder frozen
#   3. train_weights.py  W on the rotated meshes, tables and decoder frozen
#   4. inference.py      meshes decoded from the W-transformed tables
#   5. verify_smoke.py   measures the fitted fields and reads the losses back
# Usage: kairos/sbatch.sh kairos/jobs/smoke_e2e.sbatch [set name]        (default: shapenet_rr60)
# Tunables (environment): SMOKE_NUM_OBJS, SMOKE_EPOCHS, SMOKE_UNSEEN_EPOCHS, SMOKE_W_EPOCHS, SMOKE_LR
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
set -euo pipefail

SET=$IA3D_DATA/sets/${1:-shapenet_rr60}
RUN=$IA3D_RUNS/smoke-${SLURM_JOB_ID:-manual}
N=${SMOKE_NUM_OBJS:-3}
EPOCHS=${SMOKE_EPOCHS:-10}                # checkpoints are written on even epochs
UNSEEN_EPOCHS=${SMOKE_UNSEEN_EPOCHS:-20}  # the frozen-decoder fit waits on a plateau for a random number of steps
W_EPOCHS=${SMOKE_W_EPOCHS:-2}
LR=${SMOKE_LR:-1e-3}                      # the scripts default to 1e-4, which needs far more steps
[ -f "$SET/manifest.json" ] || { echo "mesh set $SET not found: run kairos/data/make_legacy_set.py" >&2; exit 1; }
mkdir -p "$RUN"

stage() {  # stage <title> <command...>: runs the command in the project environment and reports its duration
  local title=$1 t0=$SECONDS; shift
  echo; echo "### $title"; echo "+ $*"
  "$IA3D_REPO/kairos/in_env.sh" "$@"
  echo "### done in $((SECONDS - t0)) s"
}

hostname; nvidia-smi -L; date
echo "set: $SET   run: $RUN   meshes: 1..$N (+ rotated), unseen: 40"
stage "0. environment check" python kairos/env/check_env.py
stage "1. shared-decoder training" python -u train_GT2.py --data-root "$SET/obj_meshes" --workspace "$RUN/workspace" \
  --num-objs "$N" --epochs "$EPOCHS" --lr "$LR"
stage "2. unseen mesh, decoder frozen" python -u test.py --data-root "$SET/obj_meshes" --workspace "$RUN/workspace" \
  --test-workspace "$RUN/unseen" --epochs "$UNSEEN_EPOCHS" --lr "$LR"
stage "3. W on the rotated meshes" python -u train_weights.py --data-root "$SET/obj_meshes" --workspace "$RUN/workspace" \
  --num-objs "$N" --epochs "$W_EPOCHS"
stage "4. meshes through W" python -u inference.py --workspace "$RUN/workspace" --num-objs "$N" \
  --out "$RUN/inference" --resolution 64
stage "5. verification" python -u kairos/jobs/verify_smoke.py --set "$SET" --run "$RUN" --num-objs "$N"
date
echo SMOKE_E2E_OK
