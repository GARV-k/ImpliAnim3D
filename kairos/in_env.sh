#!/bin/bash
# Run a command inside the container with the project venv active, from the repo root.
# Only works on a GPU node (the container is aarch64), i.e. inside an sbatch job or an salloc session.
# Usage: kairos/in_env.sh python train_GT2.py --help
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
[ -e "$IA3D_VENV/bin/activate" ] || { echo "no venv at $IA3D_VENV: run kairos/sbatch.sh kairos/jobs/build_env.sbatch first" >&2; exit 1; }
[ $# -gt 0 ] || { echo "usage: kairos/in_env.sh <command> [arguments]" >&2; exit 2; }
# Without the checks below, a repo or stage directory that the container cannot see would let the command
# run with the image's own Python, outside the venv and with TF32 on.
exec apptainer exec --bind /tmpdir,/work --nv "$IA3D_CONTAINER" bash -c '
  source "$IA3D_REPO/kairos/env/activate.sh" \
    || { echo "in_env: cannot activate $IA3D_VENV from $IA3D_REPO inside the container (it binds /work and /tmpdir)" >&2; exit 1; }
  cd "$IA3D_REPO" || exit 1
  exec "$@"' in_env "$@"
