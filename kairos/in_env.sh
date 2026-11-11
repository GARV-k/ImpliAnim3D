#!/bin/bash
# Run a command inside the container with the project venv active, from the repo root.
# Only works on a GPU node (the container is aarch64), i.e. inside an sbatch job or an salloc session.
# Usage: kairos/in_env.sh python train_GT2.py --help
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
[ -e "$IA3D_VENV/bin/activate" ] || { echo "no venv at $IA3D_VENV: run kairos/sbatch.sh kairos/jobs/build_env.sbatch first" >&2; exit 1; }
exec apptainer exec --bind /tmpdir,/work --nv "$IA3D_CONTAINER" bash -c '
  source "$IA3D_REPO/kairos/env/activate.sh"
  cd "$IA3D_REPO"
  exec "$@"' in_env "$@"
