# KAIROS settings for ImpliAnim3D. Source this file; every value can be overridden from the environment.
#
# Login nodes are x86_64 with Internet access; the GH200 GPU nodes are aarch64 without it. The shared
# container is aarch64, so anything that executes it has to run inside a Slurm job. The container is
# never modified: the project environment is a venv layered on top of it (see kairos/README.md).

# Repo root = parent of this directory. The container only binds /work, so /gpfs/work is rewritten.
if [ -z "${IA3D_REPO:-}" ]; then
  IA3D_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
IA3D_REPO=${IA3D_REPO/#\/gpfs\/work\//\/work\/}

# Slurm
: "${IA3D_ACCOUNT:=${SBATCH_ACCOUNT:-$(id -gn)}}"
: "${IA3D_PARTITION:=shared-gpu}"
: "${IA3D_GRES:=gpu:gh200:1}"

# Container (read-only, shared)
: "${IA3D_CONTAINER:=/work/conteneurs/shared/AI/nemo_25.04.03_arm.sif}"

# Stage directory: everything that is not source code (venv, wheels, third-party sources, data, logs, runs).
# It sits outside the repo so that every checkout and worktree of the repo shares it.
: "${IA3D_STAGE:=${WORK:?WORK is not set}/irr/ImpliAnim3D-stage}"
IA3D_STAGE=${IA3D_STAGE/#\/gpfs\/work\//\/work\/}
: "${IA3D_VENV:=$IA3D_STAGE/venv}"
: "${IA3D_WHEELHOUSE:=$IA3D_STAGE/wheelhouse}"
: "${IA3D_SRC:=$IA3D_STAGE/src}"
: "${IA3D_LOGS:=$IA3D_STAGE/logs}"
: "${IA3D_RUNS:=$IA3D_STAGE/runs}"
: "${IA3D_DATA:=$IA3D_STAGE/data}"

export IA3D_REPO IA3D_ACCOUNT IA3D_PARTITION IA3D_GRES IA3D_CONTAINER IA3D_STAGE \
       IA3D_VENV IA3D_WHEELHOUSE IA3D_SRC IA3D_LOGS IA3D_RUNS IA3D_DATA
