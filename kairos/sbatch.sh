#!/bin/bash
# Submit a job script with the settings from kairos/config.sh (account, partition, GPU, log location).
# Usage: kairos/sbatch.sh [sbatch options] kairos/jobs/<name>.sbatch [job arguments]
# Options given here override both the defaults below and the #SBATCH lines of the script.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
mkdir -p "$IA3D_LOGS"
exec sbatch --account="$IA3D_ACCOUNT" --partition="$IA3D_PARTITION" --gres="$IA3D_GRES" \
  --chdir="$IA3D_REPO" --output="$IA3D_LOGS/%x-%j.out" --export=ALL "$@"
