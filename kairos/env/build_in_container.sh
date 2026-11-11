#!/bin/bash
# Runs INSIDE the container on a GPU node (called by kairos/jobs/build_env.sbatch). Offline: installs only
# from the wheelhouse and from the sources that kairos/env/fetch.sh downloaded on the login node.
# The container image is not touched; everything goes into the venv at $IA3D_VENV.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
HERE=$IA3D_REPO/kairos/env
mkdir -p "$IA3D_LOGS"

export PIP_NO_INDEX=1 PIP_FIND_LINKS=$IA3D_WHEELHOUSE PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

step() { echo; echo "### $*"; }
# run a build quietly, keep the full log, show its tail if it fails
logged() {
  local log=$IA3D_LOGS/build-$1.log; shift
  if "$@" > "$log" 2>&1; then echo "ok   (log: $log)"; else echo "FAILED (log: $log)"; tail -n 60 "$log"; exit 1; fi
}

step "venv on top of the container's site-packages: $IA3D_VENV"
if [ "${IA3D_REBUILD:-0}" = 1 ]; then rm -rf "$IA3D_VENV"; fi
[ -x "$IA3D_VENV/bin/python" ] || PYTHONNOUSERSITE=1 python3 -m venv --system-site-packages "$IA3D_VENV"
source "$HERE/activate.sh"
python --version; python -m pip --version

step "wheels"
logged wheels python -m pip install -r "$HERE/requirements.txt"
logged wheels-nodeps python -m pip install --no-deps -r "$HERE/requirements-nodeps.txt"

step "PyMCubes (Cython extension, compiled against the container's numpy)"
logged pymcubes python -m pip install --no-build-isolation --no-deps -v \
  "$(grep -i '^PyMCubes' "$HERE/requirements-source.txt")"

step "pysdf (pybind11 extension)"
# Its setup.py looks for Eigen in a few fixed places and otherwise downloads it. Offer the copy fetched on the
# login node through one of those places (~/.local/include/eigen3) with a throw-away HOME.
EIGEN=$(ls -d "$IA3D_SRC"/eigen-3.3.* | head -n 1)
FAKE_HOME=$(mktemp -d)
mkdir -p "$FAKE_HOME/.local/include"
ln -s "$EIGEN" "$FAKE_HOME/.local/include/eigen3"
HOME=$FAKE_HOME logged pysdf python -m pip install --no-build-isolation --no-deps -v \
  "$(grep -i '^pysdf' "$HERE/requirements-source.txt")"
rm -rf "$FAKE_HOME"

step "tiny-cuda-nn PyTorch bindings ($(git -C "$IA3D_SRC/tiny-cuda-nn" rev-parse --short HEAD 2>/dev/null || echo '?'))"
# The compute capability of the node's GPU is detected through torch (GH200: 90).
logged tinycudann python -m pip install --no-build-isolation --no-deps -v "$IA3D_SRC/tiny-cuda-nn/bindings/torch"

step "packages added by the venv"
python -m pip list --local --format=freeze | tee "$IA3D_LOGS/venv-freeze.txt"

step "checks"
cd "$IA3D_REPO"
python "$HERE/check_env.py"
echo BUILD_ENV_DONE
