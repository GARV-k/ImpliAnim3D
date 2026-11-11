#!/bin/bash
# Login node (x86_64, Internet): download everything the offline aarch64 build needs into the stage directory.
#   wheelhouse/  aarch64 cp312 wheels for requirements.txt and their dependencies, resolved against the
#                versions already installed in the container, plus the sdists of requirements-source.txt
#   src/         tiny-cuda-nn with its submodules at the pinned commit, and the Eigen headers
# Needs $IA3D_LOGS/container-freeze.txt, written by kairos/jobs/probe.sbatch.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.sh"

TCNN_URL=https://github.com/NVlabs/tiny-cuda-nn
TCNN_COMMIT=${TCNN_COMMIT:-$(cat "$HERE/tiny-cuda-nn.commit")}
# The version pysdf's setup.py would download by itself. Its version check rejects 3.4.0 (minor version 0).
EIGEN_VERSION=3.3.7

FREEZE=$IA3D_LOGS/container-freeze.txt
[ -s "$FREEZE" ] || { echo "missing $FREEZE: run kairos/sbatch.sh kairos/jobs/probe.sbatch first" >&2; exit 1; }
mkdir -p "$IA3D_WHEELHOUSE" "$IA3D_SRC"

# Pin every dependency the container already provides to the container's version, so that the venv adds
# packages instead of shadowing the container's (numpy 1.26, scipy, pillow, ...).
CONSTRAINTS=$IA3D_LOGS/container-constraints.txt
sort -u "$FREEZE" | grep -E '^[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.!+]*$' > "$CONSTRAINTS"

# The container is Ubuntu 24.04 (glibc 2.39) on aarch64 with CPython 3.12.
PLATFORMS=()
for v in 17 24 26 27 28 31 34 35 36 38 39; do PLATFORMS+=(--platform "manylinux_2_${v}_aarch64"); done
PLATFORMS+=(--platform manylinux2014_aarch64)
PIP=(python3 -m pip download --no-cache-dir --disable-pip-version-check --progress-bar off -d "$IA3D_WHEELHOUSE")
TARGET=(--only-binary=:all: --python-version 3.12 "${PLATFORMS[@]}")

echo "### wheels (with dependencies, constrained to the container's versions)"
"${PIP[@]}" "${TARGET[@]}" -c "$CONSTRAINTS" -r "$HERE/requirements.txt"

echo "### wheels whose dependencies are the container's torch stack (no dependency resolution)"
"${PIP[@]}" "${TARGET[@]}" --no-deps -r "$HERE/requirements-nodeps.txt"

echo "### source distributions (no aarch64 wheel on PyPI; built inside the container)"
SDISTS=$(sed -E 's/[[:space:]]*#.*//; s/[=<>!~ ].*//; /^$/d' "$HERE/requirements-source.txt" | paste -sd,)
"${PIP[@]}" --no-deps --no-binary "$SDISTS" -r "$HERE/requirements-source.txt"

echo "### Eigen $EIGEN_VERSION headers (needed to compile pysdf; the container has none)"
if [ ! -d "$IA3D_SRC/eigen-$EIGEN_VERSION" ]; then
  curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.bz2" | tar -xj -C "$IA3D_SRC"
fi
ls -d "$IA3D_SRC/eigen-$EIGEN_VERSION/Eigen"

echo "### tiny-cuda-nn @ $TCNN_COMMIT"
if [ ! -d "$IA3D_SRC/tiny-cuda-nn/.git" ]; then
  git clone --quiet "$TCNN_URL" "$IA3D_SRC/tiny-cuda-nn"
fi
git -C "$IA3D_SRC/tiny-cuda-nn" fetch --quiet origin
git -C "$IA3D_SRC/tiny-cuda-nn" checkout --quiet "$TCNN_COMMIT"
git -C "$IA3D_SRC/tiny-cuda-nn" submodule update --quiet --init --recursive
git -C "$IA3D_SRC/tiny-cuda-nn" log --oneline -1

echo "### done: $(ls "$IA3D_WHEELHOUSE" | wc -l) files in $IA3D_WHEELHOUSE"
