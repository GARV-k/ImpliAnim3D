#!/bin/bash
#SBATCH --job-name=ia3d-probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
# Record what the container provides (Python, torch, CUDA toolchain, installed packages).
# Writes $IA3D_LOGS/container-freeze.txt, which kairos/env/fetch.sh uses as the pip constraint file.
: "${IA3D_REPO:=${SLURM_SUBMIT_DIR:?}}"
source "$IA3D_REPO/kairos/config.sh"
hostname; uname -m; nproc; nvidia-smi -L
apptainer exec --bind /tmpdir,/work --nv "$IA3D_CONTAINER" bash -c '
export PYTHONNOUSERSITE=1
echo "### python"; command -v python3; python3 --version
python3 -c "import sysconfig, platform; print(sysconfig.get_platform(), platform.libc_ver())"
echo "### torch"
python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
echo "### toolchain"
for t in nvcc gcc g++ cmake ninja git make; do printf "%s: " $t; command -v $t || echo MISSING; done
nvcc --version | tail -2; gcc --version | head -1; cmake --version 2>/dev/null | head -1
python3 -c "import torch.utils.cpp_extension as c; print(\"CUDA_HOME\", c.CUDA_HOME)"
ls -d /usr/include/eigen3 2>&1
echo "### pip"; python3 -m pip --version; python3 -m pip config list
echo "PIP_CONSTRAINT=${PIP_CONSTRAINT:-<unset>}  PYTHONPATH=${PYTHONPATH:-<unset>}"
python3 -c "import ensurepip, venv; print(\"ensurepip + venv available\")"
unset PIP_CONSTRAINT
# written to a temporary name first: fetch.sh must never read a list that pip left half written
if python3 -m pip list --format=freeze 2>/dev/null > "$IA3D_LOGS/container-freeze.txt.tmp" && [ -s "$IA3D_LOGS/container-freeze.txt.tmp" ]; then
  mv "$IA3D_LOGS/container-freeze.txt.tmp" "$IA3D_LOGS/container-freeze.txt"
else
  rm -f "$IA3D_LOGS/container-freeze.txt.tmp"; echo "pip list failed: container-freeze.txt not written" >&2; exit 1
fi
echo "packages in the container: $(wc -l < "$IA3D_LOGS/container-freeze.txt")"
echo "### modules the repo imports"
for m in tinycudann torch_ema tensorboardX rich pandas matplotlib cv2 mcubes pysdf jaxtyping viser skimage igl \
         trimesh tqdm imageio packaging scipy numpy lpips torchmetrics torchvision pybind11 Cython setuptools wheel; do
  python3 -c "import $m; print(\"  $m\", getattr($m, \"__version__\", \"?\"))" 2>&1 | tail -1
done
exit 0
'
status=$?
if [ "$status" -eq 0 ]; then echo PROBE_DONE; else echo "PROBE_FAILED (exit $status)"; fi
exit $status
