# Sourced INSIDE the container (after kairos/config.sh): activates the project venv and applies the runtime
# settings of the project. kairos/in_env.sh does this for you.

# Hermetic Python: no ~/.local site-packages, and not the image's own pip pin file.
export PYTHONNOUSERSITE=1
unset PIP_CONSTRAINT

# The NVIDIA image sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which makes every float32 matmul on the GPU run in
# TF32 (10-bit mantissa, relative error about 5e-4). Stock PyTorch computes them in full float32, and the
# table-space operator (table^T @ W) and the decoder rely on that.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

source "$IA3D_VENV/bin/activate"
