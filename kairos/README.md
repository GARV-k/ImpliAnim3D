# Running ImpliAnim3D on KAIROS

KAIROS has two kinds of nodes, and they differ in architecture:

| | Login node | GPU node (GH200) |
|---|---|---|
| CPU | x86_64 | aarch64 |
| Internet | yes | no |
| GPU | no | yes |

The shared container `/work/conteneurs/shared/AI/nemo_25.04.03_arm.sif` is aarch64, so it can only be
executed inside a Slurm job. It is never modified. The project environment is a Python venv created inside
the container with `--system-site-packages`: it reuses the container's PyTorch and adds what this repo needs.
Everything from the Internet is downloaded on the login node first and installed offline in a job.

## Layout

Tracked, in this directory:

| Path | Purpose |
|---|---|
| `config.sh` | Every cluster setting (account, partition, container, stage directory). Overridable from the environment. |
| `sbatch.sh` | `sbatch` wrapper that applies `config.sh` and sends logs to the stage directory. |
| `in_env.sh` | Runs a command inside the container with the venv active, from the repo root. GPU nodes only. |
| `env/` | Requirement lists, `fetch.sh` (login node), `build_in_container.sh`, `activate.sh`, `check_env.py`. |
| `jobs/` | Job scripts: `probe`, `build_env`, `check_env`, `check_data`, `smoke_e2e`, and `run` for your own commands. |
| `data/` | Data tools and their own [README](data/README.md). |

Not tracked, in the stage directory `$IA3D_STAGE` (default `$WORK/irr/ImpliAnim3D-stage`). It sits outside the
repo so that every checkout and worktree shares one environment and one copy of the data:

| Path | Content |
|---|---|
| `venv/` | The project environment (190 MB). Its `python` is a link into the container, so it is dangling on the login node. |
| `wheelhouse/`, `src/` | aarch64 wheels, source distributions, tiny-cuda-nn and Eigen, as downloaded by `env/fetch.sh`. |
| `data/` | Links to the ShapeNet data and the mesh sets built from it. |
| `logs/` | Slurm logs (`<job name>-<job id>.out`), build logs, `container-freeze.txt`, `venv-freeze.txt`. |
| `runs/` | Outputs of the smoke test; a good place for workspaces. |

## Setting it up from nothing

Run from the repo root. Steps 1 to 3 build the environment, 4 and 5 the data, 6 and 7 check both.

```bash
kairos/sbatch.sh kairos/jobs/probe.sbatch        # 1. GPU node, 1 min: records what the container provides
kairos/env/fetch.sh                              # 2. login node: wheels and sources into the stage directory
kairos/sbatch.sh kairos/jobs/build_env.sbatch    # 3. GPU node, 4 min: venv, compiled packages, checks
kairos/data/link_shapenet.sh                     # 4. login node: link the preprocessed ShapeNet data
python3 kairos/data/make_legacy_set.py --n 60    # 5. login node: mesh set in the layout the scripts read
kairos/sbatch.sh kairos/jobs/check_data.sbatch   # 6. GPU node, 1 min: mesh set and regeneration recipe
kairos/sbatch.sh kairos/jobs/smoke_e2e.sbatch    # 7. GPU node, 16 min: the whole pipeline on three meshes
```

`sq` shows the queue; each job writes `$IA3D_STAGE/logs/<job name>-<job id>.out`. Rebuild the environment
from scratch with `IA3D_REBUILD=1 kairos/sbatch.sh kairos/jobs/build_env.sbatch`.

## What the environment contains

From the container: Python 3.12, PyTorch 2.7 (NVIDIA build 25.03) with CUDA 12.8, torchvision, numpy 1.26,
scipy, trimesh, opencv, pandas, matplotlib, rich, tqdm, torchmetrics, the CUDA compiler, gcc 13, pybind11
and Cython.

Added by the venv (`env/requirements*.txt`):

| Package | How | Used by |
|---|---|---|
| tinycudann 2.0 | compiled from the pinned commit in `env/tiny-cuda-nn.commit` | `tcnn.Network`, imported by every pipeline script |
| pysdf 0.1.9 | compiled from source (no aarch64 wheel), with Eigen 3.3.7 headers | ground-truth SDF in `sdf/provider.py` |
| PyMCubes 0.1.6 | compiled from source (no aarch64 wheel) | marching cubes in `sdf/utils2.py` |
| libigl 2.6.3, scikit-image 0.26.0 | wheels | `close.py`, `kairos/data/regen` |
| jaxtyping 0.3.11 | wheel | the vendored nerfstudio encodings in `hash_imports/` |
| torch-ema 0.3, tensorboardX 2.6.5, lpips 0.1.4 | wheels | trainer, logs |

Not installed: `dearpygui` (GUI, no display on the cluster) and `viser` (only one unused nerfstudio helper
needed it; its import is now deferred). The CUDA extensions of vanilla torch-ngp under `irrelevant_pys/`
(`gridencoder`, `raymarching`, ...) are not built or tested: the pipeline uses the PyTorch hash encoding.

One runtime setting matters for this project. The NVIDIA image enables TF32 for every float32 matrix product
on the GPU, which keeps only 10 bits of mantissa: `table^T @ W` with `W = I` came back with a relative error
of 5e-4. `env/activate.sh` turns that off (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`), which is what stock
PyTorch does, and `check_env.py` asserts that `W = I` reproduces the table bit for bit.

## Running things

Any command, as a batch job:

```bash
source kairos/config.sh      # for $IA3D_DATA and $IA3D_RUNS in your shell
kairos/sbatch.sh --time=12:00:00 --job-name=shared-dec kairos/jobs/run.sbatch \
  python -u train_GT2.py --data-root "$IA3D_DATA/sets/shapenet_rr60/obj_meshes" --workspace "$IA3D_RUNS/exp1"
```

Interactively, from a shell on an allocated GPU node (`salloc` with the same account, partition and
`--gres` as in `config.sh`):

```bash
kairos/in_env.sh python kairos/env/check_env.py     # one command
kairos/in_env.sh bash                               # or a shell with the venv active
```

Inside the container use `/work/...` paths: `/gpfs/work` is not mounted there. `config.sh` rewrites the
paths it derives.

## The pipeline scripts

The four scripts take the same options (`pipeline_args.py`); the defaults are the values that used to be
hard-coded, so a script without arguments behaves as before. `--help` lists them.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `train_GT2.py` | Shared decoder and one hash table per mesh, for meshes `1..num-objs`, each unrotated and rotated | `<data-root>/<i>.obj`, `<data-root>_90/<i>.obj` | `<workspace>/GT_mlp.pth`, `GT_enc_<k>.pth` (even k unrotated, odd k rotated) |
| `test.py` | Hash table of an unseen mesh with the decoder frozen | `<workspace>/GT_mlp.pth`, `--test-obj` (default `<data-root>/40.obj`) | `<test-workspace>/GT_enc_0.pth` |
| `train_weights.py` | W on the rotated meshes with tables and decoder frozen | `<workspace>/GT_mlp.pth`, `GT_enc_<2i>.pth`, `<data-root>_90/<i>.obj` | `<workspace>/W.pth` |
| `inference.py` | Decodes the W-transformed tables and extracts meshes | `<workspace>` | `<out>/W_<j>.obj` |

Common options: `--data-root`, `--workspace`, `--num-objs`, `--epochs`, `--steps-per-epoch`, `--num-samples`,
`--lr`, `--hashmap-size`, `--n`, `--d`.

## Changes made to the research code for this stack

- **Options instead of constants:** `train_GT2.py`, `test.py`, `train_weights.py` and `inference.py` read the
  data root, workspace, number of meshes, epochs and learning rate from the command line.
- **`torch.load(..., weights_only=False)`:** the checkpoints are whole pickled modules, which PyTorch 2.6 and
  later refuses to load by default.
- **Decoder update in `train_GT2.py`:** the trainer's decoder step was commented out, so shared-decoder
  training only moved the hash tables. `Trainer(train_net=...)` now controls it: `train_GT2.py` trains the
  decoder (`--freeze-decoder` restores the old behaviour) and `test.py` keeps it frozen.
- **`train_weights.py` creates W when the workspace has none** (random normal, the initialisation of the
  commented-out line); it used to require an existing `W.pth`.
- **`hash_imports/scene_box.py`:** `viser` is imported inside the one method that uses it.

## The smoke test

`jobs/smoke_e2e.sbatch` runs the four scripts on meshes 1 to 3 of the set, each unrotated and rotated by 90
degrees (six hash tables, one decoder), with mesh 40 as the unseen one. `jobs/verify_smoke.py` then measures
the result and writes `summary.json` and `fields.png` into the run directory `$IA3D_RUNS/smoke-<job id>`.
The test uses `--lr 1e-3`; the scripts' default of `1e-4` needs far more steps than a smoke test has.

Reference run: job 102992, one GH200, 2026-11-11, 15 min 31 s in total, `SMOKE_VERIFY_OK`.

| Stage | Steps | Time | Loss per mesh (mean of the first 20 steps -> last 20) |
|---|---|---|---|
| 1. `train_GT2.py`, 6 tables + decoder | 1,000 | 10 min 21 s | 0.378 -> 0.121 |
| 2. `test.py`, unseen mesh, decoder frozen | 2,000 | 3 min 27 s | 0.382 -> 0.123 |
| 3. `train_weights.py`, W on 3 rotated meshes | 200 | 1 min 17 s | 137 -> 1.54 |
| 4. `inference.py`, 3 meshes at 64^3 | | 6 s | |

What `verify_smoke.py` asserts: every checkpoint and output exists; stages 1 and 3 reduce their loss; every
training field is within 0.02 of the true SDF near the surface (measured: 0.005 to 0.007) and has a zero
level set; `test.py` leaves the decoder bit-identical; W is finite. It reports, without asserting, how good
the fits are (inside/outside IoU, surface distance) and the fit of the unseen mesh.

How to read the numbers of such a short run, so that they are not mistaken for a broken environment:

- **0.378 is the loss of a network that outputs 0 everywhere** (0.37 to 0.39 depending on the mesh). Every
  fit starts there and waits before it moves: about 60 steps when the decoder is trained, and anything from
  250 steps to more than 600 when it is frozen. Two of three 600-step runs of `test.py` never left that
  value, which is why the unseen-mesh fit gets 2,000 steps and is reported with a `NOTE` instead of failing
  the test.
- **The fits are not signed.** The network output is almost never negative: at most 5% of the points
  inside a mesh are classified as inside (inside/outside IoU 0.00 to 0.06 over all runs), although the
  field is accurate near the surface. A field
  that only touches zero has a ragged zero level set, which is what `fields.png` shows: the shapes are
  recognisable and surrounded by floaters. More steps do not change this: with 4,000 steps of stage 1
  (job 102989) the loss reached 0.103 and the output was still never negative. For reference, the exact
  unsigned distance `|sdf|` scores 0.06 to 0.10 under this loss and sampling, because only 8 to 10% of the
  samples lie inside a mesh.
