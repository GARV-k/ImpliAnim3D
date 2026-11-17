# ImpliAnim3D

Work done under guidance of Dr. Pinaki (https://www.pinakinathc.me/) from the SketchX lab (https://sketchx.ai),
built on [torch-ngp](https://github.com/ashawkey/torch-ngp).

The code represents each 3D shape as a signed distance field: a multiresolution hash table per shape, decoded
by one small network that all shapes share. It then trains a matrix `W` that is applied to a shape's hash
table, with the aim that the transformed table decodes to the rotated shape.

This branch, `g/revamp`, is the research code together with what it takes to run it on the KAIROS cluster.
To run anything, start with [kairos/README.md](kairos/README.md).

| Where | What |
|---|---|
| `train_GT2.py`, `test.py`, `train_weights.py`, `inference.py` | The pipeline, in that order: shared decoder and one table per mesh; table of an unseen mesh with the decoder frozen; `W`; meshes decoded through `W`. Options are in `pipeline_args.py`. |
| `sdf/`, `loss.py` | SDF dataset (`sdf/provider.py`), trainer (`sdf/utils2.py`), networks. |
| `hash_imports/` | PyTorch hash encoding taken from nerfstudio, with the hook that applies `W` to the table. |
| `kairos/` | Environment, data and job scripts for the cluster. |
| `Exp*/`, `hash_workspace_*/`, `test/` | Logs of the original experiments (December 2024 to March 2025). |
| `irrelevant_pys/`, `irrelevant_main_files/` | Parts of torch-ngp that this work does not use, kept as they were. |

The setup on this branch shows that every stage runs. It does not show that the method works; what a short
run does and does not show is in the "smoke test" section of [kairos/README.md](kairos/README.md).
