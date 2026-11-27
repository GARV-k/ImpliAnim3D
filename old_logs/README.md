# Logs of the original runs (December 2024 to March 2025)

Each folder holds the trainer's text log, `log_ngp.txt`, and TensorBoard files under `run/ngp/`. The text
logs record the start of each training session and its epochs, but not the losses. The losses, per step and
per mesh, are in the TensorBoard files; to view them, run `tensorboard --logdir old_logs`. The checkpoints
(`.pth`) and meshes were not kept.

The table below is read from the text logs:

- **Workspace:** the checkpoint directory named in each session header. It is named
  `hash_workspace_obj<obj-no>_<n>_<d>_<log2 T>`, where `n/d` is the fraction of samples taken on or near the
  surface and `log2 T = 10`.
- **Kind of run:** told apart by how the learning rate is printed. `train_GT2.py` (tables and decoder) and
  `test.py` (unseen mesh) print `lr=0.000100`; `train_weights.py` (`W`) prints `lr=0.0001`.
- **Decoder:** told apart by the parameter count, which covers one hash table (32,768 numbers) plus the
  decoder. 39,936 is the early tiny-cuda-nn MLP with 2 hidden layers of 64; 135,296 is the 8-layer, 128-wide
  MLP of `sdf/meshsdf_decoder.py`.

| Folder | Dates | Workspace | Decoder | What was run (longest session, in epochs) |
|---|---|---|---|---|
| `Exp4/` | 8 Dec 2024 | `obj12_7_8_10` | small | tables + decoder (41), then `W` (88) |
| `Exp5/` | 9–10 Dec 2024 | `obj12_7_8_10` | small | tables + decoder (19), then `W` (44) |
| `Exp11/` | 10 Dec 2024 | `Exp11/` | small | tables + decoder (14) |
| `Exp13/` | 10–12 Dec 2024 | `obj12_7_8_10` | small | tables + decoder (12), then `W` in eight sessions (up to 500) |
| `Exp12/` | 16 Dec 2024 | `Exp12/workspace/` | small | tables + decoder (5) |
| `hash_workspace_obj12_7_8_10/` | 17 Dec 2024 – 7 Jan 2025 | `obj12_7_8_10` | small | tables + decoder (203), then `W` in many sessions (up to 212) |
| `Exp17/` | 7–10 Jan 2025 | `obj12_2_3_10` | small | tables + decoder (14), then `W` (up to 500) |
| `test/` | 8 Jan – 23 Mar 2025 | `test/` | small, then large | unseen-mesh fits with the decoder frozen; the last one, on 23 Mar, ran 808 epochs |
| `hash_workspace_obj12_2_3_10/` | 21–22 Mar 2025 | `obj12_2_3_10` | large | tables + decoder (31), then `W` (76) |

Many sessions are short restarts, with the learning rate changed or a run resumed. The `Exp*` folders are
copies of a workspace's logs taken after a run, so several of them name the same workspace.
