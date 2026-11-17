"""Measure what kairos/jobs/smoke_e2e.sbatch produced (run inside the env, from the repo root).

For every fitted field (the training meshes of train_GT2.py and the unseen mesh of test.py) it compares the
network with the ground-truth SDF of the mesh and with its surface, reads the training losses back from the
tensorboard event files, checks that test.py left the shared decoder untouched, and checks the outputs of
train_weights.py and inference.py. Writes <run>/summary.json and exits non-zero if a sanity condition fails.

Every error and loss is printed next to the value an all-zero output gets on the same points. That is the
state every fit starts from, and without it small numbers mislead: within 0.01 of the surface the SDF itself
is below 0.01, so a field that learnt nothing has a "near surface" error of 0.008.

Asserted: every checkpoint and output exists; shared-decoder training and W training reduce their loss; every
training field has less than half the error of the all-zero field on uniform points, and marching cubes finds
a surface in it; the decoder is unchanged by test.py; W is finite.
Reported only: the near-surface error, inside/outside IoU and surface distance (the last two can be asserted
with --min-iou / --max-chamfer), which measure how good a fit is; the fit of the unseen mesh, which may still
sit on its starting plateau after a run this short; and whether W training got below the all-zero loss, which
it does not in 200 steps. The last two are printed as a NOTE.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from loss import mape_loss
from sdf.network_tcnn import SDFNetwork
from sdf.provider import SDFDataset
from sdf.utils2 import extract_geometry


def load_model(enc_path, mlp_path):
    decoder = torch.load(mlp_path, weights_only=False)
    model = SDFNetwork(decoder, encoding="hashgrid")
    model.encoder = torch.load(enc_path, weights_only=False)
    return model.cuda().eval()


@torch.no_grad()
def evaluate(model, obj_path, rng, n_points=100_000, resolution=128):
    """Errors of the fitted field against the SDF and the surface of the mesh, as SDFDataset normalises it.

    zero_mae_* is the error of an all-zero field on the same points, zero_loss the training loss of an all-zero
    output under the sampling the pipeline scripts use by default (--n 2 --d 3).
    """
    ds = SDFDataset(obj_path, size=1, num_samples=2 ** 18, sample_ratio_n=2, sample_ratio_d=3)
    uniform = rng.uniform(-1, 1, (n_points, 3)).astype(np.float32)
    near = (ds.mesh.sample(n_points) + 0.01 * rng.standard_normal((n_points, 3))).astype(np.float32)
    target = torch.from_numpy(ds[0]["sdfs"]).float()
    out = {"zero_loss": float(mape_loss(torch.zeros_like(target), target))}
    for name, pts in (("uniform", uniform), ("near_surface", near)):
        gt = -ds.sdf_fn(pts)  # negative inside, as in SDFDataset
        pred = model(torch.from_numpy(pts).cuda()).squeeze(1).float().cpu().numpy()
        out[f"sdf_mae_{name}"] = float(np.abs(pred - gt).mean())
        out[f"zero_mae_{name}"] = float(np.abs(gt).mean())
        if name == "uniform":
            inside_gt, inside_pred = gt < 0, pred < 0
            out["inside_iou"] = float((inside_gt & inside_pred).sum() / max((inside_gt | inside_pred).sum(), 1))
    vertices, triangles = extract_geometry(torch.FloatTensor([-1, -1, -1]), torch.FloatTensor([1, 1, 1]),
                                           resolution=resolution, threshold=0,
                                           query_func=lambda pts, m: m(pts.cuda()), model=model)
    out["extracted_vertices"] = int(len(vertices))
    b = np.asarray(ds.mesh.sample(20000))
    a = np.zeros((0, 3))
    if len(triangles):
        a = np.asarray(trimesh.Trimesh(vertices, triangles, process=False).sample(20000))
        out["chamfer"] = float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2)
    return out, {"mesh": b, "field": a}


def save_preview(path, rows):
    """One column per field: surface samples of the mesh (top) and of the zero level set of its fit (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3 * len(rows), 6.4))
    for col, (name, pts) in enumerate(rows):
        for row, key in enumerate(("mesh", "field")):
            ax = fig.add_subplot(2, len(rows), row * len(rows) + col + 1, projection="3d")
            p = pts[key][::4]
            if len(p):
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=p[:, 2], s=0.4, cmap="viridis", linewidths=0)
            ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=-60)
            ax.set_title(f"{name}\n{'mesh' if key == 'mesh' else 'fitted field, level 0'}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def loss_curves(event_dir, k=20):
    """One entry per event file (= per Trainer run, in time order): mean loss of the first and last k steps."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    runs = []
    for path in sorted(glob.glob(os.path.join(event_dir, "events.out.tfevents.*"))):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        if "train/loss" not in tags:
            continue
        values = [e.value for e in acc.Scalars("train/loss")]
        runs.append({"steps": len(values), "first": float(np.mean(values[:k])), "last": float(np.mean(values[-k:])),
                     "trains_W": "train/W_grad_norm" in tags})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, help="mesh set directory (holds obj_meshes and obj_meshes_90)")
    ap.add_argument("--run", required=True, help="run directory of the smoke job (holds workspace, unseen, inference)")
    ap.add_argument("--num-objs", type=int, required=True)
    ap.add_argument("--test-idx", type=int, default=40)
    ap.add_argument("--max-relative-error", type=float, default=0.5,
                    help="largest accepted |sdf error| of a training field on uniform points, as a fraction of the "
                         "error an all-zero field has on the same points")
    ap.add_argument("--min-iou", type=float, default=None, help="if given, the inside/outside IoU every field must reach")
    ap.add_argument("--max-chamfer", type=float, default=None, help="if given, the largest accepted surface distance (meshes span about 1.9)")
    a = ap.parse_args()
    ws, unseen, inference = (os.path.join(a.run, d) for d in ("workspace", "unseen", "inference"))
    rng = np.random.default_rng(0)
    summary, problems, notes = {"fields": []}, [], []

    def need(cond, msg, strict=True):
        if not cond:
            (problems if strict else notes).append(msg)

    # 1. fitted fields: table 2k is mesh k+1 unrotated, table 2k+1 is the same mesh rotated (order of train_GT2.py)
    #    The fit of the unseen mesh is reported but not asserted: with the decoder frozen the optimisation starts
    #    on a plateau (output 0 everywhere) and leaves it after a random number of steps, sometimes not within
    #    the budget of a smoke run.
    fields = []
    for k in range(a.num_objs):
        fields.append((f"train {k + 1}", f"{ws}/GT_enc_{2 * k}.pth", f"{ws}/GT_mlp.pth", f"{a.set}/obj_meshes/{k + 1}.obj", True))
        fields.append((f"train {k + 1} rot90", f"{ws}/GT_enc_{2 * k + 1}.pth", f"{ws}/GT_mlp.pth", f"{a.set}/obj_meshes_90/{k + 1}.obj", True))
    fields.append((f"unseen {a.test_idx}", f"{unseen}/GT_enc_0.pth", f"{unseen}/GT_mlp.pth", f"{a.set}/obj_meshes/{a.test_idx}.obj", False))
    print("|sdf error| of each fitted field; (zero) is the error of an all-zero field on the same points")
    print(f"{'field':<16} {'uniform':>8} {'(zero)':>8} {'near surface':>13} {'(zero)':>8} {'inside IoU':>11} {'chamfer':>9} {'vertices':>9}")
    preview = []
    for name, enc, mlp, obj, strict in fields:
        if not (os.path.isfile(enc) and os.path.isfile(mlp)):
            need(False, f"{name}: checkpoint missing ({enc})")
            continue
        r, points = evaluate(load_model(enc, mlp), obj, rng)
        preview.append((name, points))
        summary["fields"].append({"field": name, "mesh": obj, **r})
        print(f"{name:<16} {r['sdf_mae_uniform']:>8.4f} {r['zero_mae_uniform']:>8.4f} {r['sdf_mae_near_surface']:>13.4f} "
              f"{r['zero_mae_near_surface']:>8.4f} {r['inside_iou']:>11.3f} {r.get('chamfer', float('nan')):>9.4f} "
              f"{r['extracted_vertices']:>9d}")
        need(r["extracted_vertices"] > 0, f"{name}: marching cubes found no surface", strict)
        relative = r["sdf_mae_uniform"] / r["zero_mae_uniform"]
        need(relative <= a.max_relative_error,
             f"{name}: |sdf error| on uniform points is {relative:.2f} of the all-zero field's (limit {a.max_relative_error})", strict)
        if a.min_iou is not None:
            need(r["inside_iou"] >= a.min_iou, f"{name}: inside IoU {r['inside_iou']:.3f} < {a.min_iou}", strict)
        if a.max_chamfer is not None:
            need(r.get("chamfer", np.inf) <= a.max_chamfer, f"{name}: chamfer {r.get('chamfer', float('nan')):.4f} > {a.max_chamfer}", strict)
    if a.min_iou is None and a.max_chamfer is None:
        print("(inside IoU and chamfer are reported, not asserted: pass --min-iou / --max-chamfer to assert them)")
    if preview:
        save_preview(os.path.join(a.run, "fields.png"), preview)
        print(f"preview: {os.path.join(a.run, 'fields.png')}")

    # 2. losses, read back from the event files, next to the loss of an all-zero output on the same meshes.
    #    Training the shared decoder and W must reduce the loss by 5% or more; the unseen-mesh fit only has to
    #    stay finite and not get worse (see above).
    stage_runs = loss_curves(os.path.join(ws, "run", "ngp"))
    unseen_runs = loss_curves(os.path.join(unseen, "run", "ngp"))
    shared = [r for r in stage_runs if not r["trains_W"]]
    w_runs = [r for r in stage_runs if r["trains_W"]]

    def zero_loss(select):  # mean all-zero loss over the evaluated fields whose name satisfies select
        values = [f["zero_loss"] for f in summary["fields"] if select(f["field"])]
        return float(np.mean(values)) if values else float("nan")

    print()
    for label, runs, per, strict, zero in (
            ("shared-decoder training", shared, 2 * a.num_objs, True, zero_loss(lambda n: n.startswith("train"))),
            ("unseen-mesh fit", unseen_runs, 1, False, zero_loss(lambda n: n.startswith("unseen"))),
            ("W training", w_runs, a.num_objs, True, zero_loss(lambda n: n.endswith("rot90")))):
        need(len(runs) > 0, f"{label}: no loss curve found")
        for r in runs:
            r["first_per_mesh"], r["last_per_mesh"], r["zero_per_mesh"] = r["first"] / per, r["last"] / per, zero
            print(f"{label:<24} {r['steps']:>5d} steps, loss per mesh {r['first_per_mesh']:.4f} -> {r['last_per_mesh']:.4f}"
                  f"   (all-zero output: {zero:.4f})")
            need(np.isfinite(r["last"]) and r["last"] <= 1.05 * r["first"], f"{label}: loss got worse ({r['first']:.4f} -> {r['last']:.4f})")
            need(r["last"] < 0.95 * r["first"],
                 f"{label}: loss did not move ({r['first_per_mesh']:.4f} -> {r['last_per_mesh']:.4f} per mesh): still on the "
                 f"all-zero plateau" + ("" if strict else "; rerun or raise SMOKE_UNSEEN_EPOCHS"), strict)
            if label == "W training":
                need(not r["last_per_mesh"] > zero,
                     f"W training ended at {r['last_per_mesh']:.2f} per mesh, above the {zero:.2f} of an all-zero output: after "
                     f"{r['steps']} steps the tables transformed by W decode to worse than a blank field, so the W stage "
                     f"and the meshes in inference/ only show that the code runs", strict=False)
    summary["losses"] = {"shared_decoder": shared, "unseen": unseen_runs, "W": w_runs}

    # 3. test.py must leave the shared decoder untouched
    if os.path.isfile(f"{ws}/GT_mlp.pth") and os.path.isfile(f"{unseen}/GT_mlp.pth"):
        d0 = torch.load(f"{ws}/GT_mlp.pth", weights_only=False).state_dict()
        d1 = torch.load(f"{unseen}/GT_mlp.pth", weights_only=False).state_dict()
        frozen = all(torch.equal(d0[key], d1[key]) for key in d0)
        summary["decoder_frozen_in_test"] = frozen
        print(f"\ndecoder after test.py identical to the trained one: {frozen}")
        need(frozen, "test.py changed the shared decoder")

    # 4. W and the meshes decoded through it
    w_path = f"{ws}/W.pth"
    if os.path.isfile(w_path):
        W = torch.load(w_path, weights_only=False)
        summary["W"] = {"shape": list(W.shape), "finite": bool(torch.isfinite(W).all())}
        print(f"W: shape {tuple(W.shape)}, finite {summary['W']['finite']}")
        need(summary["W"]["finite"], "W contains non-finite values")
    else:
        need(False, f"{w_path} missing")
    sizes = [os.path.getsize(p) if os.path.isfile(p) else 0 for p in (f"{inference}/W_{j}.obj" for j in range(a.num_objs))]
    summary["inference_obj_bytes"] = sizes
    print(f"inference meshes (bytes): {sizes}")
    need(all(s > 0 for s in sizes), "inference.py did not write every mesh")

    summary["problems"], summary["notes"] = problems, notes
    with open(os.path.join(a.run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for p in notes:
        print("NOTE:", p)
    for p in problems:
        print("PROBLEM:", p)
    print("SMOKE_VERIFY_OK" if not problems else "SMOKE_VERIFY_FAILED")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
