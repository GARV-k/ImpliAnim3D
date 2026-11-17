#!/usr/bin/env python3
"""Watertight shell meshes and DeepSDF-style SDF samples from raw ShapeNet OBJs, following the DISN recipe.

This is the tool that produced the data linked under $IA3D_DATA/shapenet (taken from the MeshSDF setup on
KAIROS); the recipe below is unchanged, so a re-run reproduces that data.

ShapeNet OBJs are open, self-intersecting and inconsistently oriented, so no orientation-based sign works on
them directly. DISN (whose "marching cube reconstructed ground truth models" MeshSDF used) instead extracts a
closed shell from an unsigned distance grid. We do the same, per instance:

  1. load the OBJ (all parts concatenated) and normalize DeepSDF-style: bounding-box centre at the origin,
     furthest vertex at radius 1/1.03;
  2. unsigned distance d on a GRID^3 lattice over [-1.05, 1.05]^3 (libigl AABB tree, early exit beyond 3*eps);
  3. occupancy = d < eps (eps = 1.5 voxels); free space is flood-filled from the lattice border, every free
     region not reachable from the border is enclosed and counts as interior; marching cubes on the signed
     field (d - eps, negated in the interior) gives a watertight, consistently oriented shell;
  4. the shell is decimated to <= MAX_FACES faces and written to <out>/meshes/<id>.obj (the mesh the SDF
     datasets of this repo load);
  5. DeepSDF sampling against the shell: 95% near-surface points (surface samples perturbed with N(0, 0.0025)
     and N(0, 0.00025) variances) + 5% uniform in [-1, 1]^3; exact signed distance via libigl with the sign
     from the fast winding number (|w| > 0.5 => inside, orientation-agnostic);
     written to <out>/samples/<id>.npz as float32 "pos" / "neg" arrays of shape (N, 4).
Runs on CPU; --workers processes, each using OMP_NUM_THREADS libigl threads.
"""
import argparse, json, os, sys, time, traceback
from multiprocessing import Pool
import numpy as np, trimesh, igl
from scipy import ndimage
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from _paths import EMPTY_MODELS

NORM_SCALE = 1.03       # DeepSDF: shape fits a sphere of radius 1/1.03
GRID = 256              # distance-field lattice resolution
GRID_HALF = 1.05        # lattice covers [-GRID_HALF, GRID_HALF]^3
EPS_VOXELS = 1.5        # shell offset in voxels
MAX_FACES = 100000      # decimation target for the ground-truth shell
CHUNK = 2_000_000       # lattice points per libigl call


def load_normalized(obj_path):
    m = trimesh.load(obj_path, force="mesh", process=False, skip_materials=True)
    v = np.asarray(m.vertices, dtype=np.float64); f = np.asarray(m.faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0: raise ValueError("empty mesh")
    v = v - (v.max(0) + v.min(0)) / 2.0
    v = v / (np.linalg.norm(v, axis=1).max() * NORM_SCALE)
    return np.ascontiguousarray(v), np.ascontiguousarray(f)


def unsigned_distance_grid(V, F, eps):
    """Unsigned distance on the lattice, exact within 3*eps of the surface (larger values are clamped)."""
    ax = np.linspace(-GRID_HALF, GRID_HALF, GRID)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    d = np.empty(len(P))
    ub = (3.0 * eps) ** 2
    for s in range(0, len(P), CHUNK):
        d[s:s + CHUNK] = igl.signed_distance(np.ascontiguousarray(P[s:s + CHUNK]), V, F,
                                             sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED, upper_bound=ub)[0]
    d = np.abs(d)  # unsigned
    return d.reshape(GRID, GRID, GRID), ax


def closed_shell(V, F):
    voxel = 2 * GRID_HALF / (GRID - 1)
    eps = EPS_VOXELS * voxel
    d, ax = unsigned_distance_grid(V, F, eps)
    free = d >= eps
    labels, n = ndimage.label(free)                      # 6-connected free-space regions
    border = np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
                                       labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    outside = np.isin(labels, border[border > 0])
    field = d - eps
    field[~outside] = -np.abs(field[~outside])            # occupied and enclosed voxels are interior
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(voxel,) * 3)
    verts = verts + ax[0]                                  # lattice index -> world
    return np.ascontiguousarray(verts, dtype=np.float64), np.ascontiguousarray(faces, dtype=np.int64), float(outside.mean())


def decimate(V, F, max_faces):
    if len(F) <= max_faces: return V, F
    U, G, _, _ = igl.decimate(np.asfortranarray(V), np.asfortranarray(F.astype(np.int32)), max_faces)[:4]
    if len(G) == 0: return V, F
    return np.ascontiguousarray(U, dtype=np.float64), np.ascontiguousarray(G, dtype=np.int64)


def sample_queries(V, F, n_total, rng):
    n_surf = int(n_total * 0.95); n_unif = n_total - n_surf
    mesh = trimesh.Trimesh(V, F, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_surf, seed=int(rng.integers(1 << 31)))
    pts = np.asarray(pts, dtype=np.float64); half = len(pts) // 2
    near = np.concatenate([pts[:half] + rng.normal(0, np.sqrt(0.0025), (half, 3)),
                           pts[half:] + rng.normal(0, np.sqrt(0.00025), (len(pts) - half, 3))])
    unif = rng.uniform(-1, 1, (n_unif, 3))
    return np.ascontiguousarray(np.concatenate([near, unif]))


def signed_distance_to_shell(q, V, F):
    ud = np.abs(igl.signed_distance(q, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)[0])
    w = igl.fast_winding_number(V, F, q)
    return np.where(np.abs(w) > 0.5, -ud, ud)


def process(args):
    inst, obj_path, out_dir, n_samples, seed = args
    t0 = time.time()
    try:
        V, F = load_normalized(obj_path)
        SV, SF, outside_frac = closed_shell(V, F)
        if len(SF) == 0: return inst, "FAIL empty shell", time.time() - t0
        SV, SF = decimate(SV, SF, MAX_FACES)
        shell = trimesh.Trimesh(SV, SF, process=False)
        shell.export(os.path.join(out_dir, "meshes", inst + ".obj"))
        rng = np.random.default_rng(seed)
        q = sample_queries(SV, SF, n_samples, rng)
        sd = signed_distance_to_shell(q, SV, SF)
        data = np.concatenate([q, sd[:, None]], 1).astype(np.float32)
        pos, neg = data[sd >= 0], data[sd < 0]
        if len(neg) < 50000 or len(pos) < 50000:
            return inst, "WARN unbalanced pos=%d neg=%d shell_faces=%d" % (len(pos), len(neg), len(SF)), time.time() - t0
        # written under another name first: a shape counts as done once its .npz exists, so a file cut short
        # by the job's time limit must never carry the final name
        npz = os.path.join(out_dir, "samples", inst + ".npz")
        with open(npz + ".tmp", "wb") as fh: np.savez(fh, pos=pos, neg=neg)
        os.replace(npz + ".tmp", npz)
        return inst, "ok pos=%d neg=%d in_faces=%d shell_faces=%d watertight=%s" % (
            len(pos), len(neg), len(F), len(SF), shell.is_watertight), time.time() - t0
    except Exception:
        return inst, "FAIL %s" % traceback.format_exc().strip().splitlines()[-1], time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapenet-dir", required=True, help="dir containing <id>/model.obj (one synset)")
    ap.add_argument("--splits", nargs="*", default=[], help="MeshSDF split json files (ids are taken from all of them)")
    ap.add_argument("--all", action="store_true", help="process every <id>/model.obj under --shapenet-dir")
    ap.add_argument("--shard", default="0/1", help="i/N: process only ids with index %% N == i")
    ap.add_argument("--out", required=True, help="output dataset dir, gets samples/ and meshes/")
    ap.add_argument("--n-samples", type=int, default=500000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N ids (testing)")
    a = ap.parse_args()
    if os.path.islink(a.out.rstrip("/")):  # $IA3D_DATA/shapenet/<synset> may be a link into someone else's data
        sys.exit("refusing to write through the symlink %s -> %s" % (a.out, os.readlink(a.out.rstrip("/"))))
    ids = []
    for s in a.splits:
        for ds in json.load(open(s)).values():
            for cls in ds.values(): ids += list(cls)
    if a.all: ids += sorted(d for d in os.listdir(a.shapenet_dir) if os.path.exists(os.path.join(a.shapenet_dir, d, "model.obj")))
    ids = list(dict.fromkeys(ids))
    si, sn = map(int, a.shard.split("/")); ids = ids[si::sn]
    if a.limit: ids = ids[: a.limit]
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True); os.makedirs(os.path.join(a.out, "meshes"), exist_ok=True)
    synset = os.path.basename(os.path.normpath(a.shapenet_dir))
    todo = []
    n_ok = n_bad = 0
    for i, inst in enumerate(ids):
        if os.path.exists(os.path.join(a.out, "samples", inst + ".npz")): continue
        if (synset, inst) in EMPTY_MODELS: print(inst, "SKIPPED (ShapeNet OBJ without geometry)", flush=True); continue
        obj = os.path.join(a.shapenet_dir, inst, "model.obj")
        if not os.path.exists(obj): print(inst, "MISSING", flush=True); n_bad += 1; continue
        todo.append((inst, obj, a.out, a.n_samples, 1000 + i))
    print("ids: %d, to process: %d, workers: %d, grid: %d" % (len(ids), len(todo), a.workers, GRID), flush=True)
    with Pool(a.workers) as pool:
        for k, (inst, status, dt) in enumerate(pool.imap_unordered(process, todo)):
            n_ok += status.startswith("ok"); n_bad += not status.startswith("ok")
            print("[%d/%d] %s %s (%.1fs)" % (k + 1, len(todo), inst, status, dt), flush=True)
    print("DONE ok=%d bad=%d" % (n_ok, n_bad), flush=True)
    sys.exit(1 if n_bad else 0)  # a shape that failed, gave no usable samples, or has no model.obj
