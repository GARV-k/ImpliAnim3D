#!/usr/bin/env python3
"""Check a mesh set built by make_legacy_set.py. Needs numpy and trimesh; pysdf is used when it is installed.

For every index of the manifest:
  1. obj_meshes/<i>.obj resolves to the manifest source and loads as a non-empty triangle mesh;
  2. it is watertight, consistently wound and has positive volume (outward normals), both as stored and after
     trimesh's default processing (which is how sdf.provider.SDFDataset loads it);
  3. every vertex lies inside the unit ball;
  4. obj_meshes_<angle>/<i>.obj has the same faces, and its vertices are R v (R as in make_legacy_set.py);
  5. the normalisation of sdf.provider.SDFDataset (bounding-box centre to the origin, scale 2 / diagonal * 0.95)
     commutes with R, i.e. the dataset of the rotated file is the rotation of the dataset of the original.
     True for multiples of 90 degrees; for other angles the bounding box changes, and the deviation is
     reported as a warning instead of a failure;
  6. with pysdf: the SDF of the normalised rotated mesh at R p equals the SDF of the normalised original at p
     for random p in [-1, 1]^3, and the cube corner is outside (pysdf is positive inside; SDFDataset negates it).
Exits non-zero if any check fails.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

from make_legacy_set import cos_sin

try:
    import pysdf
except ImportError:
    pysdf = None

TOL_VERTEX = 1e-12  # the rotated file stores the exact double
TOL_NORM = 1e-9
TOL_SDF = 1e-4      # pysdf works in float32


def load(path, process=False):
    m = trimesh.load(path, force="mesh", process=process)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64), m


def normalise(v):
    """Vertex normalisation of sdf.provider.SDFDataset."""
    vmin, vmax = v.min(0), v.max(0)
    return (v - (vmin + vmax) / 2) * (2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95)


def rotation(angle):
    c, s = cos_sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def check_mesh(set_dir, entry, angles, sdf_points, rng):
    """Returns (summary text, list of failures, list of warnings) for one index."""
    fails, warns, notes = [], [], []
    path = os.path.join(set_dir, "obj_meshes", "%d.obj" % entry["index"])
    if os.path.realpath(path) != os.path.realpath(entry["source"]):
        fails.append("link does not resolve to the manifest source")
    v, f, mesh = load(path)
    if len(v) == 0 or len(f) == 0:
        return "empty", fails + ["empty mesh"], warns
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
        fails.append("not a closed, outward-oriented surface as stored")
    processed = load(path, process=True)[2]  # what SDFDataset sees
    if not (processed.is_watertight and processed.volume > 0):
        fails.append("not watertight after trimesh processing")
    r_max = float(np.linalg.norm(v, axis=1).max())
    if not r_max <= 1.0:
        fails.append("outside the unit ball (r_max %.4f)" % r_max)
    notes.append("V=%d F=%d r_max=%.4f" % (len(v), len(f), r_max))

    sdf0 = pysdf.SDF(normalise(v).astype(np.float32), f.astype(np.uint32)) if pysdf and sdf_points else None
    if sdf0 is not None:
        p = rng.uniform(-1, 1, (sdf_points, 3)).astype(np.float32)
        d0 = sdf0(p)
        inside = float((d0 > 0).mean())
        if not (sdf0(np.full((1, 3), 0.99, dtype=np.float32))[0] < 0 and 0 < inside < 0.9):
            fails.append("SDF sign convention broken (inside fraction %.3f)" % inside)
        notes.append("inside=%.3f" % inside)

    for angle in angles:
        tag = "rot%g" % angle
        rpath = os.path.join(set_dir, "obj_meshes_%g" % angle, "%d.obj" % entry["index"])
        if not os.path.isfile(rpath):
            fails.append("%s: file missing" % tag)
            continue
        rv, rf, _ = load(rpath)
        R = rotation(angle)
        if rv.shape != v.shape or not np.array_equal(rf, f):
            fails.append("%s: topology differs from the original" % tag)
            continue
        dv = float(np.abs(rv - v @ R.T).max())
        if dv > TOL_VERTEX:
            fails.append("%s: vertices are not R v (max deviation %.2e)" % (tag, dv))
        dn = float(np.abs(normalise(rv) - normalise(v) @ R.T).max())
        if dn > TOL_NORM:
            msg = "%s: SDFDataset normalisation does not commute with the rotation (max deviation %.2e)" % (tag, dn)
            (fails if angle % 90 == 0 else warns).append(msg)
        note = "%s dv=%.1e dnorm=%.1e" % (tag, dv, dn)
        if sdf0 is not None and dn <= TOL_NORM:
            sdf1 = pysdf.SDF(normalise(rv).astype(np.float32), rf.astype(np.uint32))
            ds = float(np.abs(sdf1((p @ R.T).astype(np.float32)) - d0).max())
            if ds > TOL_SDF:
                fails.append("%s: SDF of the rotated mesh is not the rotated SDF (max deviation %.2e)" % (tag, ds))
            note += " dsdf=%.1e" % ds
        notes.append(note)
    return "  ".join(notes), fails, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("set_dir", help="directory written by make_legacy_set.py")
    ap.add_argument("--limit", type=int, default=0, help="only check the first N indices")
    ap.add_argument("--sdf-points", type=int, default=20000, help="random points for the pysdf check (0 disables it)")
    a = ap.parse_args()
    with open(os.path.join(a.set_dir, "manifest.json")) as f:
        manifest = json.load(f)
    entries = manifest["meshes"][: a.limit or None]
    rng = np.random.default_rng(0)
    print("set: %s  meshes: %d  angles: %s  pysdf: %s" % (a.set_dir, len(entries), manifest["angles"],
                                                         "yes" if pysdf else "not installed, SDF check skipped"))
    n_fail = n_warn = 0
    for e in entries:
        try:
            text, fails, warns = check_mesh(a.set_dir, e, manifest["angles"], a.sdf_points, rng)
        except Exception as exc:  # an unreadable file is a failed check, not a crash
            text, fails, warns = "", ["%s: %s" % (type(exc).__name__, exc)], []
        n_fail += bool(fails)
        n_warn += bool(warns)
        print("[%3d] %s %-12s %s  %s" % (e["index"], e["synset"], e["category"][:12], text, "FAIL" if fails else "ok"),
              flush=True)
        for msg in fails:
            print("      FAIL " + msg)
        for msg in warns:
            print("      WARN " + msg)
    print("checked %d meshes: %d failed, %d with warnings" % (len(entries), n_fail, n_warn))
    print("CHECK_SET_OK" if not n_fail else "CHECK_SET_FAILED")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
