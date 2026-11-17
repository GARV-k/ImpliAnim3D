#!/usr/bin/env python3
"""Build a mesh set in the layout the research scripts read.

train_GT2.py, train_weights.py and test.py address meshes as <root>/<i>.obj and <root>_<angle>/<i>.obj with
a 1-based index i. Under <out> this writes

  obj_meshes/<i>.obj           symlink to a preprocessed ShapeNet mesh, <shapenet root>/<synset>/meshes/<id>.obj
  obj_meshes_<angle>/<i>.obj   that mesh with every vertex rotated by <angle> degrees, in the convention of
                               shift_objs.rotate_vertices_y:  x' = x cos a - z sin a,  y' = y,  z' = x sin a + z cos a
  manifest.json                index -> synset, category name, model id, source path

so <out>/obj_meshes is the --data-root of the scripts (their default is data/obj_meshes: linking <out> as
data in a checkout also works). Multiples of 90 degrees are applied exactly (no cos(90) residue).

The selection is deterministic and category-balanced: round-robin over the synsets (sorted), round k taking the
k-th model id of each synset. A larger --n therefore extends a smaller set without renumbering it.
Standard library only, so it runs on the login node as well as in the container.
"""
import argparse
import json
import math
import os
import sys

import _paths


def select(split, synsets, n):
    """First n (synset, id) pairs of the round-robin; models without geometry are skipped."""
    pools = [(syn, [i for i in split[syn] if (syn, i) not in _paths.EMPTY_MODELS]) for syn in synsets]
    picked, k = [], 0
    while len(picked) < n:
        row = [(syn, ids[k]) for syn, ids in pools if k < len(ids)]
        if not row:
            sys.exit("only %d meshes available for %d synsets" % (len(picked), len(synsets)))
        picked += row[:n - len(picked)]
        k += 1
    return picked


def cos_sin(angle):
    """cos and sin of an angle in degrees, exact for multiples of 90."""
    quarter, rest = divmod(angle % 360.0, 90.0)
    if rest == 0:
        return [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][int(quarter)]
    return math.cos(math.radians(angle)), math.sin(math.radians(angle))


def rotate_obj(src, dst, angle):
    """Copy an OBJ, rotating its vertices (and vertex normals, if any) about the Y axis."""
    c, s = cos_sin(angle)
    tmp = dst + ".tmp"
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            if line.startswith(("v ", "vn ")):
                p = line.split()
                x, z = float(p[1]), float(p[3])
                p[1], p[3] = repr(x * c - z * s), repr(x * s + z * c)
                line = " ".join(p) + "\n"
            fout.write(line)
    os.replace(tmp, dst)


def symlink(target, path):
    if os.path.islink(path) and os.readlink(path) == target:
        return
    tmp = path + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, path)


def stray_files(out, indices, angles):
    """Mesh files under <out>/obj_meshes* that a manifest with these indices and angles does not list."""
    listed = {"obj_meshes"} | {"obj_meshes_%g" % x for x in angles}
    names = {"%d.obj" % i for i in indices}
    strays = []
    for d in sorted(os.listdir(out)):
        if d.startswith("obj_meshes") and os.path.isdir(os.path.join(out, d)):
            strays += [os.path.join(out, d, f) for f in sorted(os.listdir(os.path.join(out, d)))
                       if f.endswith(".obj") and (d not in listed or f not in names)]
    return strays


def remove_strays(out, indices, angles):
    strays = stray_files(out, indices, angles)
    for path in strays:
        os.remove(path)
    for d in {os.path.dirname(path) for path in strays}:
        if not os.listdir(d):
            os.rmdir(d)
    return len(strays)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=60, help="number of meshes (indices 1..n)")
    ap.add_argument("--angles", type=float, nargs="*", default=[90.0], help="rotation angles in degrees")
    ap.add_argument("--name", help="set name (default: shapenet_rr<n>)")
    ap.add_argument("--out", help="output directory (default: $IA3D_DATA/sets/<name>)")
    ap.add_argument("--root", help="preprocessed ShapeNet root (default: $IA3D_DATA/shapenet)")
    ap.add_argument("--split", default=_paths.SPLIT)
    ap.add_argument("--synsets", help="comma-separated synset ids to draw from (default: all of the split file)")
    ap.add_argument("--force", action="store_true",
                    help="replace the set: rewrite the rotated meshes, replace a manifest that disagrees and "
                         "remove the mesh files the new selection does not list")
    a = ap.parse_args()

    name = a.name or "shapenet_rr%d" % a.n
    root = _paths.work_path(a.root or os.path.join(_paths.data_dir(), "shapenet"))
    out = _paths.work_path(a.out or os.path.join(_paths.data_dir(), "sets", name))
    split = _paths.load_split(a.split)
    synsets = sorted(a.synsets.split(",")) if a.synsets else sorted(split)
    unknown = [syn for syn in synsets if syn not in split]
    if unknown:
        sys.exit("synsets not in the split file: %s" % unknown)
    names = _paths.synset_names()

    meshes = []
    for index, (syn, mid) in enumerate(select(split, synsets, a.n), start=1):
        source = os.path.join(root, syn, "meshes", mid + ".obj")
        if not os.path.isfile(source):
            sys.exit("mesh missing: %s (run kairos/data/link_shapenet.sh)" % source)
        meshes.append({"index": index, "synset": syn, "category": names.get(syn, ""), "id": mid, "source": source})

    manifest_path = os.path.join(out, "manifest.json")
    angles = sorted(set(a.angles))
    selection = "round-robin over %d synsets, k-th model id of each synset in round k" % len(synsets)
    if os.path.exists(manifest_path) and not a.force:
        with open(manifest_path) as f:
            old = json.load(f)
        clash = [m["index"] for m, o in zip(meshes, old["meshes"]) if (m["synset"], m["id"]) != (o["synset"], o["id"])]
        if clash:
            sys.exit("%s already holds a different selection (first at index %d); use --force or another --name"
                     % (out, clash[0]))
        if len(old["meshes"]) > len(meshes):  # never shrink: keep the larger earlier build and its description
            meshes, name, selection = old["meshes"], old["name"], old["selection"]
        angles = sorted(set(angles) | set(old["angles"]))

    os.makedirs(os.path.join(out, "obj_meshes"), exist_ok=True)
    n_rotated = 0
    for m in meshes:
        symlink(m["source"], os.path.join(out, "obj_meshes", "%d.obj" % m["index"]))
        for angle in angles:
            d = os.path.join(out, "obj_meshes_%g" % angle)
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, "%d.obj" % m["index"])
            if a.force or not os.path.exists(dst):
                rotate_obj(m["source"], dst, angle)
                n_rotated += 1
    if a.force:  # the only case in which the set can shrink or lose an angle
        n_removed = remove_strays(out, {m["index"] for m in meshes}, angles)
        if n_removed:
            print("removed %d mesh files of the previous selection" % n_removed)

    manifest = {
        "name": name,
        "layout": {"original": "obj_meshes/<index>.obj", "rotated": "obj_meshes_<angle>/<index>.obj"},
        "rotation": "x' = x cos a - z sin a, y' = y, z' = x sin a + z cos a (shift_objs.rotate_vertices_y)",
        "angles": angles,
        "selection": selection,
        "split": os.path.basename(a.split),
        "shapenet_root": root,
        "meshes": meshes,
    }
    with open(manifest_path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    os.replace(manifest_path + ".tmp", manifest_path)
    print("%s: %d meshes from %d synsets, angles %s, %d rotated files written"
          % (out, len(meshes), len({m["synset"] for m in meshes}), ["%g" % x for x in angles], n_rotated))


if __name__ == "__main__":
    main()
