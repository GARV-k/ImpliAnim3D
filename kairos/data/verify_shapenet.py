#!/usr/bin/env python3
"""Check the linked ShapeNet data against the split file, using directory listings only.

Every model id of the split must have meshes/<id>.obj and samples/<id>.npz under <root>/<synset>/, except
the models whose ShapeNet OBJ is empty (_paths.EMPTY_MODELS). Exits non-zero otherwise.
Usage: python3 kairos/data/verify_shapenet.py [--root $IA3D_DATA/shapenet] [--split FILE]
"""
import argparse
import os
import sys

import _paths

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--root", help="directory holding <synset>/{meshes,samples} (default: $IA3D_DATA/shapenet)")
ap.add_argument("--split", default=_paths.SPLIT)
a = ap.parse_args()
root = a.root or os.path.join(_paths.data_dir(), "shapenet")
split = _paths.load_split(a.split)

n_ids = n_ok = 0
problems, skipped, extra = [], [], 0
for syn, ids in split.items():
    listing = {}
    for kind, ext in (("meshes", ".obj"), ("samples", ".npz")):
        d = os.path.join(root, syn, kind)
        listing[kind] = {f[:-len(ext)] for f in os.listdir(d) if f.endswith(ext)} if os.path.isdir(d) else None
        if listing[kind] is None:
            problems.append("%s: no %s directory" % (syn, kind))
    if None in listing.values():
        n_ids += len(ids)
        continue
    for i in ids:
        n_ids += 1
        missing = [kind for kind in listing if i not in listing[kind]]
        if not missing:
            n_ok += 1
        elif (syn, i) in _paths.EMPTY_MODELS and len(missing) == 2:
            skipped.append("%s/%s" % (syn, i))
        else:
            problems.append("%s/%s: missing %s" % (syn, i, " and ".join(missing)))
    extra += sum(len(listing[kind] - set(ids)) for kind in listing)

print("root: %s" % root)
print("split: %d synsets, %d model ids" % (len(split), n_ids))
print("complete (mesh + samples): %d" % n_ok)
print("known empty ShapeNet models, absent as expected: %d %s" % (len(skipped), skipped))
print("files not in the split: %d" % extra)
for p in problems[:50]:
    print("PROBLEM " + p)
print("VERIFY_OK" if not problems else "VERIFY_FAILED (%d problems)" % len(problems))
sys.exit(1 if problems else 0)
