#!/usr/bin/env python3
"""Compare regenerated shapes with a reference tree: the acceptance test of preprocess_sdf.py.

Usage: compare_regen.py <new dir> <reference dir>      (both hold meshes/<id>.obj and samples/<id>.npz)
Every file under <new dir> must exist in the reference and be byte-identical, or numerically equal within
--tol (same topology for meshes, same array shapes for samples). Exits non-zero otherwise.
"""
import argparse
import filecmp
import os
import sys

import numpy as np


def read_obj(path):
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("f "):
                f.append([t.split("/")[0] for t in line.split()[1:4]])
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def deviation(new, ref):
    """Largest absolute difference, or None when the two files do not have the same structure."""
    if new.endswith(".obj"):
        (v0, f0), (v1, f1) = read_obj(new), read_obj(ref)
        if v0.shape != v1.shape or not np.array_equal(f0, f1):
            return None
        return float(np.abs(v0 - v1).max())
    a, b = np.load(new), np.load(ref)
    if sorted(a.files) != sorted(b.files) or any(a[k].shape != b[k].shape for k in a.files):
        return None
    return max(float(np.abs(a[k] - b[k]).max()) for k in a.files)


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("new")
ap.add_argument("ref")
ap.add_argument("--tol", type=float, default=1e-6)
a = ap.parse_args()

n = n_identical = n_bad = 0
for kind in ("meshes", "samples"):
    for name in sorted(os.listdir(os.path.join(a.new, kind))):
        new, ref = os.path.join(a.new, kind, name), os.path.join(a.ref, kind, name)
        n += 1
        if not os.path.exists(ref):
            status = "NOT IN REFERENCE"
        elif filecmp.cmp(new, ref, shallow=False):
            status = "identical"
        else:
            dev = deviation(new, ref)
            status = ("DIFFERENT STRUCTURE" if dev is None else
                      "equal within tolerance (max deviation %.2e)" % dev if dev <= a.tol else
                      "DIFFERENT (max deviation %.2e)" % dev)
        n_identical += status == "identical"
        n_bad += status.isupper() or status.startswith("DIFFERENT")
        print("%s/%s: %s" % (kind, name, status))
print("%d files: %d byte-identical, %d within tolerance, %d different" % (n, n_identical, n - n_identical - n_bad, n_bad))
print("COMPARE_OK" if n and not n_bad else "COMPARE_FAILED")
sys.exit(0 if n and not n_bad else 1)
