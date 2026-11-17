#!/usr/bin/env python3
"""Extract the first N model ids (sorted; ids are hashes, so this is an arbitrary but reproducible subset) of
every synset in ShapeNetCore.v1.zip into <out>/ShapeNetCore.v1/<synset>/<id>/..., and write the split file
{"<synset>": {"samples": [ids]}, ...} covering the selection.

Usage: extract_subset.py ShapeNetCore.v1.zip <out> <N> <split.json>
With N = 200 the split it writes is kairos/data/splits/shapenet_v1_200_per_category.json.
"""
import zipfile, json, os, sys, collections, time
Z, OUT, N, SPLIT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()
zf = zipfile.ZipFile(Z)
names = zf.namelist()
by_syn = collections.defaultdict(set)
for n in names:
    p = n.split("/")
    if len(p) == 4 and p[3] == "model.obj": by_syn[p[1]].add(p[2])
sel = {syn: sorted(ids)[:N] for syn, ids in sorted(by_syn.items())}
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(SPLIT)), exist_ok=True)  # the split may live inside OUT
with open(SPLIT, "w") as f: json.dump({syn: {"samples": ids} for syn, ids in sel.items()}, f)
want = {"ShapeNetCore.v1/%s/%s/" % (syn, i) for syn, ids in sel.items() for i in ids}
members = [n for n in names if len(n.split("/")) >= 4 and "/".join(n.split("/")[:3]) + "/" in want]
print("synsets: %d, selected models: %d, zip members to extract: %d" % (len(sel), sum(len(v) for v in sel.values()), len(members)), flush=True)
for k, m in enumerate(members):
    zf.extract(m, OUT)
    if k % 10000 == 0: print("  %d / %d members (%.0fs)" % (k, len(members), time.time() - t0), flush=True)
missing = [(syn, i) for syn, ids in sel.items() for i in ids if not os.path.exists(os.path.join(OUT, "ShapeNetCore.v1", syn, i, "model.obj"))]
print("missing model.obj after extraction:", len(missing), missing[:3])
if missing: sys.exit("EXTRACT_FAILED: %d selected models have no model.obj" % len(missing))
print("EXTRACT_DONE in %.0fs" % (time.time() - t0), flush=True)
