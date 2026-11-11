"""Locations shared by the data tools. Standard library only (the tools also run on the login node).

kairos/config.sh stays the single source of truth for the stage layout: when IA3D_DATA is not in the
environment, it is read from there.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "splits", "shapenet_v1_200_per_category.json")
SYNSET_NAMES = os.path.join(HERE, "splits", "synset_names.json")

# ShapeNetCore.v1 models of the split whose OBJ contains no geometry; preprocessing cannot produce them.
EMPTY_MODELS = {
    ("03624134", "67ada28ebc79cc75a056f196c127ed77"),
    ("04074963", "b65b590a565fa2547e1c85c5c15da7fb"),
}


def work_path(path):
    """The container binds /work but not /gpfs/work, so stored paths use the /work form."""
    path = os.path.abspath(path)
    return "/work/" + path[len("/gpfs/work/"):] if path.startswith("/gpfs/work/") else path


def data_dir():
    """$IA3D_DATA, or its default from kairos/config.sh."""
    d = os.environ.get("IA3D_DATA")
    if not d:
        cfg = os.path.join(HERE, "..", "config.sh")
        d = subprocess.run(["bash", "-c", 'source "$1" && printf %s "$IA3D_DATA"', "bash", cfg],
                           check=True, capture_output=True, text=True).stdout
    return work_path(d)


def load_split(path=SPLIT):
    """{synset: [model ids]} in the order of the split file (ids are sorted within a synset)."""
    with open(path) as f:
        return {syn: list(v["samples"]) for syn, v in json.load(f).items()}


def synset_names():
    with open(SYNSET_NAMES) as f:
        return json.load(f)
