"""Command-line options shared by train_GT2.py, test.py, train_weights.py and inference.py.

The defaults are the values that used to be hard-coded in those scripts, so running a script without
arguments does what it did before.
"""
import argparse


def pipeline_parser(description, epochs):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n", type=int, default=2, help="numerator of the fraction of samples taken on the surface")
    parser.add_argument("--d", type=int, default=3, help="denominator of that fraction")
    parser.add_argument("--hashmap-size", type=int, default=10, help="log2 of the hash table size per level")
    parser.add_argument("--obj-no", type=int, default=12, help="only used in the default workspace name")
    parser.add_argument("--num-objs", type=int, default=25, help="number of meshes, read as 1.obj ... <num-objs>.obj")
    parser.add_argument("--data-root", default="data/obj_meshes",
                        help="directory of the unrotated meshes; the rotated ones are read from <data-root>_90")
    parser.add_argument("--workspace", default=None,
                        help="checkpoint directory (default: hash_workspace_obj<obj-no>_<n>_<d>_<hashmap-size>/)")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=2 ** 18, help="points per mesh and step")
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser


def as_dir(path):
    # the scripts build file names by string concatenation, so directories must end with '/'
    return path if path.endswith('/') else path + '/'


def parse(parser):
    args = parser.parse_args()
    if args.workspace is None:
        args.workspace = f'hash_workspace_obj{args.obj_no}_{args.n}_{args.d}_{args.hashmap_size}/'
    args.workspace = as_dir(args.workspace)
    args.data_root = args.data_root.rstrip('/')
    return args
