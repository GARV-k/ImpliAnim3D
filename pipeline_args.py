"""Command-line options shared by train_GT2.py, test.py, train_weights.py and inference.py.

The defaults are the values that used to be hard-coded in those scripts. A script without arguments
therefore reads and writes the same files as before; the one difference in what it does is that
train_GT2.py now also trains the shared decoder (--freeze-decoder gives the old behaviour).
"""
import argparse


def pipeline_parser(description, epochs=None, unused=()):
    # unused: options of the common set that the calling script does not read. They are left out of its
    # parser, so that passing one is an error instead of being silently ignored.
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def add(flag, **kwargs):
        if flag not in unused:
            parser.add_argument(flag, **kwargs)

    add("--n", type=int, default=2, help="numerator of the fraction of samples taken on the surface")
    add("--d", type=int, default=3, help="denominator of that fraction")
    add("--hashmap-size", type=int, default=10, help="log2 of the hash table size per level")
    add("--obj-no", type=int, default=12, help="only used in the default workspace name")
    add("--num-objs", type=int, default=25, help="number of meshes, read as 1.obj ... <num-objs>.obj")
    add("--data-root", default="data/obj_meshes",
        help="directory of the unrotated meshes; the rotated ones are read from <data-root>_90")
    add("--workspace", default=None,
        help="checkpoint directory (default: hash_workspace_obj<obj-no>_<n>_<d>_<hashmap-size>/)")
    add("--epochs", type=int, default=epochs)
    add("--steps-per-epoch", type=int, default=100)
    add("--num-samples", type=int, default=2 ** 18, help="points per mesh and step")
    add("--lr", type=float, default=1e-4)
    return parser


def as_dir(path):
    # the scripts build file names by string concatenation, so directories must end with '/'
    return path if path.endswith('/') else path + '/'


def parse(parser):
    args = parser.parse_args()
    if args.workspace is None:
        args.workspace = f'hash_workspace_obj{args.obj_no}_{args.n}_{args.d}_{args.hashmap_size}/'
    args.workspace = as_dir(args.workspace)
    if hasattr(args, 'data_root'):
        args.data_root = args.data_root.rstrip('/')
    return args
