"""Environment check for ImpliAnim3D on KAIROS.

Run from the repo root, inside the container with the venv active, on a GPU node:
    kairos/in_env.sh python kairos/env/check_env.py
Exercises every third-party piece the SDF pipeline uses (on the GPU where it applies), then the repo's own
model code. Exits non-zero if any check fails.
"""
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.getcwd())  # the repo root: its modules are imported by the last group of checks

import numpy as np
import torch

results = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            results.append((name, True, detail or ""))
        except Exception:
            results.append((name, False, traceback.format_exc().strip().splitlines()[-1]))
            traceback.print_exc()
        return fn
    return deco


def sphere_mesh(radius=0.5):
    import trimesh
    return trimesh.creation.icosphere(subdivisions=4, radius=radius)


@check("torch + CUDA")
def _():
    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(1024, 1024, device="cuda")
    assert torch.isfinite((x @ x).sum())
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}, numpy {np.__version__}"


@check("tinycudann (FullyFusedMLP + HashGrid, forward/backward)")
def _():
    import tinycudann as tcnn
    net = tcnn.Network(32, 1, {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                               "n_neurons": 64, "n_hidden_layers": 2})
    enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
                            "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 1.3819})
    x = torch.rand(4096, 3, device="cuda")
    y = net(enc(x)).float()
    y.abs().mean().backward()
    g_net, g_enc = net.params.grad, enc.params.grad
    assert y.shape == (4096, 1) and torch.isfinite(y).all()
    assert g_net is not None and g_net.abs().sum() > 0 and g_enc is not None and g_enc.abs().sum() > 0
    from importlib.metadata import version
    return f"tinycudann {version('tinycudann')}, {net.params.numel()} MLP params"


@check("pysdf (signed distance of a sphere)")
def _():
    import pysdf
    m = sphere_mesh(0.5)
    f = pysdf.SDF(m.vertices, m.faces)
    p = np.random.default_rng(0).uniform(-1, 1, (20000, 3)).astype(np.float32)
    err = np.abs(f(p) - (0.5 - np.linalg.norm(p, axis=1))).max()  # pysdf is positive inside
    assert err < 5e-3, f"max error {err}"
    return f"max |error| vs analytic {err:.1e}"


@check("PyMCubes (marching cubes)")
def _():
    import mcubes
    g = np.linspace(-1, 1, 96, dtype=np.float32)
    u = np.sqrt(sum(c ** 2 for c in np.meshgrid(g, g, g, indexing="ij"))) - 0.5
    v, t = mcubes.marching_cubes(u, 0.0)
    r = np.linalg.norm(v / 95 * 2 - 1, axis=1)
    assert len(t) > 1000 and abs(r.mean() - 0.5) < 5e-3, f"mean radius {r.mean()}"
    return f"{len(v)} vertices, mean radius {r.mean():.4f}"


@check("libigl + scikit-image (close.py and the data tools)")
def _():
    import igl
    from skimage import measure
    m = sphere_mesh(0.5)
    g = np.linspace(-0.9, 0.9, 48)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    d = igl.signed_distance(pts, np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64))[0]
    err = np.abs(d - (np.linalg.norm(pts, axis=1) - 0.5)).max()  # igl is negative inside
    assert err < 5e-3, f"max error {err}"
    v, f, _, _ = measure.marching_cubes(d.reshape(48, 48, 48), 0.0)
    assert len(f) > 1000
    return f"igl max |error| {err:.1e}, skimage {len(f)} faces"


@check("logging / utility packages")
def _():
    import cv2, jaxtyping, lpips, matplotlib, pandas, rich, tensorboardX, torch_ema, torchmetrics, tqdm, trimesh
    matplotlib.use("Agg")
    p = torch.nn.Parameter(torch.zeros(3))  # torch_ema only keeps weak references to the parameters
    ema = torch_ema.ExponentialMovingAverage([p], decay=0.95)
    ema.update()
    with tempfile.TemporaryDirectory() as d:
        w = tensorboardX.SummaryWriter(d)
        w.add_scalar("x", 1.0, 0)
        w.close()
        assert any(f.startswith("events.out") for f in os.listdir(d))
        # kairos/jobs/verify_smoke.py reads the losses back with the container's tensorboard
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(d)
        acc.Reload()
        assert [e.value for e in acc.Scalars("x")] == [1.0]
    return (f"cv2 {cv2.__version__}, trimesh {trimesh.__version__}, pandas {pandas.__version__}, "
            f"tensorboardX {tensorboardX.__version__}, jaxtyping {jaxtyping.__version__}")


@check("repo: hash encoding + shared decoder, forward/backward, W applied to the table")
def _():
    from hash_imports.encodings import HashEncoding  # noqa: F401  (the vendored nerfstudio encoding)
    from loss import mape_loss
    from sdf.meshsdf_decoder import MeshSDFDecoder
    from sdf.network_tcnn import SDFNetwork
    dec = MeshSDFDecoder(num_layers=8, skips=[], hidden_dim=128, clip_sdf=None, in_dim=32, out_dim=1)
    model = SDFNetwork(dec, encoding="hashgrid", hashmap_size=10).cuda()
    x = torch.rand(2 ** 14, 3, device="cuda") * 2 - 1
    y = torch.rand(2 ** 14, 1, device="cuda") - 0.5
    pred = model(x)
    mape_loss(pred, y).backward()
    table = model.encoder.hash_table
    assert pred.shape == (2 ** 14, 1) and table.grad is not None and table.grad.abs().sum() > 0
    assert all(p.grad is not None for p in dec.parameters())
    with torch.no_grad():
        # W = identity must reproduce the table bit for bit. It does not when float32 matmuls run in TF32,
        # which the NVIDIA image enables by default and kairos/env/activate.sh turns off.
        eye = torch.eye(table.shape[0], device="cuda")
        new_table, pred_w = model(x, "normal", eye, None, None, None, None, True)
        assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul is on: source kairos/env/activate.sh"
        assert torch.equal(new_table, table), f"table^T @ I differs from the table by {(new_table - table).abs().max():.1e}"
        assert torch.equal(pred_w, pred), f"prediction changed by {(pred_w - pred).abs().max():.1e}"
    return f"table {tuple(table.shape)}, decoder {sum(p.numel() for p in dec.parameters())} params, W = I is exact"


@check("repo: trainer and dataset modules import")
def _():
    from sdf.provider import SDFDataset  # noqa: F401
    from sdf.utils2 import Trainer, extract_geometry  # noqa: F401


width = max(len(n) for n, _, _ in results)
print()
for name, ok, detail in results:
    print(f"{'ok  ' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
if not all(ok for _, ok, _ in results):
    print("ENV_CHECK_FAILED")
    sys.exit(1)
print("ENV_CHECK_OK")
