"""CESN forward sweep with leave-one-location-out CV over train+validation (test untouched).

Folds: the 7 cleaver locations of train+validation. Each fold fits normalisation and
readout on 6 locations and evaluates on the held-out one.
Config: XY input gain (scales the 2 XY rows of Win) x feedback gain. C161 = (1, 1).
Metrics on the held-out location:
  * NRMSE per phase, teacher-forced and autonomous (true context)
  * identifiability on a grid (same teacher-forced loss as the inversion):
    XY error of the true-label argmin, and label correctness (best loss of the true
    label < best loss of the other label), per prefix window.

    python -m CLEAVER_OPTIMIZED.experiments.forward_sweep_cv --ae-run v6_ref1
"""
import argparse
import itertools
import math
import time

import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn

PHASES = {"reach": (0, 111), "close": (111, 141), "lift": (141, cfg.SEQ_LEN)}
WINDOWS = {"w10": (0, 10), "w23": (0, 23), "w67": (0, 67), "w111": (0, 111), "w222": (0, cfg.SEQ_LEN)}


def take(d, mask):
    idx = np.flatnonzero(mask)
    return {k: ([v[i] for i in idx] if k == "keys" else v[idx]) for k, v in d.items()}


def pooled_folds(latents):
    data = cesn.load_latents(latents)
    a, b = data["train"], data["validation"]
    pool = {k: (a[k] + b[k] if k == "keys" else np.concatenate([a[k], b[k]])) for k in ("keys", "pair", "label", "static", "z", "xy", "contact")}
    with h5py.File(cfg.RGB, "r") as f:
        pool["object"] = np.array([int(f[k].attrs["object_id"]) for k in pool["keys"]])
    for obj in sorted(set(pool["object"])):
        yield obj, {"train": take(pool, pool["object"] != obj), "validation": take(pool, pool["object"] == obj)}


_WEIGHTS = {}


def build(device, xy_gain, fb_gain, leak=cfg.LEAK_RATE, radius=cfg.SPECTRAL_RADIUS):
    if device not in _WEIGHTS:
        _WEIGHTS[device] = cesn.random_weights(cfg.RESERVOIR_SIZE, cfg.SEED, device, cfg.STATIC_LATENT_DIM, cfg.DYNAMIC_LATENT_DIM)
    w = dict(_WEIGHTS[device])
    w["Win"] = w["Win"].clone()
    w["Win"][-2:] *= xy_gain
    return cesn.CESN(w, radius, leak, fb_gain)


@torch.no_grad()
def evaluate(model, d, lo, hi, grid):
    y = d["yn"]
    c = model.context(torch.as_tensor(d["label"], device=y.device), d["static_n"], d["xyn"])
    tf = model.predict(c, y.shape[1], y)
    au = model.predict(c, y.shape[1])
    row = {}
    for ph, (s, e) in PHASES.items():
        row[f"tf_{ph}"] = float((tf[:, s:e] - y[:, s:e]).square().mean().sqrt())
        row[f"auto_{ph}"] = float((au[:, s:e] - y[:, s:e]).square().mean().sqrt())
    G = len(grid)
    xy_err = {w: [] for w in WINDOWS}
    label_ok = {w: [] for w in WINDOWS}
    for i in range(len(d["keys"])):
        lab = int(d["label"][i])
        cand = grid.repeat(2, 1)
        labs = torch.cat([torch.zeros(G), torch.ones(G)]).to(y.device)
        cc = model.context(labs, d["static_n"][i:i + 1].expand(2 * G, -1), cand)
        obs = y[i:i + 1].expand(2 * G, -1, -1)
        err = (model.predict(cc, y.shape[1], obs) - obs).square().mean(2)
        true_xy = lo + (d["xyn"][i] + 1) * 0.5 * (hi - lo)
        for wn, (s, e) in WINDOWS.items():
            loss = err[:, s:e].mean(1).view(2, G)
            j = int(loss[lab].argmin())
            est = lo + (grid[j] + 1) * 0.5 * (hi - lo)
            xy_err[wn].append(float(torch.linalg.vector_norm(est - true_xy) * 1000))
            label_ok[wn].append(int(loss[lab].min() < loss[1 - lab].min()))
    for wn in WINDOWS:
        row[f"xy_{wn}_mm"] = float(np.mean(xy_err[wn]))
        row[f"acc_{wn}"] = 100 * float(np.mean(label_ok[wn]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-run", required=True)
    ap.add_argument("--xy-gains", nargs="+", type=float, default=[1, 3, 10])
    ap.add_argument("--fb-gains", nargs="+", type=float, default=[0, 0.25, 0.5, 1])
    ap.add_argument("--leaks", nargs="+", type=float, default=[cfg.LEAK_RATE])
    ap.add_argument("--radii", nargs="+", type=float, default=[cfg.SPECTRAL_RADIUS])
    ap.add_argument("--ridges", nargs="+", type=float, default=[cfg.RIDGE])
    ap.add_argument("--name", default="forward_sweep_cv")
    ap.add_argument("--grid-mm", type=float, default=10.0)
    a = ap.parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    folder = cfg.OUT / "autoencoder" / a.ae_run
    out = folder / a.name
    out.mkdir(exist_ok=True)
    folds = list(pooled_folds(folder / "latents.h5"))
    print("folds (held-out object_id):", [o for o, _ in folds], flush=True)
    prepared = []
    for obj, data in folds:
        info, lo, hi = cesn.normalize(data, device)
        axes = [torch.linspace(-1, 1, max(2, math.ceil(float(w) * 1000 / a.grid_mm) + 1), device=device) for w in hi - lo]
        held_xy = data["validation"]["xy"][0]
        inside = bool(np.all(held_xy >= lo.cpu().numpy() - 1e-6) and np.all(held_xy <= hi.cpu().numpy() + 1e-6))
        prepared.append((obj, data, lo, hi, torch.cartesian_prod(*axes), inside))
    rows, summary = [], []
    for xy_gain, fb_gain, leak, radius, ridge in itertools.product(a.xy_gains, a.fb_gains, a.leaks, a.radii, a.ridges):
        t0 = time.perf_counter()
        fold_rows = []
        for obj, data, lo, hi, grid, inside in prepared:
            model = build(device, xy_gain, fb_gain, leak, radius)
            model.fit(data["train"], ridge)
            r = dict(xy_gain=xy_gain, fb_gain=fb_gain, leak=leak, radius=radius, ridge=ridge, held_out_object=obj, inside_train_bounds=int(inside),
                     **evaluate(model, data["validation"], lo, hi, grid))
            fold_rows.append(r)
        rows.extend(fold_rows)
        keys = [k for k in fold_rows[0] if k not in ("xy_gain", "fb_gain", "leak", "radius", "ridge", "held_out_object", "inside_train_bounds")]
        s = dict(xy_gain=xy_gain, fb_gain=fb_gain, leak=leak, radius=radius, ridge=ridge, **{k: float(np.mean([r[k] for r in fold_rows])) for k in keys})
        summary.append(s)
        cesn.csv_write(out / "per_fold.csv", rows)
        cesn.csv_write(out / "summary.csv", summary)
        print(f"xy={xy_gain:<4} fb={fb_gain:<4} leak={leak:<5} rad={radius:<4} ridge={ridge:<4} | TF reach/close/lift {s['tf_reach']:.2f}/{s['tf_close']:.2f}/{s['tf_lift']:.2f} "
              f"| auto {s['auto_reach']:.2f}/{s['auto_close']:.2f}/{s['auto_lift']:.2f} "
              f"| XY mm w10/23/67/111/222 {s['xy_w10_mm']:.0f}/{s['xy_w23_mm']:.0f}/{s['xy_w67_mm']:.0f}/{s['xy_w111_mm']:.0f}/{s['xy_w222_mm']:.0f} "
              f"| acc w10/23 {s['acc_w10']:.0f}/{s['acc_w23']:.0f} | {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
