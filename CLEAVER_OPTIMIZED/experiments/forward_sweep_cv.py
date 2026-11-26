"""CESN forward sweep with leave-one-location-out CV over train+validation (test untouched).

Folds: the 7 cleaver locations of train+validation. Each fold fits normalisation and
readout on 6 locations and evaluates on the held-out one.
Config: XY input gain (scales the 2 XY rows of Win) x feedback gain. C161 = (1, 1).
Metrics on the held-out location:
  * NRMSE per phase, teacher-forced and autonomous (true context)
  * SAFE/PAIN identifiability with fixed perceived XY

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
from CLEAVER_OPTIMIZED.experiments.discrete_inference import history, perceived_xy
from CLEAVER_OPTIMIZED.perception import GoalPerception, template_from, collect, fit_h

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
def evaluate(model, d, xy):
    y = d["yn"]
    c = model.context(torch.as_tensor(d["label"], device=y.device), d["static_n"], d["xyn"])
    tf = model.predict(c, y.shape[1], y)
    au = model.predict(c, y.shape[1])
    row = {}
    for ph, (s, e) in PHASES.items():
        row[f"tf_{ph}"] = float((tf[:, s:e] - y[:, s:e]).square().mean().sqrt())
        row[f"auto_{ph}"] = float((au[:, s:e] - y[:, s:e]).square().mean().sqrt())
    L=history(model,d,xy,1).cpu().numpy()
    for wn, (s,e) in WINDOWS.items():
        row[f'acc_{wn}']=100*float(np.mean(L[:,e-1].argmin(1)==d['label']))
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
        keys=set(data['train']['keys'])
        with h5py.File(cfg.RGB,'r') as f:
            rgb=f[data['train']['keys'][0]]['rgb']; tmpl=template_from(rgb[0],rgb[cfg.REFERENCE_FRAME])
        raw=collect(cfg.RGB,tmpl)
        perception=GoalPerception(tmpl,fit_h([r for r in raw if r['key'] in keys]))
        xy,_=perceived_xy(data['validation'],perception,lo,hi)
        prepared.append((obj,data,xy))
    rows, summary = [], []
    for xy_gain, fb_gain, leak, radius, ridge in itertools.product(a.xy_gains, a.fb_gains, a.leaks, a.radii, a.ridges):
        t0 = time.perf_counter()
        fold_rows = []
        for obj, data, xy in prepared:
            model = build(device, xy_gain, fb_gain, leak, radius)
            model.fit(data["train"], ridge)
            r = dict(xy_gain=xy_gain, fb_gain=fb_gain, leak=leak, radius=radius, ridge=ridge, held_out_object=obj,
                     **evaluate(model, data["validation"], xy))
            fold_rows.append(r)
        rows.extend(fold_rows)
        keys = [k for k in fold_rows[0] if k not in ("xy_gain", "fb_gain", "leak", "radius", "ridge", "held_out_object", "inside_train_bounds")]
        s = dict(xy_gain=xy_gain, fb_gain=fb_gain, leak=leak, radius=radius, ridge=ridge, **{k: float(np.mean([r[k] for r in fold_rows])) for k in keys})
        summary.append(s)
        cesn.csv_write(out / "per_fold.csv", rows)
        cesn.csv_write(out / "summary.csv", summary)
        print(f"xy={xy_gain:<4} fb={fb_gain:<4} leak={leak:<5} rad={radius:<4} ridge={ridge:<4} | TF reach/close/lift {s['tf_reach']:.2f}/{s['tf_close']:.2f}/{s['tf_lift']:.2f} "
              f"| auto {s['auto_reach']:.2f}/{s['auto_close']:.2f}/{s['auto_lift']:.2f} "
              f"| acc w10/23 {s['acc_w10']:.0f}/{s['acc_w23']:.0f} | {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
