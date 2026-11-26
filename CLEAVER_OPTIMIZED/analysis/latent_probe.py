"""Is z_dynamic usable by the CESN for *where* the interaction happens?

Linear probes fitted on train, reported on validation (median error):
  * TCP position from z_dynamic(t), per phase (reach / close / lift)
  * goal XY from z_dynamic at the end of the reach
  * temporal jitter = mean |2nd difference| / mean |1st difference| (smooth << 1, white noise ~1.7)
Probes are diagnostics only; robot state never enters the AE or the CESN.

    python -m CLEAVER_OPTIMIZED.analysis.latent_probe --ae-runs v6_ref1 v6_ref1_aux
"""
import argparse

import h5py
import numpy as np

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn.model import csv_write

# sequence steps (step 0 = frame REFERENCE_FRAME); reach ends at frame 112, close at 142
PHASES = {"reach": slice(0, 111), "close": slice(111, 141), "lift": slice(141, cfg.SEQ_LEN)}
GOAL_STEP = 105


def ridge(A, Y, lam):
    A = np.c_[A, np.ones(len(A))]
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)


def apply(W, A):
    return np.c_[A, np.ones(len(A))] @ W


def probe(path):
    train, val = set(cfg.TRAIN), set(cfg.VALIDATION)
    with h5py.File(path, "r") as f, h5py.File(cfg.RGB, "r") as src:
        X = {k: (int(f[k].attrs["pair_id"]), f[k]["z_dynamic"][:][-cfg.SEQ_LEN:]) for k in f if k.startswith("trajectory_")}
        tcp = {k: src[k]["tcp_xyz"][-cfg.SEQ_LEN:] for k in X}
        goal = {k: src[k]["target_world"][:2] for k in X}
    ktr = [k for k in X if X[k][0] in train]
    kva = [k for k in X if X[k][0] in val]
    A = np.concatenate([X[k][1] for k in ktr])
    mu, sd = A.mean(0), A.std(0) + 1e-6
    W = ridge((A - mu) / sd, np.concatenate([tcp[k] for k in ktr]), 1.0)
    row = {}
    for name, s in PHASES.items():
        e = np.concatenate([np.linalg.norm(apply(W, (X[k][1][s] - mu) / sd) - tcp[k][s], axis=1) for k in kva])
        row[f"tcp_{name}_mm"] = float(np.median(e) * 1000)
    Ag = np.stack([X[k][1][GOAL_STEP] for k in ktr])
    m2, s2 = Ag.mean(0), Ag.std(0) + 1e-6
    Wg = ridge((Ag - m2) / s2, np.stack([goal[k] for k in ktr]), 10.0)
    row["goal_xy_mm"] = float(np.median([np.linalg.norm(apply(Wg, ((X[k][1][GOAL_STEP] - m2) / s2)[None])[0] - goal[k])
                                         for k in kva]) * 1000)
    row["jitter"] = float(np.mean([np.linalg.norm(np.diff(X[k][1], 2, axis=0), axis=1).mean()
                                   / np.linalg.norm(np.diff(X[k][1], axis=0), axis=1).mean() for k in kva]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-runs", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for run in a.ae_runs:
        row = dict(ae_run=run, **probe(cfg.OUT / "autoencoder" / run / "latents.h5"))
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = cfg.OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "latent_probe.csv", rows)


if __name__ == "__main__":
    main()
