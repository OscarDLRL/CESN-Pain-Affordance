#!/usr/bin/env python3
"""Check selected inversion API against the original benchmark implementation.

Run from repository root:
 python3 -m CESN.analysis.final.check_inversion_parity --device cuda

Compares paired TEST cases with IDENTICAL full T, observed K, starts, task
hypotheses, dtype and batched loss. True task/XY are used only for reporting
by the older benchmark, NEVER by the new unknown-task inference routine.

This checks an algorithm implementation, NOT an independent holdout result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from CESN.core import parametersCESN as p, cesn
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.analysis.comparisons.benchmark_optimizers_full import MODEL, cube_xy, select_balanced, run_method
from CESN.inversion import infer_context


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.30)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tolerance", type=float, default=1e-5,
                    help="Absolute tolerance in meters for XY; for losses use same tolerance")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA unavailable; use --device cpu")
    if args.cases < 1 or args.restarts < 1 or not 0 < args.fraction <= 1:
        ap.error("Invalid cases/restarts/fraction")

    model = cesn.load_model(MODEL)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    runner = GPUBatchedCESN(model, p.LEAK_RATE, device=args.device, dtype=dtype)
    rows = select_balanced(load_split(p.TEST_SPLIT, p.DATA_PATH, joint_dim=p.JOINT_DIM), args.cases)
    train = []
    for split in p.TRAIN_SPLITS:
        train.extend(load_split(split, p.DATA_PATH, joint_dim=p.JOINT_DIM))
    training_xy = np.stack([cube_xy(row) for row in train])
    mn, mx = training_xy.min(axis=0), training_xy.max(axis=0)
    rng = np.random.default_rng(args.seed)
    starts = np.empty((len(rows), args.restarts, 2), dtype=np.float64)
    starts[:, :, 0] = rng.uniform(mn[0], mx[0], size=starts.shape[:2])
    starts[:, :, 1] = rng.uniform(mn[1], mx[1], size=starts.shape[:2])
    full = [np.asarray(r["joints"], np.float64) for r in rows]
    T = np.array([len(y) for y in full], np.int64)
    K = np.maximum(2, np.minimum(T, np.rint(args.fraction * T).astype(np.int64)))
    prefixes = [full[i][:int(K[i])].copy() for i in range(len(rows))]

    settings = SimpleNamespace(logit_bias=1.0, adam_lr=.02, adam_iters=150,
                               patience=20, lbfgs_lr=.8, lbfgs_iters=60,
                               active_sigma=0.0)
    old = run_method(runner, rows, args.fraction, full, starts.copy(),
                     "discrete", "lbfgs", settings, model)
    new = infer_context(runner, prefixes, T, starts.copy(), method="discrete",
                        optimizer="lbfgs", lbfgs_max_iter=60, lbfgs_lr=.8)
    errors_xy = []
    errors_loss = []
    task_mismatches = 0
    for a, b in zip(old, new):
        xy_old = np.array([a["estimated_x"], a["estimated_y"]])
        d_xy = float(np.max(np.abs(xy_old - b.estimated_xy)))
        d_loss = abs(a["prefix_loss"] - b.prefix_loss)
        errors_xy.append(d_xy)
        errors_loss.append(d_loss)
        task_mismatches += int(a["predicted_task"] != b.predicted_task)
    max_xy = max(errors_xy)
    max_loss = max(errors_loss)
    print(f"Cases={len(rows)}, K fraction={args.fraction}, restarts={args.restarts}")
    print(f"Max absolute XY difference: {max_xy:.9g} m")
    print(f"Max absolute loss difference: {max_loss:.9g}")
    print(f"Task mismatches: {task_mismatches}")
    print("NOTE: Old/new runs use the SAME batch and candidate ordering.")
    if max_xy > args.tolerance or max_loss > args.tolerance or task_mismatches:
        raise SystemExit("FAIL: inversion API differs from benchmark")
    print("PASS: selected inversion reproduces benchmark for these settings")


if __name__ == "__main__":
    main()
