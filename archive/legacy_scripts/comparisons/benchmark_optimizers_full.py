#!/usr/bin/env python3
"""CESN Adam/L-BFGS benchmarking without retraining or editing old results.

Run from repository root:
    python3 -m CESN.analysis.comparisons.benchmark_optimizers_full
    python3 -m CESN.analysis.comparisons.benchmark_optimizers_full --full

Quick mode: 8 balanced test cases, 30/50/70% observed, no noise.
Full mode: all test cases, 10 fractions, 6 noise levels; can be expensive.
Both modes: known task, unknown-task discrete A, unknown-task softmax B.
All methods use identical frozen CESN, prefixes, restarts, and sampled noise.

Important: LBFGS 'batched' uses shared curvature for summed independent
candidate losses. Optional --independent-lbfgs tests separate histories,
but is much slower. Means reported in CSV are descriptive, not significance
claims. No optimizer sees true XY or future frames.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
import torch

from CESN.core import parametersCESN as p, cesn
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.inversion.optimizer_backend import optimize

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "CESN/outputs/models/N_900/run_01/model.npz"
OUT = ROOT / "CESN/outputs/reports/optimizer_benchmark_full"
TASKS = ("pick", "push")


def cube_xy(row):
    cube = np.asarray(row["cube_xyz"], dtype=np.float64)
    return (cube[0] if cube.ndim == 2 else cube)[:2].copy()


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_balanced(rows, n):
    """Deterministic quick selection: 2 cases per task x region group."""
    if n is None:
        return list(rows)
    groups = {(task, region): [] for task in TASKS
              for region in ("interpolation", "extrapolation")}
    for row in rows:
        key = (str(row["kind"]).lower(), str(row.get("context_region", "")))
        if key in groups:
            groups[key].append(row)
    selected = []
    per_group = max(1, (n + 3) // 4)
    for group in groups.values():
        selected.extend(group[:per_group])
    return selected[:n]


def prepare(runner, rows, frac, noise, starts, *, task_mode, logit_bias=1.0):
    """Return tensors and candidate metadata; starts shape [C,R,2].

    known: R candidates per case, true task supplied.
    discrete: R XY starts per case per task hypothesis (2R candidates).
    softmax: identical R XY starts duplicated with +/- biased logits (2R).
    """
    C, R, _ = starts.shape
    if task_mode == "known":
        xy_candidates = starts
        hypotheses = np.array([[str(row["kind"]).lower()] * R for row in rows])
    else:
        xy_candidates = np.repeat(starts, 2, axis=1)
        hypotheses = np.array([[TASKS[j % 2] for j in range(2 * R)] for _ in rows])
    H = xy_candidates.shape[1]
    B = C * H
    xy = torch.tensor(xy_candidates.reshape(B, 2), dtype=runner.dtype,
                      device=runner.device, requires_grad=True)
    task_np = np.stack([np.asarray(p.TASK_LABELS[t], np.float64)
                        for t in hypotheses.ravel()])
    task = torch.tensor(task_np, dtype=runner.dtype, device=runner.device)
    lg = None
    if task_mode == "softmax":
        logits = np.where(task_np > 0.5, logit_bias, -logit_bias)
        lg = torch.tensor(logits, dtype=runner.dtype,
                          device=runner.device, requires_grad=True)

    lengths = np.array([len(r["joints"]) for r in rows], dtype=np.int64)
    K = np.maximum(2, np.minimum(lengths, np.rint(frac * lengths).astype(np.int64)))
    maxK = int(K.max())
    target = np.zeros((B, maxK, p.JOINT_DIM), np.float64)
    # The same noisy prefix is used for every restart and EVERY optimizer.
    for i, row in enumerate(rows):
        target[i * H:(i + 1) * H, :K[i]] = noise[i][None, :K[i], :]
    target = torch.tensor(target, dtype=runner.dtype, device=runner.device)
    T_t = torch.tensor(np.repeat(lengths, H), dtype=torch.long, device=runner.device)
    K_t = torch.tensor(np.repeat(K, H), dtype=torch.long, device=runner.device)
    return dict(xy=xy, task=task, lg=lg, target=target, T=T_t, K=K_t,
                lengths=lengths, k_cases=K, C=C, H=H)


def run_method(runner, rows, frac, noise, starts, task_mode, opt_name,
               args, model):
    problem = prepare(runner, rows, frac, noise, starts, task_mode=task_mode,
                      logit_bias=args.logit_bias)
    xy, lg = problem["xy"], problem["lg"]
    params = [xy] + ([lg] if lg is not None else [])

    def candidate_losses():
        task = torch.softmax(lg, dim=1) if lg is not None else problem["task"]
        context = torch.cat([task, xy], dim=1)
        return runner.prefix_loss(context, problem["target"], problem["T"], problem["K"])

    kwargs = dict(method="lbfgs" if opt_name.startswith("lbfgs") else "adam",
                  adam_lr=args.adam_lr, adam_iters=args.adam_iters,
                  adam_patience=args.patience,
                  lbfgs_lr=args.lbfgs_lr, lbfgs_iters=args.lbfgs_iters)

    if opt_name == "lbfgs_independent":
        # One optimizer AND one curvature history for each candidate.
        # Build a loss function slicing the candidate only, while maintaining
        # separate leaf parameters: no shared LBFGS update between candidates.
        initial_xy = xy.detach().clone()
        initial_lg = lg.detach().clone() if lg is not None else None
        final_xy, final_lg = [], []
        total_s, total_grad, total_iters = 0., 0, 0
        for j in range(xy.shape[0]):
            v = initial_xy[j:j+1].clone().detach().requires_grad_(True)
            g = (initial_lg[j:j+1].clone().detach().requires_grad_(True)
                 if initial_lg is not None else None)
            def one_loss():
                t = torch.softmax(g, 1) if g is not None else problem["task"][j:j+1]
                return runner.prefix_loss(
                    torch.cat([t, v], 1), problem["target"][j:j+1],
                    problem["T"][j:j+1], problem["K"][j:j+1])
            stats = optimize([v] + ([g] if g is not None else []), one_loss, **kwargs)
            total_s += stats.elapsed_s
            total_grad += stats.gradient_evaluations
            total_iters += stats.iterations
            final_xy.append(v.detach())
            if g is not None:
                final_lg.append(g.detach())
        xy_final = torch.cat(final_xy, dim=0)
        lg_final = torch.cat(final_lg, dim=0) if final_lg else None
        elapsed, grad_evals, iters = total_s, total_grad, total_iters
        # In independent mode counts are summed across ALL starts/cases.
    else:
        stats = optimize(params, candidate_losses, **kwargs)
        xy_final = xy.detach()
        lg_final = lg.detach() if lg is not None else None
        elapsed, grad_evals, iters = (stats.elapsed_s, stats.gradient_evaluations,
                                       stats.iterations)

    with torch.no_grad():
        task_final = (torch.softmax(lg_final, 1) if lg_final is not None
                      else problem["task"])
        losses = runner.prefix_loss(torch.cat([task_final, xy_final], 1),
                                    problem["target"], problem["T"], problem["K"])
    xy_np = xy_final.cpu().numpy().reshape(problem["C"], problem["H"], 2)
    task_np = task_final.cpu().numpy().reshape(problem["C"], problem["H"], 2)
    loss_np = losses.cpu().numpy().reshape(problem["C"], problem["H"])

    records = []
    for i, row in enumerate(rows):
        j = int(np.argmin(loss_np[i]))
        xy_est = xy_np[i, j].astype(np.float64)
        probs = task_np[i, j].astype(np.float64)
        pred_task = TASKS[int(np.argmax(probs))]
        true_task = str(row["kind"]).lower()
        true_xy = cube_xy(row)
        T, K = int(problem["lengths"][i]), int(problem["k_cases"][i])
        # Future evaluation is done only AFTER inversion; the optimizer never
        # receives the future or true context.
        context = np.r_[probs, xy_est]
        predicted = np.asarray(cesn.predict_trajectory(model, context, T), np.float64)
        truth = np.asarray(row["joints"], np.float64)
        future = (float(np.sqrt(np.mean((predicted[K:] - truth[K:]) ** 2)))
                  if K < T else float("nan"))
        records.append(dict(
            trajectory=str(row["id"]), region=str(row.get("context_region", "unknown")),
            true_task=true_task, predicted_task=pred_task, task_correct=int(pred_task == true_task),
            task_mode=task_mode, optimizer=opt_name, fraction=float(frac),
            noise_sigma=float(args.active_sigma), T=T, K=K,
            true_x=float(true_xy[0]), true_y=float(true_xy[1]),
            estimated_x=float(xy_est[0]), estimated_y=float(xy_est[1]),
            xy_error_mm=float(np.linalg.norm(xy_est - true_xy) * 1000),
            prefix_loss=float(loss_np[i, j]), future_rmse=future,
            p_pick=float(probs[0]), p_push=float(probs[1]),
            # Batch timings apply to the whole batch, not each trajectory.
            batch_cases=len(rows), batch_candidates=int(xy.shape[0]),
            batch_elapsed_s=float(elapsed), batch_gradient_evals=int(grad_evals),
            batch_iterations=int(iters),
        ))
    return records


def summarize(records):
    groups = {}
    for r in records:
        for region in ("GLOBAL", r["region"]):
            key = (r["task_mode"], r["optimizer"], r["fraction"], r["noise_sigma"], region)
            groups.setdefault(key, []).append(r)
    summary = []
    for (mode, opt, frac, sigma, region), rr in sorted(groups.items()):
        xy = np.array([r["xy_error_mm"] for r in rr])
        fut = np.array([r["future_rmse"] for r in rr])
        # A mean batch time cannot be recovered by averaging detail records:
        # detail CSV repeats a shared batch measurement per trajectory.
        # Summarize only accuracy metrics here; timings appear in batches.csv.
        summary.append(dict(task_mode=mode, optimizer=opt, fraction=frac,
                            noise_sigma=sigma, region=region, n=len(rr),
                            task_accuracy=float(np.mean([r["task_correct"] for r in rr])),
                            xy_error_mean_mm=float(xy.mean()),
                            xy_error_median_mm=float(np.median(xy)),
                            prefix_loss_mean=float(np.mean([r["prefix_loss"] for r in rr])),
                            future_rmse_mean=(float(np.nanmean(fut)) if np.isfinite(fut).any()
                                              else float("nan"))))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="All test cases, ten fractions and six noise levels")
    ap.add_argument("--max-cases", type=int, default=None, help="Override number of test cases")
    ap.add_argument("--fractions", type=float, nargs="+", default=None)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", choices=["known", "discrete", "softmax"],
                    default=["known", "discrete", "softmax"])
    ap.add_argument("--independent-lbfgs", action="store_true", help="Very slow: each restart separate")
    ap.add_argument("--restarts", type=int, default=4,
                    help="XY seeds per task hypothesis (softmax gets two logit seeds each)")
    ap.add_argument("--batch-cases", type=int, default=8)
    ap.add_argument("--adam-iters", type=int, default=150)
    ap.add_argument("--adam-lr", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lbfgs-iters", type=int, default=60)
    ap.add_argument("--lbfgs-lr", type=float, default=0.8)
    ap.add_argument("--logit-bias", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    if args.restarts < 1 or args.batch_cases < 1:
        ap.error("--restarts and --batch-cases must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA unavailable; use --device cpu to test locally")

    fractions = (args.fractions if args.fractions is not None else
                 ([i / 10 for i in range(1, 11)] if args.full else [0.3, 0.5, 0.7]))
    noise_levels = (args.noise_levels if args.noise_levels is not None else
                    ([0, .001, .002, .005, .01, .02] if args.full else [0.0]))
    if not all(0 < frac <= 1 for frac in fractions) or not all(x >= 0 for x in noise_levels):
        ap.error("Fractions must be in (0, 1] and noise >= 0")

    model = cesn.load_model(MODEL)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    runner = GPUBatchedCESN(model, leak_rate=p.LEAK_RATE, device=args.device, dtype=dtype)
    test = load_split(p.TEST_SPLIT, p.DATA_PATH, joint_dim=p.JOINT_DIM)
    max_cases = args.max_cases if args.max_cases is not None else (None if args.full else 8)
    test = select_balanced(test, max_cases)
    if not test:
        raise RuntimeError("No test trajectories selected")

    train = []
    for split in p.TRAIN_SPLITS:
        train.extend(load_split(split, p.DATA_PATH, joint_dim=p.JOINT_DIM))
    bounds = np.stack([cube_xy(r) for r in train])
    xy_min, xy_max = bounds.min(axis=0), bounds.max(axis=0)
    optimizers = ["adam", "lbfgs"] + (["lbfgs_independent"] if args.independent_lbfgs else [])

    print(f"Cases={len(test)}, fractions={fractions}, noise={noise_levels}, methods={args.methods}")
    print(f"Restarts={args.restarts}, optimizers={optimizers}, device={args.device}, dtype={args.dtype}")
    print("NOTE: 'full' can be computationally expensive; outputs overwrite only this benchmark folder.", flush=True)
    details, batches = [], []
    for fi, fraction in enumerate(fractions):
        for ni, sigma in enumerate(noise_levels):
            args.active_sigma = float(sigma)
            for start in range(0, len(test), args.batch_cases):
                rows = test[start:start + args.batch_cases]
                rng = np.random.default_rng(args.seed + fi * 100003 + ni * 1009 + start)
                starts = np.empty((len(rows), args.restarts, 2), np.float64)
                starts[:, :, 0] = rng.uniform(xy_min[0], xy_max[0], size=starts.shape[:2])
                starts[:, :, 1] = rng.uniform(xy_min[1], xy_max[1], size=starts.shape[:2])
                # Generate one noisy observation per trajectory. Reuse for ALL methods and optimizers.
                noisy = []
                for row in rows:
                    y = np.asarray(row["joints"], np.float64)
                    noisy.append(y + rng.normal(0., sigma, size=y.shape))
                for mode in args.methods:
                    for opt_name in optimizers:
                        result = run_method(runner, rows, fraction, noisy, starts.copy(),
                                            mode, opt_name, args, model)
                        details.extend(result)
                        head = result[0]
                        batches.append(dict(task_mode=mode, optimizer=opt_name,
                                            fraction=float(fraction), noise_sigma=float(sigma),
                                            first_trajectory=head["trajectory"],
                                            n_cases=len(rows), n_candidates=head["batch_candidates"],
                                            elapsed_s=head["batch_elapsed_s"],
                                            gradient_evaluations=head["batch_gradient_evals"],
                                            iterations=head["batch_iterations"]))
                print(f"fraction={fraction:.2f} sigma={sigma:.4f} cases={start+len(rows)}/{len(test)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "detail.csv", details)
    write_csv(args.out_dir / "summary.csv", summarize(details))
    write_csv(args.out_dir / "batches.csv", batches)
    print("\nSaved:")
    for name in ("detail.csv", "summary.csv", "batches.csv"):
        print(" ", args.out_dir / name)
    print("Tip: compare execution time from batches.csv, not repeated times in detail.csv")


if __name__ == "__main__":
    main()
