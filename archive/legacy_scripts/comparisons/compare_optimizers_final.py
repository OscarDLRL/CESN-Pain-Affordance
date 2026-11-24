#!/usr/bin/env python3
"""
CUDA-synchronized Adam vs batched and independent L-BFGS comparison for FINAL spatial XY inversion.

Known task; unknown cube_x, cube_y.

Fair comparison:
- same frozen CESN
- same selected test cases
- same observation fractions
- same random restart initializations
- same raw joint-space prefix loss used by the final project
- no test information is used by the optimizers beyond the observed q[0:K]

Cases:
    2 Pick interpolation
    2 Push interpolation
    2 Pick extrapolation
    2 Push extrapolation

Fractions:
    0.30, 0.50, 0.70

Output:
    CESN/outputs/reports/optimizer_comparison_validated.csv
    CESN/outputs/reports/optimizer_comparison_validated_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np
import torch

from CESN.core import parametersCESN as p
from CESN.core import cesn
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "CESN"
    / "outputs"
    / "models"
    / "N_900"
    / "run_01"
    / "model.npz"
)

OUT_DETAIL = (
    ROOT
    / "CESN"
    / "outputs"
    / "reports"
    / "optimizer_comparison_validated.csv"
)

OUT_SUMMARY = (
    ROOT
    / "CESN"
    / "outputs"
    / "reports"
    / "optimizer_comparison_validated_summary.csv"
)

FRACTIONS = (0.30, 0.50, 0.70)
RESTARTS = 4
ADAM_ITERS = 150
ADAM_LR = 0.02
LBFGS_MAX_ITER = 60
LBFGS_LR = 0.8
SEED = 20261111


def synchronize(runner):
    """Complete queued CUDA work before reading wall-clock time."""
    if runner.device.type == "cuda":
        torch.cuda.synchronize(runner.device)


def optimizer_counts(opt, parameter):
    state = opt.state.get(parameter, {})
    return int(state.get("n_iter", 0)), int(state.get("func_evals", 0))


def task_onehot(task):
    return np.asarray(
        p.TASK_LABELS[str(task).lower()],
        dtype=np.float64,
    )


def cube_xy(row):
    c = np.asarray(row["cube_xyz"], dtype=np.float64)
    c0 = c[0] if c.ndim == 2 else c
    return c0[:2].copy()


def choose_cases(rows):
    wanted = [
        ("pick", "interpolation"),
        ("push", "interpolation"),
        ("pick", "extrapolation"),
        ("push", "extrapolation"),
    ]

    selected = []

    for task, region in wanted:
        matches = [
            r for r in rows
            if str(r["kind"]).lower() == task
            and str(r.get("context_region", "")) == region
        ]
        selected.extend(matches[:2])

    return selected


def make_problem(runner, row, fraction, starts):
    y = np.asarray(row["joints"], dtype=np.float64)
    T = len(y)
    K = max(
        2,
        min(
            T,
            int(round(float(fraction) * T)),
        ),
    )

    R = len(starts)

    xy = torch.tensor(
        np.asarray(starts, dtype=np.float64),
        dtype=torch.float64,
        device=runner.device,
        requires_grad=True,
    )

    task = torch.tensor(
        np.repeat(
            task_onehot(row["kind"])[None, :],
            R,
            axis=0,
        ),
        dtype=torch.float64,
        device=runner.device,
    )

    target = torch.tensor(
        np.repeat(
            y[:K][None, :, :],
            R,
            axis=0,
        ),
        dtype=torch.float64,
        device=runner.device,
    )

    T_each = torch.full(
        (R,),
        T,
        dtype=torch.long,
        device=runner.device,
    )

    K_each = torch.full(
        (R,),
        K,
        dtype=torch.long,
        device=runner.device,
    )

    return xy, task, target, T_each, K_each, T, K


def total_loss(runner, xy, task, target, T_each, K_each):
    context = torch.cat([task, xy], dim=1)
    losses = runner.prefix_loss(
        context,
        target,
        T_each,
        K_each,
    )
    return losses.sum(), losses


def optimize_adam(runner, row, fraction, starts):
    xy, task, target, T_each, K_each, T, K = make_problem(
        runner, row, fraction, starts
    )

    optimizer = torch.optim.Adam(
        [xy],
        lr=ADAM_LR,
    )

    synchronize(runner)
    t0 = time.perf_counter()

    for _ in range(ADAM_ITERS):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = total_loss(
            runner,
            xy,
            task,
            target,
            T_each,
            K_each,
        )
        loss.backward()
        optimizer.step()

    synchronize(runner)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        _, losses = total_loss(
            runner,
            xy,
            task,
            target,
            T_each,
            K_each,
        )

    return finalize(
        runner,
        row,
        fraction,
        "adam",
        xy,
        losses,
        T,
        K,
        elapsed,
        iterations=ADAM_ITERS,
        grad_evals=ADAM_ITERS,
    )


def optimize_lbfgs(runner, row, fraction, starts):
    xy, task, target, T_each, K_each, T, K = make_problem(
        runner, row, fraction, starts
    )

    optimizer = torch.optim.LBFGS(
        [xy],
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    synchronize(runner)
    t0 = time.perf_counter()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = total_loss(
            runner,
            xy,
            task,
            target,
            T_each,
            K_each,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    synchronize(runner)
    elapsed = time.perf_counter() - t0
    n_iter, n_eval = optimizer_counts(optimizer, xy)

    with torch.no_grad():
        _, losses = total_loss(
            runner,
            xy,
            task,
            target,
            T_each,
            K_each,
        )

    return finalize(
        runner,
        row,
        fraction,
        "lbfgs",
        xy,
        losses,
        T,
        K,
        elapsed,
        iterations=n_iter,
        grad_evals=n_eval,
    )


def optimize_lbfgs_independent(runner, row, fraction, starts):
    """Optimize each restart with its OWN L-BFGS history.

    Timing covers all restarts sequentially; grad_evals is their total.
    Results are selected by the same best-of-restarts criterion.
    """
    xy_results = []
    loss_results = []
    iter_total = 0
    eval_total = 0
    synchronize(runner)
    t0 = time.perf_counter()
    for start in starts:
        xy, task, target, T_each, K_each, T, K = make_problem(
            runner, row, fraction, np.asarray(start)[None, :]
        )
        optimizer = torch.optim.LBFGS(
            [xy],
            lr=LBFGS_LR,
            max_iter=LBFGS_MAX_ITER,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = total_loss(runner, xy, task, target, T_each, K_each)
            loss.backward()
            return loss

        optimizer.step(closure)
        n_iter, n_eval = optimizer_counts(optimizer, xy)
        iter_total += n_iter
        eval_total += n_eval
        with torch.no_grad():
            _, losses = total_loss(runner, xy, task, target, T_each, K_each)
        xy_results.append(xy.detach().clone())
        loss_results.append(losses.detach().clone())

    synchronize(runner)
    elapsed = time.perf_counter() - t0
    return finalize(
        runner, row, fraction, "lbfgs_independent",
        torch.cat(xy_results, dim=0),
        torch.cat(loss_results, dim=0),
        T, K, elapsed, iterations=iter_total, grad_evals=eval_total,
    )


def finalize(
    runner,
    row,
    fraction,
    optimizer_name,
    xy_tensor,
    losses_tensor,
    T,
    K,
    elapsed,
    iterations,
    grad_evals,
):
    xy_all = xy_tensor.detach().cpu().numpy()
    losses = losses_tensor.detach().cpu().numpy()

    best = int(np.argmin(losses))
    est_xy = xy_all[best].astype(np.float64)
    true_xy = cube_xy(row).astype(np.float64)

    model = cesn.load_model(MODEL_PATH)
    context = cesn.make_context(
        str(row["kind"]).lower(),
        est_xy,
    )

    pred = np.asarray(
        cesn.predict_trajectory(
            model,
            context,
            T,
        ),
        dtype=np.float64,
    )

    truth = np.asarray(
        row["joints"],
        dtype=np.float64,
    )

    future_rmse = (
        float(
            np.sqrt(
                np.mean(
                    (pred[K:] - truth[K:]) ** 2
                )
            )
        )
        if K < T
        else np.nan
    )

    return {
        "trajectory": row["id"],
        "task": str(row["kind"]).lower(),
        "context_region": str(
            row.get("context_region", "unknown")
        ),
        "optimizer": optimizer_name,
        "fraction": float(fraction),
        "T": int(T),
        "K": int(K),
        "restarts": int(RESTARTS),
        "true_cube_x": float(true_xy[0]),
        "true_cube_y": float(true_xy[1]),
        "estimated_cube_x": float(est_xy[0]),
        "estimated_cube_y": float(est_xy[1]),
        "cube_xy_error_m": float(
            np.linalg.norm(est_xy - true_xy)
        ),
        "final_prefix_loss": float(losses[best]),
        "future_rmse": float(future_rmse),
        "elapsed_s": float(elapsed),
        "optimizer_iterations": int(iterations),
        "gradient_evaluations": int(grad_evals),
    }


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


def summarize(records):
    rows = []

    for optimizer in ("adam", "lbfgs", "lbfgs_independent"):
        for fraction in FRACTIONS:
            rr = [
                r for r in records
                if r["optimizer"] == optimizer
                and np.isclose(r["fraction"], fraction)
            ]

            rows.append(
                {
                    "optimizer": optimizer,
                    "fraction": float(fraction),
                    "n": len(rr),
                    "cube_xy_error_mean_m": float(
                        np.mean(
                            [r["cube_xy_error_m"] for r in rr]
                        )
                    ),
                    "cube_xy_error_median_m": float(
                        np.median(
                            [r["cube_xy_error_m"] for r in rr]
                        )
                    ),
                    "prefix_loss_mean": float(
                        np.mean(
                            [r["final_prefix_loss"] for r in rr]
                        )
                    ),
                    "future_rmse_mean": float(
                        np.nanmean(
                            [r["future_rmse"] for r in rr]
                        )
                    ),
                    "elapsed_mean_s": float(
                        np.mean(
                            [r["elapsed_s"] for r in rr]
                        )
                    ),
                    "gradient_evaluations_mean": float(np.mean(
                        [r["gradient_evaluations"] for r in rr]
                    )),
                    "optimizer_iterations_mean": float(np.mean(
                        [r["optimizer_iterations"] for r in rr]
                    )),
                }
            )

    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = cesn.load_model(MODEL_PATH)

    runner = GPUBatchedCESN(
        model,
        leak_rate=p.LEAK_RATE,
        device=device,
        dtype=torch.float64,
    )

    test_rows = load_split(
        "test",
        p.DATA_PATH,
        joint_dim=p.JOINT_DIM,
    )
    rows = choose_cases(test_rows)

    train_rows = []
    for split in p.TRAIN_SPLITS:
        train_rows.extend(
            load_split(
                split,
                p.DATA_PATH,
                joint_dim=p.JOINT_DIM,
            )
        )

    train_xy = np.stack(
        [cube_xy(r) for r in train_rows],
        axis=0,
    )

    xmin = float(train_xy[:, 0].min())
    xmax = float(train_xy[:, 0].max())
    ymin = float(train_xy[:, 1].min())
    ymax = float(train_xy[:, 1].max())

    rng = np.random.default_rng(SEED)

    print("=" * 96)
    print("FINAL OPTIMIZER COMPARISON — ADAM vs L-BFGS")
    print("=" * 96)
    print("Device      :", device)
    print("Cases       :", len(rows))
    print("Fractions   :", FRACTIONS)
    print("Restarts    :", RESTARTS)
    print("Adam        :", f"{ADAM_ITERS} iterations, lr={ADAM_LR}")
    print("L-BFGS      :", f"max_iter={LBFGS_MAX_ITER}, lr={LBFGS_LR}, strong_wolfe")
    print("Methods     : adam, lbfgs (batched), lbfgs_independent (sequential)")
    print("Timing      : CUDA-synchronized optimizer only; excludes final metrics")
    print()

    records = []

    for row in rows:
        for fraction in FRACTIONS:
            starts = np.column_stack(
                [
                    rng.uniform(
                        xmin,
                        xmax,
                        size=RESTARTS,
                    ),
                    rng.uniform(
                        ymin,
                        ymax,
                        size=RESTARTS,
                    ),
                ]
            ).astype(np.float64)

            adam = optimize_adam(
                runner,
                row,
                fraction,
                starts.copy(),
            )

            lbfgs = optimize_lbfgs(
                runner,
                row,
                fraction,
                starts.copy(),
            )

            independent = optimize_lbfgs_independent(
                runner, row, fraction, starts.copy(),
            )

            records.extend([adam, lbfgs, independent])

            print(
                f"{row['id']:>28s} | "
                f"K={fraction:.1f} | "
                f"Adam={adam['cube_xy_error_m']*1000:7.2f} mm | "
                f"LBFGS={lbfgs['cube_xy_error_m']*1000:7.2f} mm | "
                f"LBFGS-ind={independent['cube_xy_error_m']*1000:7.2f} mm | "
                f"evals A/B/I={adam['gradient_evaluations']}/"
                f"{lbfgs['gradient_evaluations']}/"
                f"{independent['gradient_evaluations']}"
            )

    summary = summarize(records)

    write_csv(
        OUT_DETAIL,
        records,
    )
    write_csv(
        OUT_SUMMARY,
        summary,
    )

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)

    for fraction in FRACTIONS:
        print(f"\nObserved fraction: {fraction:.0%}")
        for method in ("adam", "lbfgs", "lbfgs_independent"):
            r = next(x for x in summary if x["optimizer"] == method
                     and np.isclose(x["fraction"], fraction))
            print(
                f"  {method:18s} | "
                f"XY={r['cube_xy_error_mean_m'] * 1000:8.3f} mm | "
                f"future={r['future_rmse_mean']:.6f} | "
                f"time={r['elapsed_mean_s']:.3f}s | "
                f"grad evals={r['gradient_evaluations_mean']:.1f}"
            )

    print()
    print("Detail :", OUT_DETAIL)
    print("Summary:", OUT_SUMMARY)


if __name__ == "__main__":
    main()