from __future__ import annotations

# This patch file is intended to REPLACE the current
# CESN/analysis/method_a_lbfgs_v6_native256_ridge05_final70_gpu.py
#
# It is the same final Method A implementation, with one analytical change:
# summarize_errors() now also returns std/min/max, and those fields are written
# into the per-history CSV/JSON rows.

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch

from CLEAVER.core.cesn import ContextualESNTorch
from CLEAVER.core.parameters_explicit import (
    LATENT_DATA_PATH,
    RGB_DATA_PATH,
    OUT_DIR,
    MODEL_PATH,
    STATIC_PCA_DIM,
    GRASP_X_SCALE,
    GRASP_Y_SCALE,
    SAFE_TARGET_LOCAL_X,
    PAIN_TARGET_LOCAL_X,
    TIME_INDEX_SCALE,
)

RESULT_DIR = OUT_DIR / "method_a_lbfgs_native256_latent200_ridge05_final70"

SUMMARY_CSV = RESULT_DIR / "method_a_lbfgs_native256_latent200_ridge05_final70_summary.csv"
SUMMARY_JSON = RESULT_DIR / "method_a_lbfgs_native256_latent200_ridge05_final70_summary.json"

HISTORIES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]


def atanh_safe(x):
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def transform_static(z_static, mean, components):
    return ((z_static - mean) @ components.T).astype(np.float32)


def task_restart_points(label):
    from CLEAVER.config import RESTARTS
    return RESTARTS


def load_test_sequences(test_pairs):
    test_pairs = set(int(x) for x in test_pairs)
    rows = []

    with h5py.File(LATENT_DATA_PATH, "r") as latent_h5, h5py.File(RGB_DATA_PATH, "r") as rgb_h5:
        for key in sorted(latent_h5.keys()):
            if not str(key).startswith("trajectory_"):
                continue

            lg = latent_h5[key]
            pair_id = int(lg.attrs["pair_id"])
            if pair_id not in test_pairs:
                continue

            label = int(lg.attrs["pain_label"])
            obj_xyz = np.asarray(rgb_h5[key]["object_xyz"][0], dtype=np.float32)
            knife_x = float(obj_xyz[0])
            knife_y = float(obj_xyz[1])

            true_x, true_y = map(float, rgb_h5[key]["target_world"][:2])

            rows.append({
                "key": str(key),
                "pair_id": pair_id,
                "pain_label": label,
                "z_static": np.asarray(lg["z_static"], dtype=np.float32),
                "z_dynamic": np.asarray(lg["z_dynamic"], dtype=np.float32),
                "true_grasp_x": float(true_x),
                "true_grasp_y": float(true_y),
                "pair_contact_frame": int(np.flatnonzero(np.asarray(rgb_h5[f"trajectory_{pair_id*2+1:03d}"]["contact_impulses"])[:,:2].sum(axis=1)>1e-4)[0]),
            })

    return rows


def build_base_context_batch(label, z_static_pca, grasp_x, grasp_y, model):
    B = int(grasp_x.shape[0])

    task_one = torch.tensor(
        [1.0, 0.0] if int(label) == 0 else [0.0, 1.0],
        device=model.device,
        dtype=model.dtype,
    )

    task = task_one[None, :].expand(B, -1)
    static = torch.as_tensor(
        z_static_pca,
        device=model.device,
        dtype=model.dtype,
    )[None, :].expand(B, -1)

    grasp = torch.stack(
        [grasp_x / GRASP_X_SCALE, grasp_y / GRASP_Y_SCALE],
        dim=1,
    )

    base = torch.cat([task, static, grasp], dim=1)

    expected = model.input_dim - 1
    if int(base.shape[1]) != int(expected):
        raise RuntimeError(f"Expected base context dim {expected}, got {base.shape[1]}")

    return base


def predict_batch(model, base_contexts, T):
    B = int(base_contexts.shape[0])

    state = torch.zeros(
        B,
        model.reservoir_size,
        device=model.device,
        dtype=model.dtype,
    )

    outputs = []

    for t in range(T):
        time_col = torch.full(
            (B, 1),
            float(t) / float(TIME_INDEX_SCALE),
            device=model.device,
            dtype=model.dtype,
        )

        contexts = torch.cat([base_contexts, time_col], dim=1)

        pre = contexts @ model.Win.T + state @ model.W.T + model.bias
        candidate = torch.tanh(pre)

        state = (
            (1.0 - model.leak_rate) * state
            + model.leak_rate * candidate
        )

        ones = torch.ones(B, 1, device=model.device, dtype=model.dtype)
        features = torch.cat([ones, contexts, state], dim=1)
        outputs.append(features @ model.Wout)

    return torch.stack(outputs, dim=1)


def optimize_xy_batched(
    model,
    observed,
    z_static_pca,
    label,
    lbfgs_max_iter,
    lbfgs_history_size,
):
    obs = torch.as_tensor(observed, device=model.device, dtype=model.dtype)
    T = int(obs.shape[0])

    starts = torch.tensor(
        task_restart_points(label),
        device=model.device,
        dtype=model.dtype,
    )

    nx = starts[:, 0] / GRASP_X_SCALE
    ny = starts[:, 1] / GRASP_Y_SCALE

    raw = torch.stack([atanh_safe(nx), atanh_safe(ny)], dim=1).detach().clone()
    raw.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [raw],
        lr=1.0,
        max_iter=int(lbfgs_max_iter),
        max_eval=int(lbfgs_max_iter * 2),
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=int(lbfgs_history_size),
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)

        grasp_x = torch.tanh(raw[:, 0]) * GRASP_X_SCALE
        grasp_y = torch.tanh(raw[:, 1]) * GRASP_Y_SCALE

        base_context = build_base_context_batch(
            label,
            z_static_pca,
            grasp_x,
            grasp_y,
            model,
        )

        pred = predict_batch(model, base_context, T)

        losses = torch.mean(
            (pred - obs[None, :, :]) ** 2,
            dim=(1, 2),
        )

        objective = losses.sum()
        objective.backward()
        return objective

    optimizer.step(closure)

    with torch.no_grad():
        grasp_x = torch.tanh(raw[:, 0]) * GRASP_X_SCALE
        grasp_y = torch.tanh(raw[:, 1]) * GRASP_Y_SCALE

        base_context = build_base_context_batch(
            label,
            z_static_pca,
            grasp_x,
            grasp_y,
            model,
        )

        pred = predict_batch(model, base_context, T)

        losses = torch.mean(
            (pred - obs[None, :, :]) ** 2,
            dim=(1, 2),
        )

        idx = int(torch.argmin(losses).item())

        return (
            float(grasp_x[idx].item()),
            float(grasp_y[idx].item()),
            float(losses[idx].item()),
        )


def fixed_xy_task_losses(model, observed, z_static_pca, true_x, true_y):
    obs = torch.as_tensor(observed, device=model.device, dtype=model.dtype)
    T = int(obs.shape[0])

    gx = torch.tensor([true_x], device=model.device, dtype=model.dtype)
    gy = torch.tensor([true_y], device=model.device, dtype=model.dtype)

    out = {}

    with torch.no_grad():
        for label in [0, 1]:
            base_context = build_base_context_batch(
                label,
                z_static_pca,
                gx,
                gy,
                model,
            )

            pred = predict_batch(model, base_context, T)[0]
            loss = torch.mean((pred - obs) ** 2)
            out[label] = float(loss.item())

    return out[0], out[1]


def xy_error_mm(pred_x, pred_y, true_x, true_y):
    return math.sqrt(
        (pred_x - true_x) ** 2
        + (pred_y - true_y) ** 2
    ) * 1000.0


def summarize_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)

    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
        "min": float(np.min(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
        "within_5": float(np.mean(errors <= 5.0)),
        "within_10": float(np.mean(errors <= 10.0)),
        "within_20": float(np.mean(errors <= 20.0)),
    }


def evaluate_history(pct, lbfgs_max_iter, lbfgs_history_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = ContextualESNTorch.from_state_dict(
        checkpoint["model_state"],
        device=device,
    )

    pca_mean = checkpoint["pca_mean"].cpu().numpy().astype(np.float32)
    pca_components = checkpoint["pca_components"].cpu().numpy().astype(np.float32)

    sequences = load_test_sequences(checkpoint["test_pairs"])

    known_xy_errors = []
    joint_xy_errors = []

    joint_task_correct = []
    precontact_correct = []
    precontact_xy = []
    true_xy_task_correct = []

    safe_joint_correct = []
    pain_joint_correct = []

    safe_truexy_correct = []
    pain_truexy_correct = []

    for item in sequences:
        T_full = int(item["z_dynamic"].shape[0])

        n_obs = max(
            2,
            int(np.ceil(T_full * pct / 100.0)),
        )
        n_obs = min(n_obs, T_full)

        observed = item["z_dynamic"][:n_obs]

        z_static_pca = transform_static(
            item["z_static"],
            pca_mean,
            pca_components,
        )

        true_label = int(item["pain_label"])
        true_x = float(item["true_grasp_x"])
        true_y = float(item["true_grasp_y"])

        # Known TASK -> infer XY
        known_x, known_y, _ = optimize_xy_batched(
            model,
            observed,
            z_static_pca,
            true_label,
            lbfgs_max_iter,
            lbfgs_history_size,
        )

        known_xy_errors.append(
            xy_error_mm(known_x, known_y, true_x, true_y)
        )

        # Unknown TASK + unknown XY
        safe_x, safe_y, safe_loss = optimize_xy_batched(
            model,
            observed,
            z_static_pca,
            0,
            lbfgs_max_iter,
            lbfgs_history_size,
        )

        pain_x, pain_y, pain_loss = optimize_xy_batched(
            model,
            observed,
            z_static_pca,
            1,
            lbfgs_max_iter,
            lbfgs_history_size,
        )

        if safe_loss <= pain_loss:
            joint_label = 0
            joint_x, joint_y = safe_x, safe_y
        else:
            joint_label = 1
            joint_x, joint_y = pain_x, pain_y

        joint_ok = int(joint_label == true_label)
        joint_task_correct.append(joint_ok)
        if n_obs <= item["pair_contact_frame"]:
            precontact_correct.append(joint_ok)
            precontact_xy.append(xy_error_mm(joint_x, joint_y, true_x, true_y))

        if true_label == 0:
            safe_joint_correct.append(joint_ok)
        else:
            pain_joint_correct.append(joint_ok)

        joint_xy_errors.append(
            xy_error_mm(joint_x, joint_y, true_x, true_y)
        )

        # Unknown TASK + true XY
        fixed_safe_loss, fixed_pain_loss = fixed_xy_task_losses(
            model,
            observed,
            z_static_pca,
            true_x,
            true_y,
        )

        fixed_label = 0 if fixed_safe_loss <= fixed_pain_loss else 1
        fixed_ok = int(fixed_label == true_label)
        true_xy_task_correct.append(fixed_ok)

        if true_label == 0:
            safe_truexy_correct.append(fixed_ok)
        else:
            pain_truexy_correct.append(fixed_ok)

    known = summarize_errors(known_xy_errors)
    joint = summarize_errors(joint_xy_errors)

    return {
        "history_percent": int(pct),
        "precontact_n": len(precontact_correct),
        "precontact_task_accuracy": float(np.mean(precontact_correct)) if precontact_correct else None,
        "precontact_xy_mean_mm": float(np.mean(precontact_xy)) if precontact_xy else None,

        "known_task_xy_mean_mm": known["mean"],
        "known_task_xy_std_mm": known["std"],
        "known_task_xy_min_mm": known["min"],
        "known_task_xy_median_mm": known["median"],
        "known_task_xy_p90_mm": known["p90"],
        "known_task_xy_max_mm": known["max"],
        "known_task_xy_within_10": known["within_10"],
        "known_task_xy_within_20": known["within_20"],

        "joint_task_accuracy": float(np.mean(joint_task_correct)),
        "joint_safe_accuracy": float(np.mean(safe_joint_correct)),
        "joint_pain_accuracy": float(np.mean(pain_joint_correct)),

        "joint_xy_mean_mm": joint["mean"],
        "joint_xy_std_mm": joint["std"],
        "joint_xy_min_mm": joint["min"],
        "joint_xy_median_mm": joint["median"],
        "joint_xy_p90_mm": joint["p90"],
        "joint_xy_max_mm": joint["max"],
        "joint_xy_within_10": joint["within_10"],
        "joint_xy_within_20": joint["within_20"],

        "true_xy_task_accuracy": float(np.mean(true_xy_task_correct)),
        "true_xy_safe_accuracy": float(np.mean(safe_truexy_correct)),
        "true_xy_pain_accuracy": float(np.mean(pain_truexy_correct)),
    }


def worker(job):
    return evaluate_history(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lbfgs-max-iter", type=int, default=80)
    ap.add_argument("--lbfgs-history-size", type=int, default=20)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("METHOD A - FINAL NATIVE256 LATENT200 RIDGE05 | MEAN ± SD ENABLED")
    print("=" * 120)

    jobs = [
        (pct, args.lbfgs_max_iter, args.lbfgs_history_size)
        for pct in HISTORIES
    ]

    rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, job) for job in jobs]

        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)

            print(
                f"[DONE] {row['history_percent']:3d}% "
                f"| TASK={100*row['joint_task_accuracy']:6.2f}% "
                f"| joint XY mean={row['joint_xy_mean_mm']:7.2f} ± "
                f"{row['joint_xy_std_mm']:6.2f} mm "
                f"| known XY mean={row['known_task_xy_mean_mm']:7.2f} ± "
                f"{row['known_task_xy_std_mm']:6.2f} mm"
            )

    rows = sorted(rows, key=lambda r: r["history_percent"])

    fieldnames = list(rows[0].keys())

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    SUMMARY_JSON.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", SUMMARY_CSV.resolve())
    print("Saved:", SUMMARY_JSON.resolve())


from CLEAVER.config import RGB, LATENT, EXPLICIT_DIR
RGB_DATA_PATH = RGB
LATENT_DATA_PATH = LATENT
OUT_DIR = EXPLICIT_DIR
MODEL_PATH = OUT_DIR / "cesn.pt"
RESULT_DIR = OUT_DIR / "inversion"
SUMMARY_CSV = RESULT_DIR / "summary.csv"
SUMMARY_JSON = RESULT_DIR / "summary.json"

if __name__ == "__main__":
    main()
