from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from CESN.analysis.method_a_lbfgs_v6_native256_latent200_ridge05_final70_gpu import (
    load_test_sequences,
    transform_static,
    optimize_xy_batched,
    fixed_xy_task_losses,
    xy_error_mm,
)
from CESN.core.cesn import ContextualESNTorch
from CESN.core.parameters_grasp_context_v6_native256_latent200_ridge05_final70 import (
    MODEL_PATH,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT / "CESN" / "outputs"
    / "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70"
    / "context_stability_diagnostics"
)

HISTORIES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]


def safe_div(a, b, eps=1e-12):
    return a / max(abs(b), eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbfgs-max-iter", type=int, default=80)
    ap.add_argument("--lbfgs-history-size", type=int, default=20)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_case_dir = OUT_DIR / "flip_cases"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = ContextualESNTorch.from_state_dict(
        checkpoint["model_state"],
        device=device,
    )

    pca_mean = checkpoint["pca_mean"].cpu().numpy().astype(np.float32)
    pca_components = checkpoint["pca_components"].cpu().numpy().astype(np.float32)

    sequences = load_test_sequences(checkpoint["test_pairs"])

    rows = []

    print("=" * 118)
    print("CONTEXT STABILITY DIAGNOSTICS | native256 latent200 ridge0.5")
    print("=" * 118)
    print("Histories:", HISTORIES)
    print("Test trajectories:", len(sequences))
    print("=" * 118)

    for item in sequences:
        z_static_pca = transform_static(
            item["z_static"],
            pca_mean,
            pca_components,
        )

        true_label = int(item["pain_label"])
        true_x = float(item["true_grasp_x"])
        true_y = float(item["true_grasp_y"])
        T_full = int(item["z_dynamic"].shape[0])

        for pct in HISTORIES:
            n_obs = max(2, int(np.ceil(T_full * pct / 100.0)))
            n_obs = min(n_obs, T_full)
            observed = item["z_dynamic"][:n_obs]

            safe_x, safe_y, safe_loss = optimize_xy_batched(
                model,
                observed,
                z_static_pca,
                0,
                args.lbfgs_max_iter,
                args.lbfgs_history_size,
            )

            pain_x, pain_y, pain_loss = optimize_xy_batched(
                model,
                observed,
                z_static_pca,
                1,
                args.lbfgs_max_iter,
                args.lbfgs_history_size,
            )

            if safe_loss <= pain_loss:
                pred_label = 0
                pred_x, pred_y = safe_x, safe_y
                winning_loss = safe_loss
                losing_loss = pain_loss
            else:
                pred_label = 1
                pred_x, pred_y = pain_x, pain_y
                winning_loss = pain_loss
                losing_loss = safe_loss

            correct = int(pred_label == true_label)

            true_hyp_loss = safe_loss if true_label == 0 else pain_loss
            wrong_hyp_loss = pain_loss if true_label == 0 else safe_loss

            # Positive => true hypothesis has lower loss.
            signed_margin = (
                wrong_hyp_loss - true_hyp_loss
            ) / max(
                abs(wrong_hyp_loss) + abs(true_hyp_loss),
                1e-12,
            )

            # Magnitude of winner-vs-runner-up separation, independent of correctness.
            winner_margin = (
                losing_loss - winning_loss
            ) / max(
                abs(losing_loss) + abs(winning_loss),
                1e-12,
            )

            joint_xy_error = xy_error_mm(
                pred_x, pred_y, true_x, true_y
            )

            fixed_safe_loss, fixed_pain_loss = fixed_xy_task_losses(
                model,
                observed,
                z_static_pca,
                true_x,
                true_y,
            )

            fixed_pred = 0 if fixed_safe_loss <= fixed_pain_loss else 1
            fixed_correct = int(fixed_pred == true_label)

            rows.append({
                "trajectory": item["key"],
                "pair_id": int(item["pair_id"]),
                "true_label": true_label,
                "true_task": "SAFE" if true_label == 0 else "PAIN",
                "history_percent": int(pct),
                "n_obs": int(n_obs),
                "T_full": int(T_full),

                "safe_loss": float(safe_loss),
                "pain_loss": float(pain_loss),
                "pred_label": int(pred_label),
                "pred_task": "SAFE" if pred_label == 0 else "PAIN",
                "correct": int(correct),

                "signed_true_margin": float(signed_margin),
                "winner_margin": float(winner_margin),

                "pred_x": float(pred_x),
                "pred_y": float(pred_y),
                "true_x": float(true_x),
                "true_y": float(true_y),
                "joint_xy_error_mm": float(joint_xy_error),

                "fixed_xy_safe_loss": float(fixed_safe_loss),
                "fixed_xy_pain_loss": float(fixed_pain_loss),
                "fixed_xy_pred_label": int(fixed_pred),
                "fixed_xy_correct": int(fixed_correct),
            })

        print("DONE:", item["key"])

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "per_trajectory_context_diagnostics.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------
    # Summary by history
    # ------------------------------------------------------------
    summary = (
        df.groupby("history_percent")
        .agg(
            task_accuracy=("correct", "mean"),
            fixed_xy_task_accuracy=("fixed_xy_correct", "mean"),
            mean_signed_margin=("signed_true_margin", "mean"),
            median_signed_margin=("signed_true_margin", "median"),
            mean_winner_margin=("winner_margin", "mean"),
            mean_xy_error_mm=("joint_xy_error_mm", "mean"),
            std_xy_error_mm=("joint_xy_error_mm", "std"),
            median_xy_error_mm=("joint_xy_error_mm", "median"),
        )
        .reset_index()
    )

    summary["task_accuracy_pct"] = 100.0 * summary["task_accuracy"]
    summary["fixed_xy_task_accuracy_pct"] = 100.0 * summary["fixed_xy_task_accuracy"]

    summary_path = OUT_DIR / "history_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------
    # Identify flips
    # ------------------------------------------------------------
    pivot_correct = df.pivot(
        index="trajectory",
        columns="history_percent",
        values="correct",
    )[HISTORIES]

    flip_rows = []
    for traj, vals in pivot_correct.iterrows():
        arr = vals.to_numpy(dtype=int)
        flips = np.where(arr[1:] != arr[:-1])[0]

        if len(flips):
            transitions = []
            for idx in flips:
                transitions.append(
                    f"{HISTORIES[idx]}->{HISTORIES[idx+1]}:"
                    f"{arr[idx]}->{arr[idx+1]}"
                )

            flip_rows.append({
                "trajectory": traj,
                "num_flips": int(len(flips)),
                "transitions": "; ".join(transitions),
            })

    flip_df = pd.DataFrame(flip_rows)
    flip_path = OUT_DIR / "classification_flip_cases.csv"
    flip_df.to_csv(flip_path, index=False)

    # ------------------------------------------------------------
    # Correctness heatmap
    # ------------------------------------------------------------
    order = sorted(pivot_correct.index)
    mat = pivot_correct.loc[order, HISTORIES].to_numpy(dtype=float)

    plt.figure(figsize=(12, max(7, 0.28 * len(order))))
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.xticks(range(len(HISTORIES)), HISTORIES)
    plt.yticks(range(len(order)), order, fontsize=7)
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Test trajectory")
    plt.title("Per-Trajectory TASK Correctness (1=correct, 0=incorrect)")
    plt.colorbar(label="Correct")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "01_task_correctness_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------
    # Signed margin heatmap
    # ------------------------------------------------------------
    pivot_margin = df.pivot(
        index="trajectory",
        columns="history_percent",
        values="signed_true_margin",
    )[HISTORIES]

    margin_mat = pivot_margin.loc[order, HISTORIES].to_numpy(dtype=float)
    vmax = max(abs(np.nanmin(margin_mat)), abs(np.nanmax(margin_mat)), 1e-6)

    plt.figure(figsize=(12, max(7, 0.28 * len(order))))
    plt.imshow(
        margin_mat,
        aspect="auto",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.xticks(range(len(HISTORIES)), HISTORIES)
    plt.yticks(range(len(order)), order, fontsize=7)
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Test trajectory")
    plt.title("Signed TASK Margin: positive = true hypothesis preferred")
    plt.colorbar(label="Normalized signed margin")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "02_signed_context_margin_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------
    # Aggregate accuracy + margin
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.6, 5.4))
    ax1.plot(
        summary["history_percent"],
        summary["task_accuracy_pct"],
        marker="o",
        label="TASK accuracy",
    )
    ax1.set_xlabel("Observed trajectory (%)")
    ax1.set_ylabel("TASK accuracy (%)")
    ax1.set_ylim(45, 102)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        summary["history_percent"],
        summary["mean_signed_margin"],
        marker="s",
        label="Mean signed TASK margin",
    )
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_ylabel("Mean signed context margin")

    lines = ax1.lines[:1] + ax2.lines[:1]
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")

    plt.title("TASK Accuracy and Context Separation vs Observation")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "03_accuracy_and_margin_vs_history.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ------------------------------------------------------------
    # Per-flip trajectory SAFE/PAIN loss curves
    # ------------------------------------------------------------
    for traj in flip_df["trajectory"].tolist() if len(flip_df) else []:
        g = df[df["trajectory"] == traj].sort_values("history_percent")

        plt.figure(figsize=(8.2, 5.2))
        plt.plot(
            g["history_percent"],
            g["safe_loss"],
            marker="o",
            label="SAFE optimized loss",
        )
        plt.plot(
            g["history_percent"],
            g["pain_loss"],
            marker="o",
            label="PAIN optimized loss",
        )

        true_task = g["true_task"].iloc[0]
        pair_id = int(g["pair_id"].iloc[0])

        plt.xlabel("Observed trajectory (%)")
        plt.ylabel("Optimized latent reconstruction loss")
        plt.title(
            f"{traj} | pair {pair_id} | true={true_task}"
        )
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            per_case_dir / f"{traj}_loss_curves.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close()

    # ------------------------------------------------------------
    # Transition counts
    # ------------------------------------------------------------
    transition_rows = []

    for a, b in zip(HISTORIES[:-1], HISTORIES[1:]):
        A = df[df["history_percent"] == a].set_index("trajectory")
        B = df[df["history_percent"] == b].set_index("trajectory")

        common = A.index.intersection(B.index)

        cc = int(((A.loc[common, "correct"] == 1) & (B.loc[common, "correct"] == 1)).sum())
        ci = int(((A.loc[common, "correct"] == 1) & (B.loc[common, "correct"] == 0)).sum())
        ic = int(((A.loc[common, "correct"] == 0) & (B.loc[common, "correct"] == 1)).sum())
        ii = int(((A.loc[common, "correct"] == 0) & (B.loc[common, "correct"] == 0)).sum())

        transition_rows.append({
            "from_percent": a,
            "to_percent": b,
            "correct_to_correct": cc,
            "correct_to_incorrect": ci,
            "incorrect_to_correct": ic,
            "incorrect_to_incorrect": ii,
        })

    trans_df = pd.DataFrame(transition_rows)
    trans_df.to_csv(
        OUT_DIR / "history_transition_counts.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------
    report = []
    report.append("CONTEXT STABILITY DIAGNOSTIC REPORT")
    report.append("=" * 80)
    report.append(f"Test trajectories: {len(sequences)}")
    report.append(f"Histories: {HISTORIES}")
    report.append("")

    report.append("Accuracy by history:")
    for _, r in summary.iterrows():
        report.append(
            f"{int(r['history_percent']):3d}% | "
            f"TASK={r['task_accuracy_pct']:6.2f}% | "
            f"mean signed margin={r['mean_signed_margin']:+.6f} | "
            f"XY={r['mean_xy_error_mm']:.2f} ± {r['std_xy_error_mm']:.2f} mm"
        )

    report.append("")
    report.append(f"Trajectories that flip TASK at least once: {len(flip_df)}")

    if len(flip_df):
        for _, r in flip_df.sort_values(
            ["num_flips", "trajectory"],
            ascending=[False, True],
        ).iterrows():
            report.append(
                f"{r['trajectory']}: {int(r['num_flips'])} flips | {r['transitions']}"
            )

    report.append("")
    report.append("Interpretation rule:")
    report.append("  signed_true_margin > 0 : true TASK has lower optimized loss")
    report.append("  signed_true_margin < 0 : wrong TASK has lower optimized loss")
    report.append("  values close to 0      : ambiguous / near decision boundary")

    report_path = OUT_DIR / "diagnostic_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print()
    print("=" * 118)
    print("DONE")
    print("=" * 118)
    print("Detailed CSV :", csv_path.resolve())
    print("Summary CSV  :", summary_path.resolve())
    print("Flip cases   :", flip_path.resolve())
    print("Report       :", report_path.resolve())
    print("Figures      :", OUT_DIR.resolve())
    print("=" * 118)


if __name__ == "__main__":
    main()
