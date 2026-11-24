from __future__ import annotations

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSV = (
    ROOT / "CESN" / "outputs"
    / "rgb_grasp_context_cesn_v6_native256_ridge05_final70"
    / "method_a_lbfgs_native256_ridge05_final70"
    / "method_a_lbfgs_native256_ridge05_final70_summary.csv"
)

DEFAULT_OUT = (
    ROOT / "CESN" / "outputs" / "paper_figures"
    / "final_context_inversion"
)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def pct(x):
    return 100.0 * np.asarray(x, dtype=float)

def first_consistent(history, values, predicate):
    history = np.asarray(history)
    values = list(values)
    for i in range(len(history)):
        if all(predicate(v) for v in values[i:]):
            return int(history[i])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--task-threshold", type=float, default=0.85)
    ap.add_argument("--xy-threshold-mm", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("history_percent").reset_index(drop=True)

    required = [
        "known_task_xy_mean_mm",
        "known_task_xy_std_mm",
        "joint_xy_mean_mm",
        "joint_xy_std_mm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "CSV is missing mean±SD fields. Re-run the updated Method A first. "
            f"Missing: {missing}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    h = df["history_percent"].to_numpy()

    # 1 TASK accuracy
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(h, pct(df["joint_task_accuracy"]), marker="o", label="Inverse context: TASK+XY unknown")
    plt.plot(h, pct(df["true_xy_task_accuracy"]), marker="o", label="TASK only (true XY diagnostic)")
    plt.axhline(100*args.task_threshold, linestyle="--", linewidth=1, label=f"{100*args.task_threshold:.0f}% reference")
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Task accuracy (%)")
    plt.ylim(45, 102)
    plt.title("Pain Affordance Context Recovery vs Observed Trajectory")
    plt.grid(alpha=0.25); plt.legend()
    savefig(args.out_dir / "01_task_accuracy_vs_observation.png")

    # 2 SAFE/PAIN
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(h, pct(df["joint_safe_accuracy"]), marker="o", label="SAFE accuracy")
    plt.plot(h, pct(df["joint_pain_accuracy"]), marker="o", label="PAIN accuracy")
    plt.plot(h, pct(df["joint_task_accuracy"]), marker="o", linestyle="--", label="Overall TASK accuracy")
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.title("SAFE vs PAIN Inverse-Context Classification")
    plt.grid(alpha=0.25); plt.legend()
    savefig(args.out_dir / "02_safe_pain_accuracy_vs_observation.png")

    # 3 MAIN XY: mean ± SD
    plt.figure(figsize=(8.4, 5.4))

    known_mean = df["known_task_xy_mean_mm"].to_numpy()
    known_std = df["known_task_xy_std_mm"].to_numpy()
    joint_mean = df["joint_xy_mean_mm"].to_numpy()
    joint_std = df["joint_xy_std_mm"].to_numpy()

    plt.plot(h, known_mean, marker="o", label="Known TASK: mean XY error")
    plt.fill_between(
        h,
        np.maximum(0.0, known_mean - known_std),
        known_mean + known_std,
        alpha=0.18,
        label="Known TASK: ±1 SD",
    )

    plt.plot(h, joint_mean, marker="o", label="Unknown TASK: mean XY error")
    plt.fill_between(
        h,
        np.maximum(0.0, joint_mean - joint_std),
        joint_mean + joint_std,
        alpha=0.18,
        label="Unknown TASK: ±1 SD",
    )

    plt.axhline(args.xy_threshold_mm, linestyle="--", linewidth=1, label=f"{args.xy_threshold_mm:.0f} mm reference")
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("XY error (mm)")
    plt.title("Grasp-Context Inversion Precision: Mean ± SD")
    plt.grid(alpha=0.25); plt.legend()
    savefig(args.out_dir / "03_xy_mean_sd_vs_observation.png")

    # 4 Robustness: mean + P90
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(h, df["joint_xy_mean_mm"], marker="o", label="Mean")
    plt.plot(h, df["joint_xy_p90_mm"], marker="o", label="90th percentile")
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Joint XY error (mm)")
    plt.title("Joint TASK+XY Inversion Robustness")
    plt.grid(alpha=0.25); plt.legend()
    savefig(args.out_dir / "04_joint_xy_mean_p90_vs_observation.png")

    # 5 success thresholds
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(h, pct(df["joint_xy_within_10"]), marker="o", label="XY error ≤ 10 mm")
    plt.plot(h, pct(df["joint_xy_within_20"]), marker="o", label="XY error ≤ 20 mm")
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Trajectories satisfying threshold (%)")
    plt.ylim(0, 105)
    plt.title("Spatial Reliability of Inverse Context Recovery")
    plt.grid(alpha=0.25); plt.legend()
    savefig(args.out_dir / "05_joint_xy_success_thresholds.png")

    # 6 combined
    task = pct(df["joint_task_accuracy"])
    xy = df["joint_xy_mean_mm"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
    l1 = ax1.plot(h, task, marker="o", label="TASK accuracy")
    ax1.axhline(100*args.task_threshold, linestyle="--", linewidth=1)
    ax1.set_xlabel("Observed trajectory (%)")
    ax1.set_ylabel("TASK accuracy (%)")
    ax1.set_ylim(45, 102)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    l2 = ax2.plot(h, xy, marker="s", label="Joint XY mean error")
    ax2.axhline(args.xy_threshold_mm, linestyle=":", linewidth=1)
    ax2.set_ylabel("Joint XY mean error (mm)")

    ax1.legend(l1+l2, [x.get_label() for x in l1+l2], loc="best")
    plt.title("When Does Inverse Context Become Consistent?")
    savefig(args.out_dir / "06_context_consistency_combined.png")

    # Stability summaries use mean error now.
    task80 = first_consistent(h, df["joint_task_accuracy"], lambda v: v >= 0.80)
    task85 = first_consistent(h, df["joint_task_accuracy"], lambda v: v >= 0.85)
    task90 = first_consistent(h, df["joint_task_accuracy"], lambda v: v >= 0.90)
    known10 = first_consistent(h, df["known_task_xy_mean_mm"], lambda v: v <= 10.0)
    joint10 = first_consistent(h, df["joint_xy_mean_mm"], lambda v: v <= 10.0)
    joint20 = first_consistent(h, df["joint_xy_mean_mm"], lambda v: v <= 20.0)
    combined = first_consistent(
        h,
        zip(df["joint_task_accuracy"], df["joint_xy_mean_mm"]),
        lambda p: p[0] >= args.task_threshold and p[1] <= args.xy_threshold_mm,
    )

    summary = {
        "consistency_definition": "Criterion is satisfied at this observation percentage and every later tested percentage.",
        "xy_metric_for_consistency": "mean XY error",
        "minimum_consistent_observation_percent": {
            "task_accuracy_ge_80pct": task80,
            "task_accuracy_ge_85pct": task85,
            "task_accuracy_ge_90pct": task90,
            "known_task_xy_mean_le_10mm": known10,
            "joint_xy_mean_le_10mm": joint10,
            "joint_xy_mean_le_20mm": joint20,
            "combined_task85_and_joint_xy_mean10": combined,
        },
    }

    (args.out_dir / "context_consistency_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "PAIN AFFORDANCE — INVERSE CONTEXT CONSISTENCY",
        "================================================",
        "XY consistency metric: MEAN error",
        f"TASK >= 80% consistently: {task80}%",
        f"TASK >= 85% consistently: {task85}%",
        f"TASK >= 90% consistently: {task90}",
        f"Known-TASK mean XY <= 10 mm consistently: {known10}%",
        f"Joint TASK+XY mean XY <= 10 mm consistently: {joint10}%",
        f"Joint TASK+XY mean XY <= 20 mm consistently: {joint20}%",
        f"Combined TASK >= {100*args.task_threshold:.0f}% and joint mean XY <= {args.xy_threshold_mm:.0f} mm consistently: {combined}%",
    ]

    (args.out_dir / "context_consistency_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print("\nSaved figures to:", args.out_dir.resolve())

if __name__ == "__main__":
    main()
