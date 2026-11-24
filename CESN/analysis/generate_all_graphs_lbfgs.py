#!/usr/bin/env python3
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path.home() / "Projects" / "CESN_Context_Inversion_Pain_Affordance"

METHOD_DIR = (
    ROOT
    / "CESN"
    / "outputs"
    / "rgb_grasp_context_cesn_v6_noanchor_time"
    / "method_a_lbfgs"
)

SUMMARY_CSV = METHOD_DIR / "method_a_lbfgs_summary.csv"

FIG_DIR = METHOD_DIR / "paper_graphs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON = FIG_DIR / "paper_metrics_lbfgs.json"
TABLE_CSV = FIG_DIR / "paper_summary_lbfgs.csv"


def load_rows():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing L-BFGS summary:\n{SUMMARY_CSV}\n\n"
            "Run method_a_lbfgs_v6_noanchor_time_gpu.py first."
        )

    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def conv(v):
        try:
            return float(v)
        except Exception:
            return v

    rows = [
        {k: conv(v) for k, v in r.items()}
        for r in rows
    ]

    rows.sort(
        key=lambda r: float(r["history_percent"])
    )

    return rows


def arr(rows, key):
    return np.asarray(
        [float(r[key]) for r in rows],
        dtype=np.float64,
    )


def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def main():
    rows = load_rows()

    history = arr(rows, "history_percent")

    # ========================================================
    # FIGURE 1 — Main Method A task accuracy
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        100.0 * arr(rows, "joint_task_accuracy"),
        marker="o",
        label="Method A: TASK + XY unknown",
    )
    plt.plot(
        history,
        100.0 * arr(rows, "true_xy_task_accuracy"),
        marker="o",
        label="Diagnostic: TASK unknown, true XY",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Task accuracy (%)")
    plt.ylim(0, 105)
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("L-BFGS Method A: Task inference")
    savefig("01_task_accuracy_lbfgs.png")

    # ========================================================
    # FIGURE 2 — SAFE vs PAIN accuracy under Method A
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        100.0 * arr(rows, "joint_safe_accuracy"),
        marker="o",
        label="SAFE",
    )
    plt.plot(
        history,
        100.0 * arr(rows, "joint_pain_accuracy"),
        marker="o",
        label="PAIN",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Task accuracy (%)")
    plt.ylim(0, 105)
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("L-BFGS Method A: SAFE vs PAIN accuracy")
    savefig("02_safe_vs_pain_accuracy_lbfgs.png")

    # ========================================================
    # FIGURE 3 — Median XY error
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        arr(rows, "known_task_xy_median_mm"),
        marker="o",
        label="Known TASK → unknown XY",
    )
    plt.plot(
        history,
        arr(rows, "joint_xy_median_mm"),
        marker="o",
        label="Method A: TASK + XY unknown",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Median XY error (mm)")
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("L-BFGS spatial inversion: Median XY error")
    savefig("03_xy_median_error_lbfgs.png")

    # ========================================================
    # FIGURE 4 — Mean / median / P90 Method A XY error
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        arr(rows, "joint_xy_mean_mm"),
        marker="o",
        label="Mean",
    )
    plt.plot(
        history,
        arr(rows, "joint_xy_median_mm"),
        marker="o",
        label="Median",
    )
    plt.plot(
        history,
        arr(rows, "joint_xy_p90_mm"),
        marker="o",
        label="P90",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("XY error (mm)")
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("L-BFGS Method A: XY error distribution")
    savefig("04_xy_mean_median_p90_lbfgs.png")

    # ========================================================
    # FIGURE 5 — Spatial success thresholds
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        100.0 * arr(rows, "joint_xy_within_10"),
        marker="o",
        label="≤ 10 mm",
    )
    plt.plot(
        history,
        100.0 * arr(rows, "joint_xy_within_20"),
        marker="o",
        label="≤ 20 mm",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Trajectories within threshold (%)")
    plt.ylim(0, 105)
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("L-BFGS Method A: Spatial success")
    savefig("05_xy_success_thresholds_lbfgs.png")

    # ========================================================
    # FIGURE 6 — Known-task threshold diagnostic
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        100.0 * arr(rows, "known_task_xy_within_10"),
        marker="o",
        label="≤ 10 mm",
    )
    plt.plot(
        history,
        100.0 * arr(rows, "known_task_xy_within_20"),
        marker="o",
        label="≤ 20 mm",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Trajectories within threshold (%)")
    plt.ylim(0, 105)
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Known TASK diagnostic: Spatial success")
    savefig("06_known_task_xy_success_lbfgs.png")

    # ========================================================
    # FIGURE 7 — True-XY SAFE vs PAIN diagnostic
    # ========================================================
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history,
        100.0 * arr(rows, "true_xy_safe_accuracy"),
        marker="o",
        label="SAFE",
    )
    plt.plot(
        history,
        100.0 * arr(rows, "true_xy_pain_accuracy"),
        marker="o",
        label="PAIN",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Task accuracy (%)")
    plt.ylim(0, 105)
    plt.xticks(history)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("True-XY diagnostic: SAFE vs PAIN separability")
    savefig("07_true_xy_safe_vs_pain_lbfgs.png")

    # ========================================================
    # FIGURE 8 — Compact paper figure: accuracy + XY
    # One chart only (no subplot) to keep each artifact clean.
    # Normalized scales intentionally NOT mixed.
    # Instead plot task accuracy only with key late-stage region.
    # ========================================================
    late = history >= 30
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(
        history[late],
        100.0 * arr(rows, "joint_task_accuracy")[late],
        marker="o",
        label="Method A task accuracy",
    )
    plt.plot(
        history[late],
        100.0 * arr(rows, "true_xy_task_accuracy")[late],
        marker="o",
        label="True-XY task diagnostic",
    )
    plt.xlabel("Observed trajectory (%)")
    plt.ylabel("Task accuracy (%)")
    plt.ylim(70, 102)
    plt.xticks(history[late])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Late-stage task inference (30–100%)")
    savefig("08_late_stage_task_accuracy_lbfgs.png")

    # ========================================================
    # Export clean table and headline metrics
    # ========================================================
    fields = [
        "history_percent",
        "joint_task_accuracy",
        "joint_safe_accuracy",
        "joint_pain_accuracy",
        "joint_xy_mean_mm",
        "joint_xy_median_mm",
        "joint_xy_p90_mm",
        "joint_xy_within_10",
        "joint_xy_within_20",
        "known_task_xy_median_mm",
        "known_task_xy_p90_mm",
        "true_xy_task_accuracy",
        "true_xy_safe_accuracy",
        "true_xy_pain_accuracy",
    ]

    with TABLE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    final = rows[-1]

    metrics = {
        "method": "L-BFGS Method A",
        "architecture": "V6 no-anchor + causal time index",
        "history_points_percent": [int(v) for v in history],
        "final_100pct": {
            "task_accuracy_percent":
                100.0 * float(final["joint_task_accuracy"]),
            "safe_accuracy_percent":
                100.0 * float(final["joint_safe_accuracy"]),
            "pain_accuracy_percent":
                100.0 * float(final["joint_pain_accuracy"]),
            "joint_xy_mean_mm":
                float(final["joint_xy_mean_mm"]),
            "joint_xy_median_mm":
                float(final["joint_xy_median_mm"]),
            "joint_xy_p90_mm":
                float(final["joint_xy_p90_mm"]),
            "joint_xy_within_10_percent":
                100.0 * float(final["joint_xy_within_10"]),
            "joint_xy_within_20_percent":
                100.0 * float(final["joint_xy_within_20"]),
            "known_task_xy_median_mm":
                float(final["known_task_xy_median_mm"]),
            "true_xy_task_accuracy_percent":
                100.0 * float(final["true_xy_task_accuracy"]),
        },
    }

    METRICS_JSON.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("L-BFGS PAPER GRAPH GENERATION COMPLETE")
    print("=" * 90)
    print(f"Input : {SUMMARY_CSV}")
    print(f"Output: {FIG_DIR}")
    print()
    print("100% observation:")
    print(
        f"  Method A TASK accuracy = "
        f"{metrics['final_100pct']['task_accuracy_percent']:.2f}%"
    )
    print(
        f"  Method A XY median     = "
        f"{metrics['final_100pct']['joint_xy_median_mm']:.2f} mm"
    )
    print(
        f"  Known-TASK XY median   = "
        f"{metrics['final_100pct']['known_task_xy_median_mm']:.2f} mm"
    )
    print(
        f"  True-XY TASK accuracy  = "
        f"{metrics['final_100pct']['true_xy_task_accuracy_percent']:.2f}%"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
