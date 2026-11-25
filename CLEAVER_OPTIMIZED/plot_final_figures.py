"""Generate clean, paper-oriented figures from final CESN comparison CSVs."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

LABELS = {
    "E20_Raw128_Absolute": "Absolute (Raw128)",
    "E22_Raw128_Increment": "Incremental (Raw128)",
    "E24_Bounded_Increment": "Bounded incremental",
    "E25_InitialState_Memory": "Initial-state memory",
    "E26_Hybrid": "Hybrid",
}
ORDER = list(LABELS)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    if v in (None, "", "None", "nan", "NaN"):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def metric_plot(ax, rows, metric, subset, ylabel, title=None, ylim=None):
    for exp in ORDER:
        rr = [r for r in rows if r["experiment"] == exp and r["subset"] == subset]
        rr = sorted(rr, key=lambda r: int(r["frames"]))
        rr = [r for r in rr if np.isfinite(num(r.get(metric)))]
        if rr:
            ax.plot([int(r["frames"]) for r in rr], [num(r[metric]) for r in rr], marker="o", ms=4, lw=1.8, label=LABELS[exp])
    ax.set_xlabel("Observed frames")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def make_figures(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(out / "comparison.csv")
    paper = out / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)

    for split in sorted({r["split"] for r in rows}):
        rr = [r for r in rows if r["split"] == split]
        # 1. Accuracy, all trajectories and pre-contact subset.
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
        metric_plot(axes[0], rr, "accuracy_pct", "all", "Intent accuracy (%)", "All observations", (-3, 103))
        metric_plot(axes[1], rr, "accuracy_pct", "precontact", "Intent accuracy (%)", "Before first blade contact", (-3, 103))
        axes[0].axhline(50, ls="--", lw=0.9, color="gray")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Intent classification — {split}")
        fig.tight_layout()
        fig.savefig(paper / f"01_intent_accuracy_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # 2. PAIN recall: the key safety-sensitive class metric.
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
        metric_plot(axes[0], rr, "pain_recall_pct", "all", "PAIN recall (%)", "All PAIN trajectories", (-3, 103))
        metric_plot(axes[1], rr, "pain_recall_pct", "precontact", "PAIN recall (%)", "PAIN before first contact", (-3, 103))
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Hazard anticipation — {split}")
        fig.tight_layout()
        fig.savefig(paper / f"02_pain_recall_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # 3. Spatial context recovery.
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
        metric_plot(axes[0], rr, "xy_mean_mm", "all", "Mean XY error (mm)", "All trajectories")
        metric_plot(axes[1], rr, "xy_mean_mm", "precontact", "Mean XY error (mm)", "Before first contact")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Spatial goal recovery — {split}")
        fig.tight_layout()
        fig.savefig(paper / f"03_spatial_error_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # 4. Autonomous future latent prediction quality.
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        metric_plot(ax, rr, "future_rmse", "all", "Future latent RMSE", "Autonomous future prediction")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(paper / f"04_future_latent_rmse_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # 5. Runtime cost.
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        metric_plot(ax, rr, "amortized_inference_ms", "all", "Inference time (ms / trajectory)", "Context inversion runtime")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(paper / f"05_inference_time_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # 6. Early spatial estimates (5 and 10 observed frames).
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True)
        markers = ["o", "s", "^", "D", "P"]
        for ax, frame_count in zip(axes, (5, 10)):
            all_true = None
            for exp, marker in zip(ORDER, markers):
                per_path = out / exp / split / "per_trajectory.csv"
                if not per_path.exists():
                    continue
                traj = read_csv(per_path)
                traj = [r for r in traj if int(r["frames"]) == frame_count]
                if not traj:
                    continue
                if all_true is None:
                    all_true = traj
                    ax.scatter([1000*num(r["true_x_m"]) for r in traj], [1000*num(r["true_y_m"]) for r in traj], c="black", marker="x", s=35, label="True goal")
                ax.scatter([1000*num(r["pred_x_m"]) for r in traj], [1000*num(r["pred_y_m"]) for r in traj], marker=marker, s=28, alpha=0.85, label=LABELS[exp])
            ax.set_title(f"{frame_count} observed frames")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.grid(True, alpha=0.25)
            ax.set_aspect("equal", adjustable="box")
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"Early spatial context estimates — {split}")
        fig.tight_layout()
        fig.savefig(paper / f"06_early_xy_estimates_{split}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print("Paper figures saved to:", paper.resolve())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True, help="Run output folder containing comparison.csv")
    args = p.parse_args()
    make_figures(args.output)


if __name__ == "__main__":
    main()
