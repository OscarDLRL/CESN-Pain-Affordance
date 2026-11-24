from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DETAIL_CSV = (
    ROOT / "CESN" / "outputs"
    / "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70"
    / "context_stability_diagnostics"
    / "per_trajectory_context_diagnostics.csv"
)

RGB_H5 = (
    ROOT / "datasets"
    / "pain_affordance_162"
    / "pain_affordance_162.h5"
)

OUT_DIR = (
    ROOT / "CESN" / "outputs"
    / "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70"
    / "geometric_task_from_xy"
)

HISTORIES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]


def infer_columns(df):
    candidates = {
        "trajectory": ["trajectory", "key", "trajectory_key"],
        "history": ["history_percent", "history", "percent", "observation_percent"],
        "true_label": ["true_label", "pain_label", "task_label"],
        "pred_label": ["pred_label", "joint_pred_label", "joint_task_pred"],
        "pred_x": ["pred_x", "joint_pred_x", "joint_x", "predicted_x"],
        "pred_y": ["pred_y", "joint_pred_y", "joint_y", "predicted_y"],
    }

    out = {}
    for logical, names in candidates.items():
        for name in names:
            if name in df.columns:
                out[logical] = name
                break

    missing = [k for k in candidates if k not in out]
    if missing:
        raise RuntimeError(
            f"Could not infer required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=DETAIL_CSV,
        help="Per-trajectory Method A CSV containing predicted XY."
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Method A per-trajectory CSV not found: {args.csv}\n"
            "If your detailed CSV has another name, pass it with --csv."
        )

    df = pd.read_csv(args.csv)
    cols = infer_columns(df)

    object_xy = {}

    with h5py.File(RGB_H5, "r") as h5:
        for key in h5.keys():
            if not str(key).startswith("trajectory_"):
                continue

            g = h5[key]
            xyz = np.asarray(g["object_xyz"][0], dtype=np.float64)
            object_xy[str(key)] = (float(xyz[0]), float(xyz[1]))

    rows = []

    for _, r in df.iterrows():
        key = str(r[cols["trajectory"]])
        if key not in object_xy:
            raise KeyError(f"{key} not found in RGB dataset")

        knife_x, knife_y = object_xy[key]

        pred_x = float(r[cols["pred_x"]])
        pred_y = float(r[cols["pred_y"]])
        true_label = int(r[cols["true_label"]])
        loss_pred_label = int(r[cols["pred_label"]])
        pct = int(r[cols["history"]])

        rel_x = pred_x - knife_x
        rel_y = pred_y - knife_y

        # Dataset geometry:
        # SAFE handle is +X from knife center
        # PAIN blade is -X from knife center
        geometric_label = 0 if rel_x >= 0.0 else 1

        rows.append({
            "trajectory": key,
            "history_percent": pct,
            "true_label": true_label,
            "loss_pred_label": loss_pred_label,
            "geometric_pred_label": geometric_label,
            "loss_correct": int(loss_pred_label == true_label),
            "geometric_correct": int(geometric_label == true_label),
            "pred_x": pred_x,
            "pred_y": pred_y,
            "knife_x": knife_x,
            "knife_y": knife_y,
            "relative_grasp_x": rel_x,
            "relative_grasp_y": rel_y,
        })

    out_df = pd.DataFrame(rows)

    detail_path = OUT_DIR / "geometric_task_per_trajectory.csv"
    out_df.to_csv(detail_path, index=False)

    summary_rows = []

    for pct in sorted(out_df["history_percent"].unique()):
        g = out_df[out_df["history_percent"] == pct]

        loss_acc = 100.0 * g["loss_correct"].mean()
        geom_acc = 100.0 * g["geometric_correct"].mean()

        disagree = int(
            (g["loss_pred_label"] != g["geometric_pred_label"]).sum()
        )

        geom_fixes = int(
            (
                (g["loss_correct"] == 0)
                & (g["geometric_correct"] == 1)
            ).sum()
        )

        geom_breaks = int(
            (
                (g["loss_correct"] == 1)
                & (g["geometric_correct"] == 0)
            ).sum()
        )

        summary_rows.append({
            "history_percent": int(pct),
            "loss_task_accuracy_pct": float(loss_acc),
            "xy_geometric_task_accuracy_pct": float(geom_acc),
            "difference_pct_points": float(geom_acc - loss_acc),
            "loss_vs_geometric_disagreements": disagree,
            "geometric_fixes": geom_fixes,
            "geometric_breaks": geom_breaks,
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "geometric_task_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 108)
    print("TASK FROM INFERRED XY GEOMETRY | native256 latent200 ridge0.5")
    print("=" * 108)
    print("Rule: relative grasp X >= 0 -> SAFE, relative grasp X < 0 -> PAIN")
    print("=" * 108)

    for _, r in summary.iterrows():
        print(
            f"{int(r['history_percent']):3d}%"
            f" | latent-loss TASK={r['loss_task_accuracy_pct']:6.2f}%"
            f" | XY-derived TASK={r['xy_geometric_task_accuracy_pct']:6.2f}%"
            f" | delta={r['difference_pct_points']:+6.2f} pp"
            f" | fixes={int(r['geometric_fixes'])}"
            f" | breaks={int(r['geometric_breaks'])}"
        )

    print()
    print("Saved:", detail_path.resolve())
    print("Saved:", summary_path.resolve())
    print("=" * 108)


if __name__ == "__main__":
    main()
