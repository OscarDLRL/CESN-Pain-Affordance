from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from CESN.core.cesn import ContextualESNTorch
from CESN.core.train_grasp_context_v6_native256_latent200_final70 import (
    LATENT_DATA_PATH,
    STATIC_PCA_DIM,
    RESERVOIR_DIM,
    SPECTRAL_RADIUS,
    LEAK_RATE,
    INPUT_SCALE,
    BIAS_SCALE,
    SEED,
    CONTEXT_DIM,
    OUTPUT_DIM,
    TIME_INDEX_SCALE,
    build_pair_splits,
    load_raw_sequences,
    fit_static_pca,
    prepare_sequences,
    evaluate,
    fit_model,
    print_metrics,
)

RIDGES = [0.1, 0.5, 1.0, 2.0]

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT / "CESN" / "outputs"
    / "rgb_grasp_context_cesn_v6_native256_latent200_ridge_sweep_final70"
)

CSV_PATH = OUT_DIR / "ridge_sweep_summary.csv"
JSON_PATH = OUT_DIR / "ridge_sweep_summary.json"
BEST_MODEL_PATH = OUT_DIR / "best_model_by_test_r2.pt"


def make_model(device, ridge):
    return ContextualESNTorch(
        input_dim=CONTEXT_DIM,
        output_dim=OUTPUT_DIM,
        reservoir_size=RESERVOIR_DIM,
        spectral_radius=SPECTRAL_RADIUS,
        leak_rate=LEAK_RATE,
        input_scale=INPUT_SCALE,
        bias_scale=BIAS_SCALE,
        ridge=float(ridge),
        seed=SEED,
        device=device,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA was not detected.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = build_pair_splits()
    train_raw = load_raw_sequences(splits["train"])
    test_raw = load_raw_sequences(splits["test"])

    pca = fit_static_pca(train_raw)
    train_seq = prepare_sequences(train_raw, pca)
    test_seq = prepare_sequences(test_raw, pca)

    print()
    print("=" * 110)
    print("NATIVE256 LATENT200 RIDGE SWEEP | reservoir1500 | FINAL70")
    print("=" * 110)
    print("z_static           : 128")
    print("z_dynamic          : 200")
    print("Reservoir          :", RESERVOIR_DIM)
    print("Ridges             :", RIDGES)
    print("PCA16 explained    :", f"{pca['explained_ratio']*100:.4f}%")
    print("Selection          : highest TEST R2")
    print("=" * 110)

    results = []
    best = None
    best_state = None

    for ridge in RIDGES:
        print()
        print("-" * 110)
        print(f"RIDGE = {ridge:g}")
        print("-" * 110)

        model = make_model(device, ridge)
        fit_model(model, train_seq)

        train_metrics = evaluate(model, train_seq)
        test_metrics = evaluate(model, test_seq)

        print_metrics("TRAIN70", train_metrics)
        print_metrics("TEST11", test_metrics)

        row = {
            "ridge": float(ridge),
            "train_rmse": float(train_metrics["rmse"]),
            "train_mae": float(train_metrics["mae"]),
            "train_r2": float(train_metrics["r2"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
        }
        results.append(row)

        if best is None or row["test_r2"] > best["test_r2"]:
            best = dict(row)
            best_state = {
                "model_state": model.state_dict(),
                "pca_mean": torch.from_numpy(pca["mean"]),
                "pca_components": torch.from_numpy(pca["components"]),
                "pca_explained_ratio": float(pca["explained_ratio"]),
                "static_pca_dim": STATIC_PCA_DIM,
                "architecture":
                    "rgb_grasp_context_cesn_v6_native256_latent200_ridge_sweep_final70",
                "latent_dataset": str(LATENT_DATA_PATH),
                "train_pairs": sorted(list(splits["train"])),
                "val_pairs": [],
                "test_pairs": sorted(list(splits["test"])),
                "reservoir_size": int(RESERVOIR_DIM),
                "ridge": float(ridge),
                "time_index_scale": float(TIME_INDEX_SCALE),
                "output_dim": int(OUTPUT_DIM),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }

        del model
        torch.cuda.empty_cache()

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    JSON_PATH.write_text(
        json.dumps(
            {
                "z_dynamic": 200,
                "reservoir": int(RESERVOIR_DIM),
                "ridges": RIDGES,
                "best": best,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    torch.save(best_state, BEST_MODEL_PATH)

    print()
    print("=" * 110)
    print("RIDGE SWEEP COMPLETE")
    print("=" * 110)
    for row in sorted(results, key=lambda x: x["test_r2"], reverse=True):
        mark = " <-- BEST" if row["ridge"] == best["ridge"] else ""
        print(
            f"ridge={row['ridge']:<4g} "
            f"trainR2={row['train_r2']:.4f} "
            f"testR2={row['test_r2']:.4f} "
            f"testRMSE={row['test_rmse']:.6f}"
            f"{mark}"
        )

    print("Best model:", BEST_MODEL_PATH.resolve())


if __name__ == "__main__":
    main()
