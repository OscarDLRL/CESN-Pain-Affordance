from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from CESN.core.cesn import ContextualESNTorch

from CESN.core.parametersCESN import (
    LATENT_DATA_PATH,
    OUT_DIR,
    PAIN_DIM,
    OUTPUT_DIM,
    RESERVOIR_SIZE,
    SPECTRAL_RADIUS,
    LEAK_RATE,
    INPUT_SCALE,
    BIAS_SCALE,
    RIDGE,
    RANDOM_SEED,
    TRAIN_PAIRS,
    VAL_PAIRS,
    TEST_PAIRS,
    SPLIT_SEED,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


PCA_DIMS = [2, 4, 8, 16, 32]


# ============================================================
# Split
# ============================================================

def build_pair_splits(path):
    pair_ids = set()

    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            pair_ids.add(
                int(h5[key].attrs["pair_id"])
            )

    pair_ids = np.asarray(
        sorted(pair_ids),
        dtype=np.int64,
    )

    if len(pair_ids) != 81:
        raise ValueError(
            f"Expected 81 pairs, got {len(pair_ids)}"
        )

    rng = np.random.default_rng(
        SPLIT_SEED
    )
    rng.shuffle(pair_ids)

    train = set(
        pair_ids[:TRAIN_PAIRS].tolist()
    )

    val = set(
        pair_ids[
            TRAIN_PAIRS:
            TRAIN_PAIRS + VAL_PAIRS
        ].tolist()
    )

    test = set(
        pair_ids[
            TRAIN_PAIRS + VAL_PAIRS:
        ].tolist()
    )

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


# ============================================================
# Data
# ============================================================

def load_raw_sequences(path, pair_ids):
    sequences = []

    with h5py.File(path, "r") as h5:

        for key in sorted(h5.keys()):
            g = h5[key]

            pair_id = int(
                g.attrs["pair_id"]
            )

            if pair_id not in pair_ids:
                continue

            sequences.append(
                {
                    "key": str(key),
                    "pair_id": pair_id,
                    "pain_label": int(
                        g.attrs["pain_label"]
                    ),
                    "z_static": np.asarray(
                        g["z_static"],
                        dtype=np.float32,
                    ),
                    "z_dynamic": np.asarray(
                        g["z_dynamic"],
                        dtype=np.float32,
                    ),
                    "time": np.asarray(
                        g["time"],
                        dtype=np.float32,
                    ),
                }
            )

    return sequences


# ============================================================
# PCA
# ============================================================

def fit_static_pca(train_sequences, n_components):
    X = np.stack(
        [
            item["z_static"]
            for item in train_sequences
        ],
        axis=0,
    ).astype(np.float64)

    mean = X.mean(
        axis=0,
        keepdims=True,
    )

    Xc = X - mean

    _, S, Vt = np.linalg.svd(
        Xc,
        full_matrices=False,
    )

    components = Vt[
        :n_components
    ].astype(np.float32)

    mean = mean[0].astype(
        np.float32
    )

    explained_variance = (
        S ** 2
    ) / max(
        X.shape[0] - 1,
        1,
    )

    total_var = explained_variance.sum()

    explained_ratio = (
        explained_variance[
            :n_components
        ].sum()
        / max(
            total_var,
            1e-12,
        )
    )

    per_component_ratio = (
        explained_variance
        / max(
            total_var,
            1e-12,
        )
    )

    return {
        "mean": mean,
        "components": components,
        "explained_ratio": float(
            explained_ratio
        ),
        "component_ratios": per_component_ratio.astype(
            np.float32
        ),
    }


def transform_static(z_static, pca):
    centered = (
        np.asarray(
            z_static,
            dtype=np.float32,
        )
        - pca["mean"]
    )

    return (
        centered
        @ pca["components"].T
    ).astype(np.float32)


# ============================================================
# CESN input
# ============================================================

def pain_onehot(pain_label):
    out = np.zeros(
        PAIN_DIM,
        dtype=np.float32,
    )

    out[
        int(pain_label)
    ] = 1.0

    return out


def build_input_sequence(
    pain_label,
    z_static_pca,
    time,
):
    T = int(len(time))

    pain = np.repeat(
        pain_onehot(
            pain_label
        )[None, :],
        T,
        axis=0,
    )

    static = np.repeat(
        np.asarray(
            z_static_pca,
            dtype=np.float32,
        )[None, :],
        T,
        axis=0,
    )

    time_column = np.asarray(
        time,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    return np.concatenate(
        [
            pain,
            static,
            time_column,
        ],
        axis=1,
    ).astype(np.float32)


def prepare_sequences(
    raw_sequences,
    pca,
):
    sequences = []

    for item in raw_sequences:

        z_static_pca = transform_static(
            item["z_static"],
            pca,
        )

        U = build_input_sequence(
            item["pain_label"],
            z_static_pca,
            item["time"],
        )

        sequences.append(
            {
                **item,
                "z_static_pca":
                    z_static_pca,
                "U":
                    U,
                "Y":
                    item["z_dynamic"],
            }
        )

    return sequences


# ============================================================
# Metrics
# ============================================================

def r2_score_global(pred, target):
    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    ss_res = np.sum(
        (
            target - pred
        ) ** 2
    )

    target_mean = np.mean(
        target,
        axis=0,
        keepdims=True,
    )

    ss_tot = np.sum(
        (
            target - target_mean
        ) ** 2
    )

    if ss_tot < 1e-12:
        return float("nan")

    return float(
        1.0
        - ss_res
        / ss_tot
    )


@torch.no_grad()
def evaluate(model, sequences):
    preds = []
    targets = []

    for item in sequences:

        pred = (
            model.predict(
                item["U"]
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        target = item["Y"]

        preds.append(pred)
        targets.append(target)

    pred_all = np.concatenate(
        preds,
        axis=0,
    )

    target_all = np.concatenate(
        targets,
        axis=0,
    )

    error = (
        pred_all
        - target_all
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    return {
        "rmse":
            float(
                np.sqrt(mse)
            ),

        "mae":
            mae,

        "r2":
            r2_score_global(
                pred_all,
                target_all,
            ),
    }


def baseline_mean_predictor(
    train_raw,
    eval_raw,
):
    train_dynamic = np.concatenate(
        [
            item["z_dynamic"]
            for item in train_raw
        ],
        axis=0,
    )

    mean_dynamic = train_dynamic.mean(
        axis=0,
        keepdims=True,
    )

    target = np.concatenate(
        [
            item["z_dynamic"]
            for item in eval_raw
        ],
        axis=0,
    )

    pred = np.repeat(
        mean_dynamic,
        target.shape[0],
        axis=0,
    )

    error = pred - target

    mse = float(
        np.mean(
            error ** 2
        )
    )

    return {
        "rmse":
            float(
                np.sqrt(mse)
            ),

        "mae":
            float(
                np.mean(
                    np.abs(error)
                )
            ),

        "r2":
            r2_score_global(
                pred,
                target,
            ),
    }


# ============================================================
# One sweep run
# ============================================================

def run_one(
    pca_dim,
    train_raw,
    val_raw,
    test_raw,
    device,
):
    pca = fit_static_pca(
        train_raw,
        pca_dim,
    )

    train_sequences = prepare_sequences(
        train_raw,
        pca,
    )

    val_sequences = prepare_sequences(
        val_raw,
        pca,
    )

    test_sequences = prepare_sequences(
        test_raw,
        pca,
    )

    context_dim = (
        PAIN_DIM
        + pca_dim
        + 1
    )

    model = ContextualESNTorch(
        input_dim=
            context_dim,

        output_dim=
            OUTPUT_DIM,

        reservoir_size=
            RESERVOIR_SIZE,

        spectral_radius=
            SPECTRAL_RADIUS,

        leak_rate=
            LEAK_RATE,

        input_scale=
            INPUT_SCALE,

        bias_scale=
            BIAS_SCALE,

        ridge=
            RIDGE,

        seed=
            RANDOM_SEED,

        device=
            device,
    )

    model.fit(
        [
            (
                item["U"],
                item["Y"],
            )
            for item in train_sequences
        ],
        solve_dtype=
            torch.float64,
    )

    train_metrics = evaluate(
        model,
        train_sequences,
    )

    val_metrics = evaluate(
        model,
        val_sequences,
    )

    test_metrics = evaluate(
        model,
        test_sequences,
    )

    return {
        "pca_dim":
            int(
                pca_dim
            ),

        "context_dim":
            int(
                context_dim
            ),

        "explained_variance":
            float(
                pca[
                    "explained_ratio"
                ]
            ),

        "train":
            train_metrics,

        "validation":
            val_metrics,

        "test":
            test_metrics,
    }


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA was not detected."
        )

    props = torch.cuda.get_device_properties(
        device
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_pair_splits(
        LATENT_DATA_PATH
    )

    train_raw = load_raw_sequences(
        LATENT_DATA_PATH,
        splits["train"],
    )

    val_raw = load_raw_sequences(
        LATENT_DATA_PATH,
        splits["validation"],
    )

    test_raw = load_raw_sequences(
        LATENT_DATA_PATH,
        splits["test"],
    )

    baseline_val = baseline_mean_predictor(
        train_raw,
        val_raw,
    )

    baseline_test = baseline_mean_predictor(
        train_raw,
        test_raw,
    )

    print()
    print("=" * 96)
    print(
        "RGB-ONLY CESN V4 - STATIC PCA SWEEP - PYTORCH CUDA"
    )
    print("=" * 96)

    print(
        "GPU =",
        props.name,
    )

    print(
        "PCA DIMS =",
        PCA_DIMS,
    )

    print(
        "RESERVOIR =",
        RESERVOIR_SIZE,
    )

    print(
        "RIDGE =",
        RIDGE,
    )

    print("=" * 96)

    results = []

    for pca_dim in PCA_DIMS:

        torch.cuda.empty_cache()

        print()
        print(
            f">>> PCA DIM = {pca_dim}"
        )

        result = run_one(
            pca_dim,
            train_raw,
            val_raw,
            test_raw,
            device,
        )

        results.append(
            result
        )

        print(
            f"explained="
            f"{result['explained_variance'] * 100:.4f}%"
            f" | train="
            f"{result['train']['rmse']:.6f}"
            f" | val="
            f"{result['validation']['rmse']:.6f}"
            f" | test="
            f"{result['test']['rmse']:.6f}"
            f" | test R2="
            f"{result['test']['r2']:.4f}"
        )

    summary_path = (
        OUT_DIR
        / "pca_dimension_sweep.json"
    )

    summary = {
        "architecture":
            "rgb_only_static_pca_dimension_sweep",

        "reservoir_size":
            RESERVOIR_SIZE,

        "ridge":
            RIDGE,

        "pca_dims":
            PCA_DIMS,

        "baseline_mean":
            {
                "validation":
                    baseline_val,

                "test":
                    baseline_test,
            },

        "results":
            results,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 96)

    print(
        f"{'PCA':>5} "
        f"{'CTX':>5} "
        f"{'EXPL%':>10} "
        f"{'TRAIN':>10} "
        f"{'VAL':>10} "
        f"{'TEST':>10} "
        f"{'TEST R2':>10}"
    )

    for r in results:
        print(
            f"{r['pca_dim']:>5d} "
            f"{r['context_dim']:>5d} "
            f"{r['explained_variance'] * 100:>10.4f} "
            f"{r['train']['rmse']:>10.6f} "
            f"{r['validation']['rmse']:>10.6f} "
            f"{r['test']['rmse']:>10.6f} "
            f"{r['test']['r2']:>10.4f}"
        )

    print()
    print(
        f"VAL baseline  RMSE="
        f"{baseline_val['rmse']:.6f}"
        f" | R2="
        f"{baseline_val['r2']:.4f}"
    )

    print(
        f"TEST baseline RMSE="
        f"{baseline_test['rmse']:.6f}"
        f" | R2="
        f"{baseline_test['r2']:.4f}"
    )

    best_test = min(
        results,
        key=lambda x:
            x["test"]["rmse"],
    )

    best_val = min(
        results,
        key=lambda x:
            x["validation"]["rmse"],
    )

    print()
    print(
        "BEST TEST:",
        f"PCA={best_test['pca_dim']}",
        f"RMSE={best_test['test']['rmse']:.6f}",
        f"R2={best_test['test']['r2']:.4f}",
    )

    print(
        "BEST VAL :",
        f"PCA={best_val['pca_dim']}",
        f"RMSE={best_val['validation']['rmse']:.6f}",
        f"R2={best_val['validation']['r2']:.4f}",
    )

    print()
    print(
        "Summary:",
        summary_path.resolve(),
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
