from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from CESN.core.cesn import ContextualESNTorch

from CESN.core.parametersCESN import (
    LATENT_DATA_PATH,
    OUT_DIR,
    MODEL_PATH,
    SUMMARY_PATH,
    PAIN_DIM,
    STATIC_PCA_DIM,
    CONTEXT_DIM,
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
    ARCHITECTURE_TAG,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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

    rng = np.random.default_rng(
        SPLIT_SEED
    )
    rng.shuffle(pair_ids)

    return {
        "train": set(
            pair_ids[:TRAIN_PAIRS].tolist()
        ),
        "validation": set(
            pair_ids[
                TRAIN_PAIRS:
                TRAIN_PAIRS + VAL_PAIRS
            ].tolist()
        ),
        "test": set(
            pair_ids[
                TRAIN_PAIRS + VAL_PAIRS:
            ].tolist()
        ),
    }


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
                }
            )

    return sequences


def fit_static_pca(
    train_sequences,
    n_components,
):
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

    explained_ratio = (
        explained_variance[
            :n_components
        ].sum()
        / max(
            explained_variance.sum(),
            1e-12,
        )
    )

    return {
        "mean": mean,
        "components": components,
        "explained_ratio": float(
            explained_ratio
        ),
    }


def transform_static(
    z_static,
    pca,
):
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


def pain_onehot(label):
    out = np.zeros(
        PAIN_DIM,
        dtype=np.float32,
    )
    out[int(label)] = 1.0
    return out


def build_input_sequence(
    pain_label,
    z_static_pca,
    T,
):
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

    U = np.concatenate(
        [
            pain,
            static,
        ],
        axis=1,
    )

    if U.shape[1] != CONTEXT_DIM:
        raise RuntimeError(
            f"Expected input dim {CONTEXT_DIM}, "
            f"got {U.shape[1]}"
        )

    return U.astype(
        np.float32
    )


def prepare_sequences(
    raw_sequences,
    pca,
):
    out = []

    for item in raw_sequences:
        z_static_pca = transform_static(
            item["z_static"],
            pca,
        )

        T = int(
            item["z_dynamic"].shape[0]
        )

        U = build_input_sequence(
            item["pain_label"],
            z_static_pca,
            T,
        )

        out.append(
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

    return out


def r2_score_global(
    pred,
    target,
):
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

    return float(
        1.0
        - ss_res
        / max(
            ss_tot,
            1e-12,
        )
    )


@torch.no_grad()
def evaluate(
    model,
    sequences,
):
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

        preds.append(pred)
        targets.append(
            item["Y"]
        )

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

    return {
        "rmse": float(
            np.sqrt(mse)
        ),
        "mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "r2": r2_score_global(
            pred_all,
            target_all,
        ),
    }


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

    pca = fit_static_pca(
        train_raw,
        STATIC_PCA_DIM,
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

    print()
    print("=" * 84)
    print(
        "RGB-ONLY CESN V5 - NO TIME - PYTORCH CUDA"
    )
    print("=" * 84)

    print(
        "INPUT = "
        "[pain_onehot(2), static_pca(16)]"
    )

    print(
        "OUTPUT = z_dynamic RAW (128)"
    )

    print(
        "CONTEXT DIM =",
        CONTEXT_DIM,
    )

    print(
        "STATIC PCA explained variance =",
        f"{pca['explained_ratio'] * 100:.4f}%",
    )

    print(
        "RESERVOIR =",
        RESERVOIR_SIZE,
    )

    print(
        "RIDGE =",
        RIDGE,
    )

    print("=" * 84)

    model = ContextualESNTorch(
        input_dim=
            CONTEXT_DIM,
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
            for item
            in train_sequences
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

    torch.save(
        {
            "model_state":
                model.state_dict(),

            "pca_mean":
                torch.from_numpy(
                    pca["mean"]
                ),

            "pca_components":
                torch.from_numpy(
                    pca["components"]
                ),

            "pca_explained_ratio":
                pca["explained_ratio"],

            "static_pca_dim":
                STATIC_PCA_DIM,

            "architecture":
                ARCHITECTURE_TAG,
        },
        MODEL_PATH,
    )

    summary = {
        "architecture":
            ARCHITECTURE_TAG,
        "static_pca_dim":
            STATIC_PCA_DIM,
        "reservoir_size":
            RESERVOIR_SIZE,
        "ridge":
            RIDGE,
        "train":
            train_metrics,
        "validation":
            val_metrics,
        "test":
            test_metrics,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("RESULTS")
    print("=" * 84)

    print(
        f"TRAIN RMSE="
        f"{train_metrics['rmse']:.6f}"
        f" | R2="
        f"{train_metrics['r2']:.4f}"
    )

    print(
        f"VAL   RMSE="
        f"{val_metrics['rmse']:.6f}"
        f" | R2="
        f"{val_metrics['r2']:.4f}"
    )

    print(
        f"TEST  RMSE="
        f"{test_metrics['rmse']:.6f}"
        f" | R2="
        f"{test_metrics['r2']:.4f}"
    )

    print()
    print(
        "Model:",
        MODEL_PATH.resolve(),
    )

    print(
        "Summary:",
        SUMMARY_PATH.resolve(),
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
