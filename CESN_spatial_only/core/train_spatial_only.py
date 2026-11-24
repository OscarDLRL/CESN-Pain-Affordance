
from __future__ import annotations

import json
import h5py
import numpy as np
import torch

from CESN.core.cesn import ContextualESNTorch
from CESN_spatial_only.core.parameters_spatial_only import (
    LATENT_DATA_PATH, RGB_DATA_PATH, OUT_DIR, MODEL_PATH,
    STATIC_PCA_DIM, RESERVOIR_DIM, RIDGE, SPECTRAL_RADIUS,
    LEAK_RATE, INPUT_SCALE, BIAS_SCALE, SEED,
    GRASP_X_SCALE, GRASP_Y_SCALE, TRAIN_PAIRS, TEST_PAIRS,
    SAFE_TARGET_LOCAL_X, PAIN_TARGET_LOCAL_X,
    TIME_INDEX_SCALE, OUTPUT_DIM, CONTEXT_DIM,
)


def build_pair_splits():
    pair_ids = set()
    with h5py.File(LATENT_DATA_PATH, "r") as h5:
        for key in h5.keys():
            if str(key).startswith("trajectory_"):
                pair_ids.add(int(h5[key].attrs["pair_id"]))

    pair_ids = np.asarray(sorted(pair_ids), dtype=np.int64)
    if len(pair_ids) != 81:
        raise RuntimeError(f"Expected 81 pairs, got {len(pair_ids)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(pair_ids)

    train = set(pair_ids[:TRAIN_PAIRS].tolist())
    test = set(pair_ids[TRAIN_PAIRS:].tolist())

    if len(test) != TEST_PAIRS:
        raise RuntimeError(f"Expected {TEST_PAIRS} test pairs, got {len(test)}")
    return {"train": train, "test": test}


def load_raw_sequences(pair_ids):
    rows = []
    with h5py.File(LATENT_DATA_PATH, "r") as latent_h5, \
         h5py.File(RGB_DATA_PATH, "r") as rgb_h5:

        for key in sorted(latent_h5.keys()):
            if not str(key).startswith("trajectory_"):
                continue

            lg = latent_h5[key]
            pair_id = int(lg.attrs["pair_id"])
            if pair_id not in pair_ids:
                continue

            rg = rgb_h5[key]
            label = int(lg.attrs["pain_label"])
            object_xyz = np.asarray(rg["object_xyz"][0], dtype=np.float32)

            knife_x = float(object_xyz[0])
            knife_y = float(object_xyz[1])

            grasp_x = knife_x + (
                SAFE_TARGET_LOCAL_X if label == 0 else PAIN_TARGET_LOCAL_X
            )
            grasp_y = knife_y

            rows.append({
                "key": str(key),
                "pair_id": pair_id,
                "pain_label": label,
                "z_static": np.asarray(lg["z_static"], dtype=np.float32),
                "z_dynamic": np.asarray(lg["z_dynamic"], dtype=np.float32),
                "knife_x": knife_x,
                "knife_y": knife_y,
                "grasp_x": float(grasp_x),
                "grasp_y": float(grasp_y),
            })
    return rows


def fit_static_pca(raw_sequences):
    X = np.stack([x["z_static"] for x in raw_sequences], axis=0).astype(np.float64)
    mean = X.mean(axis=0)
    centered = X - mean[None, :]
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:STATIC_PCA_DIM].astype(np.float32)

    total_var = float(np.sum(S ** 2))
    explained = float(np.sum(S[:STATIC_PCA_DIM] ** 2) / max(total_var, 1e-12))
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "explained_ratio": explained,
    }


def transform_static(z_static, pca):
    return ((z_static - pca["mean"]) @ pca["components"].T).astype(np.float32)


def prepare_sequences(raw_sequences, pca):
    out = []
    for item in raw_sequences:
        static = transform_static(item["z_static"], pca)
        grasp = np.asarray([
            item["grasp_x"] / GRASP_X_SCALE,
            item["grasp_y"] / GRASP_Y_SCALE,
        ], dtype=np.float32)

        base_context = np.concatenate([static, grasp], axis=0).astype(np.float32)
        if base_context.shape[0] != CONTEXT_DIM - 1:
            raise RuntimeError(
                f"Expected base context dim {CONTEXT_DIM-1}, got {base_context.shape[0]}"
            )

        T = int(item["z_dynamic"].shape[0])
        base = np.repeat(base_context[None, :], T, axis=0)
        time_index = (
            np.arange(T, dtype=np.float32) / float(TIME_INDEX_SCALE)
        )[:, None]

        U = np.concatenate([base, time_index], axis=1).astype(np.float32)

        out.append({**item, "U": U, "Y": item["z_dynamic"].astype(np.float32)})
    return out


def r2_score_global(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ss_res = float(np.sum((target - pred) ** 2))
    target_mean = np.mean(target, axis=0, keepdims=True)
    ss_tot = float(np.sum((target - target_mean) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


@torch.no_grad()
def evaluate(model, sequences):
    preds, targets = [], []
    for item in sequences:
        pred = model.predict(item["U"]).detach().cpu().numpy().astype(np.float32)
        preds.append(pred)
        targets.append(item["Y"])

    pred_all = np.concatenate(preds, axis=0)
    target_all = np.concatenate(targets, axis=0)
    error = pred_all - target_all

    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": r2_score_global(pred_all, target_all),
    }


def make_model(device):
    return ContextualESNTorch(
        input_dim=CONTEXT_DIM,
        output_dim=OUTPUT_DIM,
        reservoir_size=RESERVOIR_DIM,
        spectral_radius=SPECTRAL_RADIUS,
        leak_rate=LEAK_RATE,
        input_scale=INPUT_SCALE,
        bias_scale=BIAS_SCALE,
        ridge=RIDGE,
        seed=SEED,
        device=device,
    )


def print_metrics(prefix, metrics):
    print(
        f"{prefix:<12s}RMSE={metrics['rmse']:.6f}"
        f" | MAE={metrics['mae']:.6f}"
        f" | R2={metrics['r2']:.4f}"
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
    print("=" * 104)
    print("CESN SPATIAL-ONLY | NATIVE256 LATENT200 | RIDGE0.5 | FINAL70")
    print("=" * 104)
    print("INPUT  = [static PCA16, grasp XY(2), causal time index(1)]")
    print("TASK   = NOT PROVIDED TO CESN")
    print("OUTPUT = z_dynamic native256 latent200")
    print("CONTEXT DIM =", CONTEXT_DIM)
    print("RESERVOIR   =", RESERVOIR_DIM)
    print("RIDGE       =", RIDGE)
    print("TIME INPUT  = frame_index / 100 (causal)")
    print("=" * 104)

    model = make_model(device)
    model.fit(
        [(item["U"], item["Y"]) for item in train_seq],
        solve_dtype=torch.float64,
    )

    train_metrics = evaluate(model, train_seq)
    test_metrics = evaluate(model, test_seq)

    print(f"PCA16 explained variance = {pca['explained_ratio']*100:.4f}%")
    print_metrics("TRAIN70", train_metrics)
    print_metrics("TEST11", test_metrics)

    torch.save({
        "model_state": model.state_dict(),
        "pca_mean": torch.from_numpy(pca["mean"]),
        "pca_components": torch.from_numpy(pca["components"]),
        "pca_explained_ratio": float(pca["explained_ratio"]),
        "architecture": "native256_latent200_spatial_only_ridge05_final70",
        "context": "[static_pca16, grasp_xy(2), causal_time_index(1)]",
        "task_as_context": False,
        "latent_dataset": str(LATENT_DATA_PATH),
        "train_pairs": sorted(list(splits["train"])),
        "test_pairs": sorted(list(splits["test"])),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }, MODEL_PATH)

    (OUT_DIR / "training_summary.json").write_text(
        json.dumps({
            "architecture": "native256_latent200_spatial_only_ridge05_final70",
            "task_as_context": False,
            "context_dim": CONTEXT_DIM,
            "reservoir_size": RESERVOIR_DIM,
            "ridge": RIDGE,
            "train": train_metrics,
            "test": test_metrics,
            "pca16_explained": float(pca["explained_ratio"]),
        }, indent=2),
        encoding="utf-8",
    )

    print("Saved model:", MODEL_PATH.resolve())


if __name__ == "__main__":
    main()
