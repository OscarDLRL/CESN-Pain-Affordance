from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from CESN.core.cesn import ContextualESNTorch
from CESN.core.autoencoder import PainStaticDynamicAutoencoder
from CESN.core.parametersCESN import (
    LATENT_DATA_PATH,
    MODEL_PATH,
    PAIN_DIM,
    TRAIN_PAIRS,
    VAL_PAIRS,
    TEST_PAIRS,
    SPLIT_SEED,
)


# ============================================================
# Final evaluation configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    ROOT
    / "datasets"
    / "pain_affordance_162"
    / "pain_affordance_162.h5"
)

AUTOENCODER_PATH = (
    ROOT
    / "CESN"
    / "outputs"
    / "pain_affordance_static_dynamic_autoencoder_v4"
    / "best_static_dynamic_autoencoder_v4.pt"
)

OUT_DIR = (
    ROOT
    / "CESN"
    / "outputs"
    / "rgb_only_cesn_no_time_v1"
    / "final_metrics"
)

HISTORY_FRACTIONS = [
    x / 100.0
    for x in range(5, 101, 5)
]

CONSISTENCY_TARGET = 0.90

IMAGE_SIZE = 128
STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 128

GIF_HISTORY_FRACTION = 0.20
GIF_DURATION_MS = 120


# ============================================================
# Paths
# ============================================================

HISTORY_CSV = OUT_DIR / "history_consistency_table.csv"
HISTORY_JSON = OUT_DIR / "history_consistency_table.json"

AMBIGUITY_PNG = OUT_DIR / "ambiguity_vs_history.png"
SPLIT_GRID_PNG = OUT_DIR / "train_validation_test_grid.png"

GIF_PATH = OUT_DIR / "true_vs_prediction.gif"
GIF_INFO_JSON = OUT_DIR / "true_vs_prediction_info.json"

SUMMARY_JSON = OUT_DIR / "final_metrics_summary.json"


# ============================================================
# Split
# ============================================================

def build_pair_splits(path):
    pair_ids = set()

    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            if not str(key).startswith("trajectory_"):
                continue
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

    validation = set(
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
        "validation": validation,
        "test": test,
    }


# ============================================================
# Load latent data
# ============================================================

def load_latent_sequences(path, pair_ids):
    sequences = []

    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys()):
            if not str(key).startswith("trajectory_"):
                continue

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


# ============================================================
# CESN context
# ============================================================

def pain_onehot(label):
    out = np.zeros(
        PAIN_DIM,
        dtype=np.float32,
    )

    out[int(label)] = 1.0
    return out


def transform_static(
    z_static,
    pca_mean,
    pca_components,
):
    centered = (
        np.asarray(
            z_static,
            dtype=np.float32,
        )
        - pca_mean
    )

    return (
        centered
        @ pca_components.T
    ).astype(np.float32)


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

    return np.concatenate(
        [
            pain,
            static,
        ],
        axis=1,
    ).astype(np.float32)


@torch.no_grad()
def predict_candidate(
    model,
    z_static_pca,
    T,
    candidate_label,
):
    U = build_input_sequence(
        candidate_label,
        z_static_pca,
        T,
    )

    return (
        model.predict(U)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def prefix_rmse(
    prediction,
    target,
    fraction,
):
    T = int(
        target.shape[0]
    )

    n_obs = max(
        1,
        int(
            np.ceil(
                fraction * T
            )
        ),
    )

    n_obs = min(
        n_obs,
        T,
    )

    err = (
        prediction[:n_obs]
        - target[:n_obs]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                err ** 2
            )
        )
    )

    return rmse, n_obs


# ============================================================
# Evaluate every prefix
# ============================================================

def evaluate_prefixes(
    model,
    test_sequences,
    pca_mean,
    pca_components,
):
    per_trajectory = {}

    for item in test_sequences:

        z_static_pca = transform_static(
            item["z_static"],
            pca_mean,
            pca_components,
        )

        T = int(
            item["z_dynamic"].shape[0]
        )

        pred_safe = predict_candidate(
            model,
            z_static_pca,
            T,
            candidate_label=0,
        )

        pred_pain = predict_candidate(
            model,
            z_static_pca,
            T,
            candidate_label=1,
        )

        rows = []

        for fraction in HISTORY_FRACTIONS:

            safe_rmse, n_obs = prefix_rmse(
                pred_safe,
                item["z_dynamic"],
                fraction,
            )

            pain_rmse, _ = prefix_rmse(
                pred_pain,
                item["z_dynamic"],
                fraction,
            )

            predicted = (
                0
                if safe_rmse <= pain_rmse
                else 1
            )

            true_label = int(
                item["pain_label"]
            )

            correct = bool(
                predicted == true_label
            )

            # 0 = maximally separable, 1 = maximally ambiguous.
            denom = (
                safe_rmse
                + pain_rmse
                + 1e-12
            )

            ambiguity = float(
                1.0
                - abs(
                    safe_rmse
                    - pain_rmse
                )
                / denom
            )

            rows.append(
                {
                    "fraction": float(
                        fraction
                    ),
                    "n_observed_frames": int(
                        n_obs
                    ),
                    "safe_rmse": float(
                        safe_rmse
                    ),
                    "pain_rmse": float(
                        pain_rmse
                    ),
                    "predicted_label": int(
                        predicted
                    ),
                    "true_label": int(
                        true_label
                    ),
                    "correct": correct,
                    "ambiguity": ambiguity,
                }
            )

        # Earliest prefix after which every later decision is correct.
        stable_fraction = None
        stable_frames = None

        for i, row in enumerate(rows):
            if all(
                later["correct"]
                for later in rows[i:]
            ):
                stable_fraction = float(
                    row["fraction"]
                )
                stable_frames = int(
                    row["n_observed_frames"]
                )
                break

        per_trajectory[
            item["key"]
        ] = {
            "pair_id": int(
                item["pair_id"]
            ),
            "pain_label": int(
                item["pain_label"]
            ),
            "total_frames": T,
            "stable_fraction":
                stable_fraction,
            "stable_frames":
                stable_frames,
            "rows": rows,
        }

    return per_trajectory


# ============================================================
# History consistency table
# ============================================================

def build_history_table(
    per_trajectory,
):
    trajectory_values = list(
        per_trajectory.values()
    )

    summary_rows = []

    for idx, fraction in enumerate(
        HISTORY_FRACTIONS
    ):

        instant_correct = []
        stable_correct = []

        safe_instant = []
        pain_instant = []

        safe_stable = []
        pain_stable = []

        ambiguities = []

        observed_frames = []

        for item in trajectory_values:
            row = item["rows"][idx]

            instant = bool(
                row["correct"]
            )

            stable = all(
                later["correct"]
                for later in item[
                    "rows"
                ][idx:]
            )

            instant_correct.append(
                instant
            )

            stable_correct.append(
                stable
            )

            ambiguities.append(
                float(
                    row["ambiguity"]
                )
            )

            observed_frames.append(
                int(
                    row[
                        "n_observed_frames"
                    ]
                )
            )

            if item[
                "pain_label"
            ] == 0:
                safe_instant.append(
                    instant
                )
                safe_stable.append(
                    stable
                )
            else:
                pain_instant.append(
                    instant
                )
                pain_stable.append(
                    stable
                )

        summary_rows.append(
            {
                "history_percent":
                    int(
                        round(
                            fraction * 100
                        )
                    ),

                "mean_observed_frames":
                    float(
                        np.mean(
                            observed_frames
                        )
                    ),

                "instant_accuracy":
                    float(
                        np.mean(
                            instant_correct
                        )
                    ),

                "stable_accuracy":
                    float(
                        np.mean(
                            stable_correct
                        )
                    ),

                "safe_instant_accuracy":
                    float(
                        np.mean(
                            safe_instant
                        )
                    ),

                "pain_instant_accuracy":
                    float(
                        np.mean(
                            pain_instant
                        )
                    ),

                "safe_stable_accuracy":
                    float(
                        np.mean(
                            safe_stable
                        )
                    ),

                "pain_stable_accuracy":
                    float(
                        np.mean(
                            pain_stable
                        )
                    ),

                "mean_ambiguity":
                    float(
                        np.mean(
                            ambiguities
                        )
                    ),

                "median_ambiguity":
                    float(
                        np.median(
                            ambiguities
                        )
                    ),
            }
        )

    return summary_rows


# ============================================================
# Ambiguity plot
# ============================================================

def save_ambiguity_plot(
    history_rows,
):
    x = [
        row[
            "history_percent"
        ]
        for row in history_rows
    ]

    mean_ambiguity = [
        row[
            "mean_ambiguity"
        ]
        for row in history_rows
    ]

    median_ambiguity = [
        row[
            "median_ambiguity"
        ]
        for row in history_rows
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        mean_ambiguity,
        marker="o",
        label="Mean ambiguity",
    )

    plt.plot(
        x,
        median_ambiguity,
        marker="o",
        label="Median ambiguity",
    )

    plt.xlabel(
        "Observed trajectory (%)"
    )

    plt.ylabel(
        "Ambiguity score"
    )

    plt.ylim(
        0.0,
        1.02,
    )

    plt.title(
        "SAFE vs PAIN ambiguity as history increases"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        AMBIGUITY_PNG,
        dpi=180,
    )

    plt.close()


# ============================================================
# 9x9 split grid
# ============================================================

def save_split_grid(
    splits,
):
    grid = np.full(
        (9, 9),
        np.nan,
        dtype=np.float32,
    )

    labels = np.empty(
        (9, 9),
        dtype=object,
    )

    for pair_id in range(81):
        object_id = (
            pair_id // 9
        )

        start_id = (
            pair_id % 9
        )

        if pair_id in splits["train"]:
            value = 0.0
            label = "TR"
        elif pair_id in splits["validation"]:
            value = 1.0
            label = "VA"
        elif pair_id in splits["test"]:
            value = 2.0
            label = "TE"
        else:
            continue

        grid[
            object_id,
            start_id,
        ] = value

        labels[
            object_id,
            start_id,
        ] = (
            f"{label}\n"
            f"{pair_id}"
        )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        grid,
        interpolation="nearest",
        aspect="equal",
    )

    for row in range(9):
        for col in range(9):
            ax.text(
                col,
                row,
                labels[row, col],
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_xlabel(
        "Robot start ID"
    )

    ax.set_ylabel(
        "Knife position ID"
    )

    ax.set_xticks(
        range(9)
    )

    ax.set_yticks(
        range(9)
    )

    ax.set_title(
        "Dataset split across the 9 × 9 physical-condition grid"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[
            0,
            1,
            2,
        ],
    )

    cbar.ax.set_yticklabels(
        [
            "Train",
            "Validation",
            "Test",
        ]
    )

    plt.tight_layout()

    plt.savefig(
        SPLIT_GRID_PNG,
        dpi=180,
    )

    plt.close()


# ============================================================
# Autoencoder loading
# ============================================================

def load_autoencoder(
    path,
    device,
):
    model = PainStaticDynamicAutoencoder(
        static_latent_dim=
            STATIC_LATENT_DIM,
        dynamic_latent_dim=
            DYNAMIC_LATENT_DIM,
    ).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    state = checkpoint

    if isinstance(
        checkpoint,
        dict,
    ):
        for key in [
            "model_state",
            "model_state_dict",
            "state_dict",
        ]:
            if key in checkpoint:
                state = checkpoint[key]
                break

    model.load_state_dict(
        state
    )

    model.eval()

    return model


# ============================================================
# Image helpers
# ============================================================

def rgb_uint8_to_tensor(
    rgb,
    device,
):
    image = torch.from_numpy(
        np.asarray(
            rgb,
            dtype=np.float32,
        )
        / 255.0
    )

    image = image.permute(
        2,
        0,
        1,
    )[None, ...]

    image = F.interpolate(
        image,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return image.to(
        device
    )


def tensor_to_pil(
    x,
):
    x = (
        x.detach()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    if x.ndim == 4:
        x = x[0]

    arr = (
        x.permute(
            1,
            2,
            0,
        )
        .numpy()
        * 255.0
    )

    arr = np.clip(
        arr,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        arr
    )


# ============================================================
# Representative trajectory for GIF
# ============================================================

def select_representative_pain(
    per_trajectory,
    target_fraction,
):
    candidates = []

    target_idx = min(
        range(
            len(
                HISTORY_FRACTIONS
            )
        ),
        key=lambda i:
            abs(
                HISTORY_FRACTIONS[i]
                - target_fraction
            ),
    )

    for key, item in per_trajectory.items():

        if item["pain_label"] != 1:
            continue

        row = item["rows"][
            target_idx
        ]

        stable_here = all(
            later["correct"]
            for later in item[
                "rows"
            ][target_idx:]
        )

        if not stable_here:
            continue

        margin = abs(
            row["safe_rmse"]
            - row["pain_rmse"]
        )

        candidates.append(
            (
                margin,
                key,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No consistently-correct PAIN test trajectory "
            "was found at the GIF history fraction."
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    # Choose the median difficulty rather than easiest case.
    return candidates[
        len(candidates) // 2
    ][1]


# ============================================================
# TRUE | PREDICTION GIF
# ============================================================

@torch.no_grad()
def save_true_vs_prediction_gif(
    cesn_model,
    autoencoder,
    pca_mean,
    pca_components,
    per_trajectory,
    trajectory_key,
    device,
):
    info = per_trajectory[
        trajectory_key
    ]

    with h5py.File(
        RAW_DATA_PATH,
        "r",
    ) as h5:
        raw_group = h5[
            trajectory_key
        ]

        true_rgb = np.asarray(
            raw_group["rgb"]
        )

        raw_label = int(
            raw_group.attrs[
                "label"
            ]
        )

    with h5py.File(
        LATENT_DATA_PATH,
        "r",
    ) as h5:
        latent_group = h5[
            trajectory_key
        ]

        z_static = np.asarray(
            latent_group[
                "z_static"
            ],
            dtype=np.float32,
        )

        z_dynamic_true = np.asarray(
            latent_group[
                "z_dynamic"
            ],
            dtype=np.float32,
        )

    T = min(
        len(true_rgb),
        len(z_dynamic_true),
    )

    z_static_pca = transform_static(
        z_static,
        pca_mean,
        pca_components,
    )

    history_frames = max(
        1,
        int(
            np.ceil(
                GIF_HISTORY_FRACTION
                * T
            )
        ),
    )

    # Infer class ONLY from the observed prefix.
    safe_pred = predict_candidate(
        cesn_model,
        z_static_pca,
        T,
        0,
    )

    pain_pred = predict_candidate(
        cesn_model,
        z_static_pca,
        T,
        1,
    )

    safe_rmse, _ = prefix_rmse(
        safe_pred,
        z_dynamic_true,
        GIF_HISTORY_FRACTION,
    )

    pain_rmse, _ = prefix_rmse(
        pain_pred,
        z_dynamic_true,
        GIF_HISTORY_FRACTION,
    )

    inferred_label = (
        0
        if safe_rmse <= pain_rmse
        else 1
    )

    predicted_z = (
        safe_pred
        if inferred_label == 0
        else pain_pred
    )

    frame0 = rgb_uint8_to_tensor(
        true_rgb[0],
        device,
    )

    z_static_tensor = torch.from_numpy(
        z_static
    ).to(
        device=device,
        dtype=torch.float32,
    )[None, :]

    frames = []

    for t in range(T):

        true_image = (
            Image.fromarray(
                true_rgb[t]
            )
            .convert("RGB")
            .resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            )
        )

        z_dynamic_tensor = (
            torch.from_numpy(
                predicted_z[t]
            )
            .to(
                device=device,
                dtype=torch.float32,
            )[None, :]
        )

        delta_pred = (
            autoencoder.decode_dynamic(
                z_dynamic_tensor,
                z_static_tensor,
            )
        )

        pred_frame = torch.clamp(
            frame0
            + delta_pred,
            0.0,
            1.0,
        )

        predicted_image = tensor_to_pil(
            pred_frame
        )

        header = 42
        canvas = Image.new(
            "RGB",
            (
                IMAGE_SIZE * 2,
                IMAGE_SIZE
                + header,
            ),
            "white",
        )

        canvas.paste(
            true_image,
            (
                0,
                header,
            ),
        )

        canvas.paste(
            predicted_image,
            (
                IMAGE_SIZE,
                header,
            ),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        draw.text(
            (
                8,
                5,
            ),
            "TRUE RGB",
            fill="black",
        )

        draw.text(
            (
                IMAGE_SIZE + 8,
                5,
            ),
            "CESN PREDICTION",
            fill="black",
        )

        phase = (
            "OBSERVED PREFIX"
            if t < history_frames
            else "PREDICTED FUTURE"
        )

        draw.text(
            (
                8,
                22,
            ),
            (
                f"{phase} | "
                f"frame {t+1}/{T} | "
                f"true={'PAIN' if raw_label else 'SAFE'} | "
                f"inferred={'PAIN' if inferred_label else 'SAFE'}"
            ),
            fill="black",
        )

        frames.append(
            canvas
        )

    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=
            frames[1:],
        duration=
            GIF_DURATION_MS,
        loop=0,
        optimize=False,
    )

    gif_info = {
        "trajectory":
            trajectory_key,

        "pair_id":
            int(
                info["pair_id"]
            ),

        "true_label":
            int(
                raw_label
            ),

        "inferred_label_at_history":
            int(
                inferred_label
            ),

        "history_fraction":
            float(
                GIF_HISTORY_FRACTION
            ),

        "history_frames":
            int(
                history_frames
            ),

        "total_frames":
            int(
                T
            ),

        "safe_prefix_rmse":
            float(
                safe_rmse
            ),

        "pain_prefix_rmse":
            float(
                pain_rmse
            ),
    }

    GIF_INFO_JSON.write_text(
        json.dumps(
            gif_info,
            indent=2,
        ),
        encoding="utf-8",
    )

    return gif_info


# ============================================================
# Save table
# ============================================================

def save_history_table(
    history_rows,
):
    with HISTORY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            history_rows
        )

    HISTORY_JSON.write_text(
        json.dumps(
            history_rows,
            indent=2,
        ),
        encoding="utf-8",
    )


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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_pair_splits(
        LATENT_DATA_PATH
    )

    test_sequences = load_latent_sequences(
        LATENT_DATA_PATH,
        splits["test"],
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    cesn_model = (
        ContextualESNTorch.from_state_dict(
            checkpoint[
                "model_state"
            ],
            device=device,
        )
    )

    pca_mean = (
        checkpoint[
            "pca_mean"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    pca_components = (
        checkpoint[
            "pca_components"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    print()
    print("=" * 100)
    print(
        "FINAL RGB-ONLY PAIN AFFORDANCE METRICS"
    )
    print("=" * 100)

    print(
        "Test trajectories:",
        len(
            test_sequences
        ),
    )

    print(
        "History sweep:",
        [
            int(
                x * 100
            )
            for x
            in HISTORY_FRACTIONS
        ],
    )

    print(
        "Consistency criterion:",
        f">={CONSISTENCY_TARGET*100:.0f}% "
        "of test trajectories remain correct "
        "from that prefix onward",
    )

    print(
        "No extrapolation experiment."
    )

    print("=" * 100)

    per_trajectory = evaluate_prefixes(
        cesn_model,
        test_sequences,
        pca_mean,
        pca_components,
    )

    history_rows = build_history_table(
        per_trajectory
    )

    save_history_table(
        history_rows
    )

    save_ambiguity_plot(
        history_rows
    )

    save_split_grid(
        splits
    )

    stable_threshold = None

    for row in history_rows:
        if (
            row[
                "stable_accuracy"
            ]
            >= CONSISTENCY_TARGET
        ):
            stable_threshold = int(
                row[
                    "history_percent"
                ]
            )
            break

    print()
    print(
        f"{'HISTORY':>8}"
        f" {'INST ACC':>10}"
        f" {'STABLE':>10}"
        f" {'SAFE ST':>10}"
        f" {'PAIN ST':>10}"
        f" {'AMBIG':>10}"
    )

    for row in history_rows:
        print(
            f"{row['history_percent']:>7d}%"
            f" {row['instant_accuracy']*100:>9.2f}%"
            f" {row['stable_accuracy']*100:>9.2f}%"
            f" {row['safe_stable_accuracy']*100:>9.2f}%"
            f" {row['pain_stable_accuracy']*100:>9.2f}%"
            f" {row['mean_ambiguity']:>10.4f}"
        )

    print()

    if stable_threshold is None:
        print(
            "Stable >=90% threshold: NOT REACHED"
        )
    else:
        print(
            "Stable >=90% threshold:",
            f"{stable_threshold}%",
        )

    autoencoder = load_autoencoder(
        AUTOENCODER_PATH,
        device,
    )

    gif_trajectory = (
        select_representative_pain(
            per_trajectory,
            GIF_HISTORY_FRACTION,
        )
    )

    gif_info = (
        save_true_vs_prediction_gif(
            cesn_model,
            autoencoder,
            pca_mean,
            pca_components,
            per_trajectory,
            gif_trajectory,
            device,
        )
    )

    summary = {
        "evaluation":
            "RGB-only pain affordance context inversion",

        "no_extrapolation":
            True,

        "test_pairs":
            TEST_PAIRS,

        "test_trajectories":
            len(
                test_sequences
            ),

        "consistency_target":
            CONSISTENCY_TARGET,

        "stable_history_threshold_percent":
            stable_threshold,

        "gif":
            gif_info,

        "outputs":
            {
                "history_table_csv":
                    str(
                        HISTORY_CSV
                    ),

                "history_table_json":
                    str(
                        HISTORY_JSON
                    ),

                "ambiguity_plot":
                    str(
                        AMBIGUITY_PNG
                    ),

                "split_grid":
                    str(
                        SPLIT_GRID_PNG
                    ),

                "true_vs_prediction_gif":
                    str(
                        GIF_PATH
                    ),
            },
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        "History table :",
        HISTORY_CSV.resolve(),
    )

    print(
        "Ambiguity     :",
        AMBIGUITY_PNG.resolve(),
    )

    print(
        "Split grid    :",
        SPLIT_GRID_PNG.resolve(),
    )

    print(
        "GIF           :",
        GIF_PATH.resolve(),
    )

    print(
        "Summary       :",
        SUMMARY_JSON.resolve(),
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
