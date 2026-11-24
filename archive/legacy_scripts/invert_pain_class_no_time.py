from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from CESN.core.cesn import ContextualESNTorch
from CESN.core.parametersCESN import (
    LATENT_DATA_PATH,
    MODEL_PATH,
    PAIN_DIM,
    TRAIN_PAIRS,
    VAL_PAIRS,
    TEST_PAIRS,
    SPLIT_SEED,
)


OBSERVATION_FRACTIONS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

OUT_DIR = (
    Path(MODEL_PATH).parent
    / "inversion_no_time_v1"
)

SUMMARY_PATH = (
    OUT_DIR
    / "pain_classification_by_prefix_no_time.json"
)

CSV_PATH = (
    OUT_DIR
    / "pain_classification_by_prefix_no_time.csv"
)


def build_pair_splits(path):
    pair_ids = set()

    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            pair_ids.add(
                int(
                    h5[key].attrs["pair_id"]
                )
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


def load_sequences(path, pair_ids):
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
def candidate_rmse(
    model,
    observed_z,
    z_static_pca,
    candidate_label,
):
    T = int(
        observed_z.shape[0]
    )

    U = build_input_sequence(
        candidate_label,
        z_static_pca,
        T,
    )

    pred = (
        model.predict(U)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    pred
                    - observed_z
                ) ** 2
            )
        )
    )


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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model = ContextualESNTorch.from_state_dict(
        checkpoint["model_state"],
        device=device,
    )

    pca_mean = (
        checkpoint["pca_mean"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    pca_components = (
        checkpoint["pca_components"]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    splits = build_pair_splits(
        LATENT_DATA_PATH
    )

    test_sequences = load_sequences(
        LATENT_DATA_PATH,
        splits["test"],
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 92)
    print(
        "RGB-ONLY CESN INVERSION - NO TIME"
    )
    print("=" * 92)

    print(
        "TEST trajectories =",
        len(test_sequences),
    )

    print(
        "STATIC PCA DIM =",
        pca_components.shape[0],
    )

    print(
        "No explicit time input."
    )

    print("=" * 92)

    rows = []

    for fraction in OBSERVATION_FRACTIONS:
        correct = 0
        total = 0

        correct_safe = 0
        total_safe = 0

        correct_pain = 0
        total_pain = 0

        margins = []

        for item in test_sequences:
            T = int(
                item["z_dynamic"].shape[0]
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

            observed_z = item[
                "z_dynamic"
            ][:n_obs]

            z_static_pca = transform_static(
                item["z_static"],
                pca_mean,
                pca_components,
            )

            rmse_safe = candidate_rmse(
                model,
                observed_z,
                z_static_pca,
                0,
            )

            rmse_pain = candidate_rmse(
                model,
                observed_z,
                z_static_pca,
                1,
            )

            predicted = (
                0
                if rmse_safe <= rmse_pain
                else 1
            )

            true_label = int(
                item["pain_label"]
            )

            ok = int(
                predicted
                == true_label
            )

            correct += ok
            total += 1

            if true_label == 0:
                total_safe += 1
                correct_safe += ok
            else:
                total_pain += 1
                correct_pain += ok

            margins.append(
                abs(
                    rmse_safe
                    - rmse_pain
                )
            )

        accuracy = (
            correct
            / max(
                total,
                1,
            )
        )

        safe_accuracy = (
            correct_safe
            / max(
                total_safe,
                1,
            )
        )

        pain_accuracy = (
            correct_pain
            / max(
                total_pain,
                1,
            )
        )

        mean_margin = float(
            np.mean(
                margins
            )
        )

        row = {
            "fraction":
                float(
                    fraction
                ),
            "accuracy":
                float(
                    accuracy
                ),
            "non_painful_accuracy":
                float(
                    safe_accuracy
                ),
            "painful_accuracy":
                float(
                    pain_accuracy
                ),
            "correct":
                int(
                    correct
                ),
            "total":
                int(
                    total
                ),
            "mean_absolute_rmse_margin":
                mean_margin,
        }

        rows.append(row)

        print(
            f"OBS={fraction * 100:>5.0f}%"
            f" | ACC="
            f"{accuracy * 100:>6.2f}%"
            f" ({correct:>2}/{total})"
            f" | SAFE="
            f"{safe_accuracy * 100:>6.2f}%"
            f" | PAIN="
            f"{pain_accuracy * 100:>6.2f}%"
            f" | margin="
            f"{mean_margin:.6f}"
        )

    first_90 = None

    for row in rows:
        if row["accuracy"] >= 0.90:
            first_90 = row["fraction"]
            break

    summary = {
        "architecture":
            "rgb_only_inverse_no_time_v1",

        "first_fraction_at_or_above_90_percent_accuracy":
            first_90,

        "results":
            rows,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 92)

    if first_90 is None:
        print(
            "90% overall accuracy was NOT reached."
        )
    else:
        print(
            "First observation fraction with >=90% accuracy:",
            f"{first_90 * 100:.0f}%",
        )

    print(
        "Summary:",
        SUMMARY_PATH.resolve(),
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
