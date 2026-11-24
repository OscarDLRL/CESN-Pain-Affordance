from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from CESN.core.cesn import ContextualESNTorch
from CESN.core.parameters_grasp_context import (
    LATENT_DATA_PATH,
    OUT_DIR,
    MODEL_PATH,
    SUMMARY_PATH,
    SPLIT_SEED,
    TRAIN_PAIRS,
    VAL_PAIRS,
    TEST_PAIRS,
    FINAL_TRAIN_PAIRS,
    TASK_DIM,
    STATIC_PCA_DIM,
    CONTEXT_DIM,
    OUTPUT_DIM,
    KNIFE_X_VALUES,
    KNIFE_Y_VALUES,
    SAFE_GRASP_OFFSET_X,
    PAIN_GRASP_OFFSET_X,
    GRASP_X_SCALE,
    GRASP_Y_SCALE,
    RESERVOIR_SIZE,
    SPECTRAL_RADIUS,
    LEAK_RATE,
    INPUT_SCALE,
    BIAS_SCALE,
    RIDGE,
    DENSITY,
    RANDOM_SEED,
    ARCHITECTURE_TAG,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def trajectory_number(name: str) -> int:
    return int(
        str(name).split("_")[-1]
    )


def build_pair_splits(path):
    pair_ids = set()

    with h5py.File(
        path,
        "r",
    ) as h5:
        for key in h5.keys():
            if str(key).startswith(
                "trajectory_"
            ):
                pair_ids.add(
                    int(
                        h5[key].attrs[
                            "pair_id"
                        ]
                    )
                )

    pair_ids = np.asarray(
        sorted(pair_ids),
        dtype=np.int64,
    )

    if len(pair_ids) != 81:
        raise RuntimeError(
            f"Expected 81 physical pairs, "
            f"found {len(pair_ids)}"
        )

    rng = np.random.default_rng(
        SPLIT_SEED
    )

    rng.shuffle(
        pair_ids
    )

    train = set(
        pair_ids[
            :TRAIN_PAIRS
        ].tolist()
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
        "train":
            train,

        "validation":
            validation,

        "test":
            test,

        "final_train":
            train | validation,
    }


def decode_knife_xy(
    pair_id: int,
):
    object_id = int(
        pair_id
    ) // 9

    object_row, object_col = divmod(
        object_id,
        3,
    )

    knife_x = float(
        KNIFE_X_VALUES[
            object_col
        ]
    )

    knife_y = float(
        KNIFE_Y_VALUES[
            object_row
        ]
    )

    return (
        knife_x,
        knife_y,
    )


def true_grasp_xy(
    pair_id: int,
    pain_label: int,
):
    knife_x, knife_y = (
        decode_knife_xy(
            pair_id
        )
    )

    if int(pain_label) == 0:
        offset_x = (
            SAFE_GRASP_OFFSET_X
        )
    elif int(pain_label) == 1:
        offset_x = (
            PAIN_GRASP_OFFSET_X
        )
    else:
        raise ValueError(
            f"Unexpected pain label: "
            f"{pain_label}"
        )

    grasp_x = (
        knife_x
        + offset_x
    )

    grasp_y = knife_y

    return (
        float(grasp_x),
        float(grasp_y),
    )


def load_raw_sequences(
    path,
    allowed_pairs,
):
    sequences = []

    with h5py.File(
        path,
        "r",
    ) as h5:
        keys = sorted(
            [
                key
                for key in h5.keys()
                if str(key).startswith(
                    "trajectory_"
                )
            ],
            key=trajectory_number,
        )

        for key in keys:
            g = h5[key]

            pair_id = int(
                g.attrs[
                    "pair_id"
                ]
            )

            if pair_id not in allowed_pairs:
                continue

            pain_label = int(
                g.attrs[
                    "pain_label"
                ]
            )

            knife_x, knife_y = (
                decode_knife_xy(
                    pair_id
                )
            )

            grasp_x, grasp_y = (
                true_grasp_xy(
                    pair_id,
                    pain_label,
                )
            )

            sequences.append(
                {
                    "key":
                        str(key),

                    "pair_id":
                        pair_id,

                    "pain_label":
                        pain_label,

                    "knife_x":
                        knife_x,

                    "knife_y":
                        knife_y,

                    "grasp_x":
                        grasp_x,

                    "grasp_y":
                        grasp_y,

                    "z_static":
                        np.asarray(
                            g[
                                "z_static"
                            ],
                            dtype=np.float32,
                        ),

                    "z_dynamic":
                        np.asarray(
                            g[
                                "z_dynamic"
                            ],
                            dtype=np.float32,
                        ),
                }
            )

    return sequences


def fit_static_pca(
    sequences,
    n_components,
):
    X = np.stack(
        [
            item[
                "z_static"
            ]
            for item
            in sequences
        ],
        axis=0,
    ).astype(
        np.float64
    )

    mean = X.mean(
        axis=0,
        keepdims=True,
    )

    Xc = (
        X
        - mean
    )

    _, S, Vt = np.linalg.svd(
        Xc,
        full_matrices=False,
    )

    components = (
        Vt[
            :n_components
        ]
        .astype(
            np.float32
        )
    )

    mean = (
        mean[0]
        .astype(
            np.float32
        )
    )

    explained_variance = (
        S ** 2
    ) / max(
        X.shape[0] - 1,
        1,
    )

    explained_ratio = float(
        explained_variance[
            :n_components
        ].sum()
        / max(
            explained_variance.sum(),
            1e-12,
        )
    )

    return {
        "mean":
            mean,

        "components":
            components,

        "explained_ratio":
            explained_ratio,
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
        - pca[
            "mean"
        ]
    )

    return (
        centered
        @ pca[
            "components"
        ].T
    ).astype(
        np.float32
    )


def task_onehot(
    pain_label,
):
    out = np.zeros(
        TASK_DIM,
        dtype=np.float32,
    )

    out[
        int(pain_label)
    ] = 1.0

    return out


def build_context(
    item,
    pca,
):
    z_static_pca = transform_static(
        item[
            "z_static"
        ],
        pca,
    )

    grasp_norm = np.array(
        [
            item[
                "grasp_x"
            ]
            / GRASP_X_SCALE,

            item[
                "grasp_y"
            ]
            / GRASP_Y_SCALE,
        ],
        dtype=np.float32,
    )

    context = np.concatenate(
        [
            task_onehot(
                item[
                    "pain_label"
                ]
            ),

            z_static_pca,

            grasp_norm,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    if (
        context.shape[0]
        != CONTEXT_DIM
    ):
        raise RuntimeError(
            f"Expected context dim "
            f"{CONTEXT_DIM}, "
            f"got {context.shape[0]}"
        )

    return (
        context,
        z_static_pca,
    )


def prepare_sequences(
    raw_sequences,
    pca,
):
    out = []

    for item in raw_sequences:
        context, z_static_pca = (
            build_context(
                item,
                pca,
            )
        )

        T = int(
            item[
                "z_dynamic"
            ].shape[0]
        )

        U = np.repeat(
            context[
                None,
                :
            ],
            T,
            axis=0,
        ).astype(
            np.float32
        )

        out.append(
            {
                **item,

                "z_static_pca":
                    z_static_pca,

                "U":
                    U,

                "Y":
                    item[
                        "z_dynamic"
                    ],
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

    ss_res = float(
        np.sum(
            (
                target
                - pred
            ) ** 2
        )
    )

    target_mean = np.mean(
        target,
        axis=0,
        keepdims=True,
    )

    ss_tot = float(
        np.sum(
            (
                target
                - target_mean
            ) ** 2
        )
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
                item[
                    "U"
                ]
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        preds.append(
            pred
        )

        targets.append(
            item[
                "Y"
            ]
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

    return {
        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),

        "mae":
            float(
                np.mean(
                    np.abs(
                        error
                    )
                )
            ),

        "r2":
            r2_score_global(
                pred_all,
                target_all,
            ),
    }


def make_model(
    device,
):
    return ContextualESNTorch(
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

        density=
            DENSITY,
    )


def fit_model(
    model,
    sequences,
):
    model.fit(
        [
            (
                item[
                    "U"
                ],

                item[
                    "Y"
                ],
            )
            for item
            in sequences
        ],
        solve_dtype=
            torch.float64,
    )


def print_metrics(
    prefix,
    metrics,
):
    print(
        f"{prefix:<12s}"
        f"RMSE="
        f"{metrics['rmse']:.6f}"
        f" | MAE="
        f"{metrics['mae']:.6f}"
        f" | R2="
        f"{metrics['r2']:.4f}"
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_pair_splits(
        LATENT_DATA_PATH
    )

    train_raw = load_raw_sequences(
        LATENT_DATA_PATH,
        splits[
            "train"
        ],
    )

    val_raw = load_raw_sequences(
        LATENT_DATA_PATH,
        splits[
            "validation"
        ],
    )

    final_train_raw = (
        load_raw_sequences(
            LATENT_DATA_PATH,
            splits[
                "final_train"
            ],
        )
    )

    test_raw = load_raw_sequences(
        LATENT_DATA_PATH,
        splits[
            "test"
        ],
    )

    print()
    print(
        "=" * 100
    )
    print(
        "RGB CESN GRASP-CONTEXT V1"
    )
    print(
        "=" * 100
    )
    print(
        "CONTEXT TO INVERT = "
        "[TASK, GRASP_X, GRASP_Y]"
    )
    print(
        "FORWARD INPUT = "
        "[task_onehot(2), "
        "z_static_pca(16), "
        "grasp_xy(2)]"
    )
    print(
        "CONTEXT DIM =",
        CONTEXT_DIM,
    )
    print(
        "OUTPUT = z_dynamic RAW (128)"
    )
    print(
        "TIME INPUT = NONE"
    )
    print()
    print(
        f"Development train : "
        f"{len(splits['train'])} pairs / "
        f"{len(train_raw)} trajectories"
    )
    print(
        f"Validation        : "
        f"{len(splits['validation'])} pairs / "
        f"{len(val_raw)} trajectories"
    )
    print(
        f"Final train       : "
        f"{len(splits['final_train'])} pairs / "
        f"{len(final_train_raw)} trajectories"
    )
    print(
        f"Untouched test    : "
        f"{len(splits['test'])} pairs / "
        f"{len(test_raw)} trajectories"
    )
    print(
        "=" * 100
    )

    # ========================================================
    # Development model:
    # PCA fitted only on 60 training pairs.
    # ========================================================
    dev_pca = fit_static_pca(
        train_raw,
        STATIC_PCA_DIM,
    )

    train_sequences = (
        prepare_sequences(
            train_raw,
            dev_pca,
        )
    )

    val_sequences = (
        prepare_sequences(
            val_raw,
            dev_pca,
        )
    )

    development_model = (
        make_model(
            device
        )
    )

    fit_model(
        development_model,
        train_sequences,
    )

    train_metrics = evaluate(
        development_model,
        train_sequences,
    )

    val_metrics = evaluate(
        development_model,
        val_sequences,
    )

    print()
    print(
        "DEVELOPMENT MODEL"
    )
    print(
        "PCA16 explained variance = "
        f"{dev_pca['explained_ratio'] * 100:.4f}%"
    )

    print_metrics(
        "TRAIN",
        train_metrics,
    )

    print_metrics(
        "VAL",
        val_metrics,
    )

    # ========================================================
    # Final model:
    # PCA + CESN fitted on 70 final-training pairs.
    # Untouched 11-pair test evaluated once.
    # ========================================================
    final_pca = fit_static_pca(
        final_train_raw,
        STATIC_PCA_DIM,
    )

    final_train_sequences = (
        prepare_sequences(
            final_train_raw,
            final_pca,
        )
    )

    test_sequences = (
        prepare_sequences(
            test_raw,
            final_pca,
        )
    )

    final_model = make_model(
        device
    )

    fit_model(
        final_model,
        final_train_sequences,
    )

    final_train_metrics = (
        evaluate(
            final_model,
            final_train_sequences,
        )
    )

    test_metrics = evaluate(
        final_model,
        test_sequences,
    )

    print()
    print(
        "FINAL MODEL"
    )
    print(
        "PCA16 explained variance = "
        f"{final_pca['explained_ratio'] * 100:.4f}%"
    )

    print_metrics(
        "FINAL TRAIN",
        final_train_metrics,
    )

    print_metrics(
        "TEST",
        test_metrics,
    )

    checkpoint = {
        "architecture":
            ARCHITECTURE_TAG,

        "model_state":
            final_model.state_dict(),

        "pca_mean":
            torch.from_numpy(
                final_pca[
                    "mean"
                ]
            ),

        "pca_components":
            torch.from_numpy(
                final_pca[
                    "components"
                ]
            ),

        "pca_explained_ratio":
            final_pca[
                "explained_ratio"
            ],

        "static_pca_dim":
            STATIC_PCA_DIM,

        "context_definition":
            [
                "task_onehot_2",
                "z_static_pca16_known",
                "grasp_x_norm",
                "grasp_y_norm",
            ],

        "context_to_invert":
            [
                "task",
                "grasp_x",
                "grasp_y",
            ],

        "grasp_x_scale":
            GRASP_X_SCALE,

        "grasp_y_scale":
            GRASP_Y_SCALE,

        "safe_grasp_offset_x":
            SAFE_GRASP_OFFSET_X,

        "pain_grasp_offset_x":
            PAIN_GRASP_OFFSET_X,

        "train_pairs":
            sorted(
                splits[
                    "train"
                ]
            ),

        "validation_pairs":
            sorted(
                splits[
                    "validation"
                ]
            ),

        "final_train_pairs":
            sorted(
                splits[
                    "final_train"
                ]
            ),

        "test_pairs":
            sorted(
                splits[
                    "test"
                ]
            ),

        "development_train_metrics":
            train_metrics,

        "validation_metrics":
            val_metrics,

        "final_train_metrics":
            final_train_metrics,

        "test_metrics":
            test_metrics,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    summary = {
        "architecture":
            ARCHITECTURE_TAG,

        "context_to_invert":
            [
                "task",
                "grasp_x",
                "grasp_y",
            ],

        "forward_input":
            "[task_onehot(2), z_static_pca16, grasp_xy(2)]",

        "output":
            "z_dynamic(t), 128D",

        "explicit_time_input":
            False,

        "development_train_pairs":
            TRAIN_PAIRS,

        "validation_pairs":
            VAL_PAIRS,

        "final_train_pairs":
            FINAL_TRAIN_PAIRS,

        "test_pairs":
            TEST_PAIRS,

        "development_pca_explained_ratio":
            dev_pca[
                "explained_ratio"
            ],

        "final_pca_explained_ratio":
            final_pca[
                "explained_ratio"
            ],

        "development_train_metrics":
            train_metrics,

        "validation_metrics":
            val_metrics,

        "final_train_metrics":
            final_train_metrics,

        "test_metrics":
            test_metrics,

        "model_path":
            str(
                MODEL_PATH
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding=
            "utf-8",
    )

    print()
    print(
        "Saved model:"
    )
    print(
        MODEL_PATH.resolve()
    )
    print()
    print(
        "Saved summary:"
    )
    print(
        SUMMARY_PATH.resolve()
    )
    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
