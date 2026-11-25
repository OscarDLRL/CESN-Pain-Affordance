from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from CLEAVER.core.cesn import ContextualESNTorch
from CLEAVER.core.parameters_explicit import (
    LATENT_DATA_PATH,
    RGB_DATA_PATH,
    OUT_DIR,
    MODEL_PATH,
    STATIC_PCA_DIM,
    RESERVOIR_DIM,
    RIDGE,
    SPECTRAL_RADIUS,
    LEAK_RATE,
    INPUT_SCALE,
    BIAS_SCALE,
    SEED,
    GRASP_X_SCALE,
    GRASP_Y_SCALE,
    TRAIN_PAIRS,
    VAL_PAIRS,
    SAFE_TARGET_LOCAL_X,
    PAIN_TARGET_LOCAL_X,
    TIME_INDEX_SCALE,
)


OUTPUT_DIM = 200
BASE_CONTEXT_DIM = (
    2
    + STATIC_PCA_DIM
    + 2
)

CONTEXT_DIM = (
    BASE_CONTEXT_DIM
    + 1  # causal time index
)


def build_pair_splits():
    from CLEAVER.config import TRAIN, TEST
    return {"train": TRAIN, "validation": set(), "test": TEST}


def load_raw_sequences(
    pair_ids,
):
    rows = []

    with h5py.File(
        LATENT_DATA_PATH,
        "r",
    ) as latent_h5, h5py.File(
        RGB_DATA_PATH,
        "r",
    ) as rgb_h5:

        for key in sorted(
            latent_h5.keys()
        ):
            if not str(
                key
            ).startswith(
                "trajectory_"
            ):
                continue

            lg = latent_h5[
                key
            ]

            pair_id = int(
                lg.attrs[
                    "pair_id"
                ]
            )

            if pair_id not in pair_ids:
                continue

            rg = rgb_h5[
                key
            ]

            label = int(
                lg.attrs[
                    "pain_label"
                ]
            )

            object_xyz = np.asarray(
                rg[
                    "object_xyz"
                ][
                    0
                ],
                dtype=np.float32,
            )

            knife_x = float(
                object_xyz[
                    0
                ]
            )

            knife_y = float(
                object_xyz[
                    1
                ]
            )

            grasp_x, grasp_y = map(float, rg["target_world"][:2])

            rows.append(
                {
                    "key":
                        str(
                            key
                        ),

                    "pair_id":
                        pair_id,

                    "pain_label":
                        label,

                    "z_static":
                        np.asarray(
                            lg[
                                "z_static"
                            ],
                            dtype=np.float32,
                        ),

                    "z_dynamic":
                        np.asarray(
                            lg[
                                "z_dynamic"
                            ],
                            dtype=np.float32,
                        ),

                    "grasp_x":
                        float(
                            grasp_x
                        ),

                    "grasp_y":
                        float(
                            grasp_y
                        ),
                }
            )

    return rows


def fit_static_pca(
    raw_sequences,
):
    X = np.stack(
        [
            item[
                "z_static"
            ]
            for item in raw_sequences
        ],
        axis=0,
    ).astype(
        np.float64
    )

    mean = X.mean(
        axis=0
    )

    centered = (
        X
        - mean[
            None,
            :
        ]
    )

    _, S, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = Vt[
        :STATIC_PCA_DIM
    ].astype(
        np.float32
    )

    total_var = float(
        np.sum(
            S ** 2
        )
    )

    explained = float(
        np.sum(
            S[
                :STATIC_PCA_DIM
            ] ** 2
        )
        / max(
            total_var,
            1e-12,
        )
    )

    return {
        "mean":
            mean.astype(
                np.float32
            ),

        "components":
            components,

        "explained_ratio":
            explained,
    }


def transform_static(
    z_static,
    pca,
):
    return (
        (
            z_static
            - pca[
                "mean"
            ]
        )
        @ pca[
            "components"
        ].T
    ).astype(
        np.float32
    )


def build_context(
    item,
    pca,
):
    label = int(
        item[
            "pain_label"
        ]
    )

    task = np.asarray(
        [
            1.0,
            0.0,
        ]
        if label == 0
        else [
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )

    static = transform_static(
        item[
            "z_static"
        ],
        pca,
    )

    grasp = np.asarray(
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
            task,
            static,
            grasp,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    if context.shape[
        0
    ] != BASE_CONTEXT_DIM:
        raise RuntimeError(
            f"Expected base context dim {BASE_CONTEXT_DIM}, "
            f"got {context.shape[0]}"
        )

    return context


def prepare_sequences(
    raw_sequences,
    pca,
):
    out = []

    for item in raw_sequences:
        context = build_context(
            item,
            pca,
        )

        T = int(
            item[
                "z_dynamic"
            ].shape[
                0
            ]
        )

        base = np.repeat(
            context[
                None,
                :
            ],
            T,
            axis=0,
        ).astype(
            np.float32
        )

        # Causal online-compatible time index:
        # frame 0 -> 0.00, frame 5 -> 0.05, frame 50 -> 0.50, ...
        # No knowledge of the final trajectory length is required.
        time_index = (
            np.arange(
                T,
                dtype=np.float32,
            )
            / float(
                TIME_INDEX_SCALE
            )
        )[
            :,
            None
        ]

        U = np.concatenate(
            [
                base,
                time_index,
            ],
            axis=1,
        ).astype(
            np.float32
        )

        out.append(
            {
                **item,

                "U":
                    U,

                "Y":
                    item[
                        "z_dynamic"
                    ].astype(
                        np.float32
                    ),
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
            RESERVOIR_DIM,

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
            SEED,

        device=
            device,
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
        f"RMSE={metrics['rmse']:.6f}"
        f" | MAE={metrics['mae']:.6f}"
        f" | R2={metrics['r2']:.4f}"
    )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("CUDA was not detected.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = build_pair_splits()

    train_raw = load_raw_sequences(splits["train"])
    test_raw = load_raw_sequences(splits["test"])

    if len(splits["train"]) != 70:
        raise RuntimeError(
            f"Expected 70 final training pairs, got {len(splits['train'])}"
        )
    if len(splits["test"]) != 11:
        raise RuntimeError(
            f"Expected 11 test pairs, got {len(splits['test'])}"
        )

    print()
    print("=" * 100)
    print("V6 GRASP-CONTEXT CESN NATIVE256 LATENT200 RIDGE05 FINAL70")
    print("=" * 100)
    print("INPUT  = [task onehot(2), static PCA16, grasp XY(2), causal time index(1)]")
    print("OUTPUT = z_dynamic V6 native256 latent200 final70 (200)")
    print("FINAL TRAIN PAIRS = 70")
    print("TEST PAIRS        = 11")
    print("TIME INPUT        = frame_index / 100 (causal)")
    print("CONTEXT DIM       =", CONTEXT_DIM)
    print("RESERVOIR         =", RESERVOIR_DIM)
    print("RIDGE             =", RIDGE)
    print("DEVICE            =", device)
    print("=" * 100)

    pca = fit_static_pca(train_raw)
    train_seq = prepare_sequences(train_raw, pca)
    test_seq = prepare_sequences(test_raw, pca)

    model = make_model(device)
    fit_model(model, train_seq)

    train_metrics = evaluate(model, train_seq)
    test_metrics = evaluate(model, test_seq)

    print()
    print("FINAL MODEL")
    print(
        "PCA16 explained variance = "
        f"{pca['explained_ratio']*100:.4f}%"
    )
    print_metrics("TRAIN70", train_metrics)
    print_metrics("TEST11", test_metrics)

    torch.save(
        {
            "model_state": model.state_dict(),
            "pca_mean": torch.from_numpy(pca["mean"]),
            "pca_components": torch.from_numpy(pca["components"]),
            "pca_explained_ratio": float(pca["explained_ratio"]),
            "static_pca_dim": STATIC_PCA_DIM,
            "architecture": "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70",
            "latent_dataset": str(LATENT_DATA_PATH),
            "train_pairs": sorted(list(splits["train"])),
            "val_pairs": [],
            "test_pairs": sorted(list(splits["test"])),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
        MODEL_PATH,
    )

    summary = {
        "architecture":
            "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70",
        "context":
            "[task_onehot(2), static_pca16, grasp_xy(2), time_index(1)]",
        "output": "z_dynamic_v6_native256_latent200_final70(200)",
        "time_input": True,
        "time_index_scale": TIME_INDEX_SCALE,
        "reservoir_size": RESERVOIR_DIM,
        "ridge": RIDGE,
        "final_train_pairs": 70,
        "test_pairs": 11,
        "train": train_metrics,
        "test": test_metrics,
        "pca16_explained": float(pca["explained_ratio"]),
    }

    (OUT_DIR / "training_summary_final70.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("Saved model:", MODEL_PATH.resolve())
    print("=" * 100)


from CLEAVER.config import RGB, LATENT, EXPLICIT_DIR
RGB_DATA_PATH = RGB
LATENT_DATA_PATH = LATENT
OUT_DIR = EXPLICIT_DIR
MODEL_PATH = OUT_DIR / "cesn.pt"
TRAIN_PAIRS = 70
VAL_PAIRS = 0

if __name__ == "__main__":
    main()
