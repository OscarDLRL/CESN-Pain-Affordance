from __future__ import annotations

import argparse
import csv
import json
import math

import h5py
import numpy as np
import torch

from CESN.core.cesn import (
    ContextualESNTorch,
)

from CESN.core.parameters_grasp_context import (
    LATENT_DATA_PATH,
    MODEL_PATH,
    OUT_DIR,
    STATIC_PCA_DIM,
    CONTEXT_DIM,
    KNIFE_X_VALUES,
    KNIFE_Y_VALUES,
    SAFE_GRASP_OFFSET_X,
    PAIN_GRASP_OFFSET_X,
    GRASP_X_SCALE,
    GRASP_Y_SCALE,
    GRASP_X_MIN,
    GRASP_X_MAX,
    GRASP_Y_MIN,
    GRASP_Y_MAX,
)

INVERSION_DIR = (
    OUT_DIR
    / "context_inversion"
)


def trajectory_number(
    name,
):
    return int(
        str(name).split(
            "_"
        )[-1]
    )


def decode_knife_xy(
    pair_id,
):
    object_id = int(
        pair_id
    ) // 9

    object_row, object_col = divmod(
        object_id,
        3,
    )

    return (
        float(
            KNIFE_X_VALUES[
                object_col
            ]
        ),
        float(
            KNIFE_Y_VALUES[
                object_row
            ]
        ),
    )


def true_grasp_xy(
    pair_id,
    pain_label,
):
    knife_x, knife_y = (
        decode_knife_xy(
            pair_id
        )
    )

    offset_x = (
        SAFE_GRASP_OFFSET_X
        if int(pain_label) == 0
        else PAIN_GRASP_OFFSET_X
    )

    return (
        float(
            knife_x
            + offset_x
        ),
        float(
            knife_y
        ),
    )


def load_test_sequences(
    test_pairs,
):
    test_pairs = set(
        int(x)
        for x
        in test_pairs
    )

    sequences = []

    with h5py.File(
        LATENT_DATA_PATH,
        "r",
    ) as h5:
        keys = sorted(
            [
                key
                for key
                in h5.keys()
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

            if pair_id not in test_pairs:
                continue

            pain_label = int(
                g.attrs[
                    "pain_label"
                ]
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

                    "true_grasp_x":
                        grasp_x,

                    "true_grasp_y":
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
    ).astype(
        np.float32
    )


def inverse_tanh_clipped(
    value,
):
    value = float(
        np.clip(
            value,
            -0.999,
            0.999,
        )
    )

    return (
        0.5
        * math.log(
            (1.0 + value)
            / (1.0 - value)
        )
    )


def differentiable_predict(
    model,
    context,
    T,
):
    """
    Frozen CESN recurrence with autograd enabled.
    Only grasp_x and grasp_y are optimized.
    """
    state = torch.zeros(
        model.reservoir_size,
        device=model.device,
        dtype=model.dtype,
    )

    outputs = []

    for _ in range(
        T
    ):
        pre = (
            model.Win
            @ context
            + model.W
            @ state
            + model.bias
        )

        candidate = torch.tanh(
            pre
        )

        state = (
            (
                1.0
                - model.leak_rate
            )
            * state
            + model.leak_rate
            * candidate
        )

        features = torch.cat(
            [
                torch.ones(
                    1,
                    device=model.device,
                    dtype=model.dtype,
                ),

                context,

                state,
            ],
            dim=0,
        )

        outputs.append(
            features
            @ model.Wout
        )

    return torch.stack(
        outputs,
        dim=0,
    )


def task_restart_points(
    candidate_label,
):
    """
    Nine spatial restarts corresponding to the nine
    physical knife locations for each discrete task hypothesis.
    """
    offset_x = (
        SAFE_GRASP_OFFSET_X
        if int(candidate_label) == 0
        else PAIN_GRASP_OFFSET_X
    )

    return [
        (
            float(
                knife_x
                + offset_x
            ),
            float(
                knife_y
            ),
        )
        for knife_y
        in KNIFE_Y_VALUES
        for knife_x
        in KNIFE_X_VALUES
    ]


def optimize_grasp_xy(
    model,
    observed_z,
    z_static_pca,
    candidate_label,
    max_iter,
):
    observed = torch.as_tensor(
        observed_z,
        device=model.device,
        dtype=model.dtype,
    )

    T = int(
        observed.shape[0]
    )

    task = torch.tensor(
        [1.0, 0.0]
        if int(candidate_label) == 0
        else [0.0, 1.0],
        device=model.device,
        dtype=model.dtype,
    )

    static = torch.as_tensor(
        z_static_pca,
        device=model.device,
        dtype=model.dtype,
    )

    restart_points = (
        task_restart_points(
            candidate_label
        )
    )

    results = []

    for restart_id, (
        x0,
        y0,
    ) in enumerate(
        restart_points
    ):
        x0_norm = (
            x0
            / GRASP_X_SCALE
        )

        y0_norm = (
            y0
            / GRASP_Y_SCALE
        )

        raw_xy = torch.nn.Parameter(
            torch.tensor(
                [
                    inverse_tanh_clipped(
                        x0_norm
                    ),

                    inverse_tanh_clipped(
                        y0_norm
                    ),
                ],
                device=model.device,
                dtype=model.dtype,
            )
        )

        optimizer = torch.optim.LBFGS(
            [
                raw_xy
            ],
            lr=0.8,
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn=
                "strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            grasp_norm = torch.tanh(
                raw_xy
            )

            context = torch.cat(
                [
                    task,
                    static,
                    grasp_norm,
                ],
                dim=0,
            )

            if (
                context.numel()
                != CONTEXT_DIM
            ):
                raise RuntimeError(
                    f"Expected context dim "
                    f"{CONTEXT_DIM}, "
                    f"got "
                    f"{context.numel()}"
                )

            pred = differentiable_predict(
                model,
                context,
                T,
            )

            loss = torch.mean(
                (
                    pred
                    - observed
                ) ** 2
            )

            loss.backward()

            return loss

        optimizer.step(
            closure
        )

        with torch.no_grad():
            grasp_norm = torch.tanh(
                raw_xy
            )

            context = torch.cat(
                [
                    task,
                    static,
                    grasp_norm,
                ],
                dim=0,
            )

            pred = differentiable_predict(
                model,
                context,
                T,
            )

            mse = float(
                torch.mean(
                    (
                        pred
                        - observed
                    ) ** 2
                ).item()
            )

            pred_x = float(
                grasp_norm[
                    0
                ].item()
                * GRASP_X_SCALE
            )

            pred_y = float(
                grasp_norm[
                    1
                ].item()
                * GRASP_Y_SCALE
            )

            pred_x = float(
                np.clip(
                    pred_x,
                    GRASP_X_MIN,
                    GRASP_X_MAX,
                )
            )

            pred_y = float(
                np.clip(
                    pred_y,
                    GRASP_Y_MIN,
                    GRASP_Y_MAX,
                )
            )

        results.append(
            {
                "restart_id":
                    int(
                        restart_id
                    ),

                "restart_x_m":
                    float(
                        x0
                    ),

                "restart_y_m":
                    float(
                        y0
                    ),

                "pred_x_m":
                    pred_x,

                "pred_y_m":
                    pred_y,

                "mse":
                    mse,
            }
        )

    best = min(
        results,
        key=lambda item:
            item[
                "mse"
            ],
    )

    return (
        best,
        results,
    )


def invert_one(
    model,
    item,
    pca_mean,
    pca_components,
    history_fraction,
    max_iter,
):
    T_full = int(
        item[
            "z_dynamic"
        ].shape[0]
    )

    n_observed = max(
        2,
        int(
            np.ceil(
                history_fraction
                * T_full
            )
        ),
    )

    n_observed = min(
        n_observed,
        T_full,
    )

    observed_z = item[
        "z_dynamic"
    ][
        :n_observed
    ]

    z_static_pca = transform_static(
        item[
            "z_static"
        ],
        pca_mean,
        pca_components,
    )

    safe_best, safe_restarts = (
        optimize_grasp_xy(
            model=model,
            observed_z=observed_z,
            z_static_pca=z_static_pca,
            candidate_label=0,
            max_iter=max_iter,
        )
    )

    pain_best, pain_restarts = (
        optimize_grasp_xy(
            model=model,
            observed_z=observed_z,
            z_static_pca=z_static_pca,
            candidate_label=1,
            max_iter=max_iter,
        )
    )

    if (
        safe_best[
            "mse"
        ]
        <= pain_best[
            "mse"
        ]
    ):
        predicted_label = 0
        winner = safe_best
    else:
        predicted_label = 1
        winner = pain_best

    dx = (
        winner[
            "pred_x_m"
        ]
        - item[
            "true_grasp_x"
        ]
    )

    dy = (
        winner[
            "pred_y_m"
        ]
        - item[
            "true_grasp_y"
        ]
    )

    grasp_error_mm = (
        math.sqrt(
            dx * dx
            + dy * dy
        )
        * 1000.0
    )

    return {
        "trajectory":
            item[
                "key"
            ],

        "pair_id":
            int(
                item[
                    "pair_id"
                ]
            ),

        "history_fraction":
            float(
                history_fraction
            ),

        "history_percent":
            float(
                history_fraction
                * 100.0
            ),

        "n_observed_frames":
            int(
                n_observed
            ),

        "total_frames":
            int(
                T_full
            ),

        "true_label":
            int(
                item[
                    "pain_label"
                ]
            ),

        "predicted_label":
            int(
                predicted_label
            ),

        "classification_correct":
            bool(
                predicted_label
                == item[
                    "pain_label"
                ]
            ),

        "true_grasp_x_m":
            float(
                item[
                    "true_grasp_x"
                ]
            ),

        "true_grasp_y_m":
            float(
                item[
                    "true_grasp_y"
                ]
            ),

        "pred_grasp_x_m":
            float(
                winner[
                    "pred_x_m"
                ]
            ),

        "pred_grasp_y_m":
            float(
                winner[
                    "pred_y_m"
                ]
            ),

        "grasp_error_mm":
            float(
                grasp_error_mm
            ),

        "best_safe_mse":
            float(
                safe_best[
                    "mse"
                ]
            ),

        "best_pain_mse":
            float(
                pain_best[
                    "mse"
                ]
            ),

        "winning_restart_id":
            int(
                winner[
                    "restart_id"
                ]
            ),

        "safe_restarts":
            safe_restarts,

        "pain_restarts":
            pain_restarts,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    if (
        args.history <= 0.0
        or args.history > 100.0
    ):
        raise ValueError(
            "--history must be in "
            "(0,100]."
        )

    history_fraction = (
        args.history
        / 100.0
    )

    INVERSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        map_location=
            "cpu",
    )

    model = (
        ContextualESNTorch
        .from_state_dict(
            checkpoint[
                "model_state"
            ],
            device=
                device,
        )
    )

    pca_mean = (
        checkpoint[
            "pca_mean"
        ]
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
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    if (
        pca_components.shape[0]
        != STATIC_PCA_DIM
    ):
        raise RuntimeError(
            "Unexpected PCA dimension."
        )

    test_sequences = (
        load_test_sequences(
            checkpoint[
                "test_pairs"
            ]
        )
    )

    if args.trajectory is not None:
        test_sequences = [
            item
            for item
            in test_sequences
            if item[
                "key"
            ]
            == args.trajectory
        ]

        if not test_sequences:
            raise RuntimeError(
                f"{args.trajectory} "
                f"is not in the untouched "
                f"test split."
            )

    print()
    print(
        "=" * 112
    )
    print(
        "CESN CONTEXT INVERSION: TASK + GRASP X,Y"
    )
    print(
        "=" * 112
    )
    print(
        f"Observed history = "
        f"{args.history:.1f}%"
    )
    print(
        "Known condition = "
        "z_static PCA16"
    )
    print(
        "Unknown context = "
        "TASK + GRASP_X + GRASP_Y"
    )
    print(
        "TASK = discrete hypotheses "
        "SAFE vs PAIN"
    )
    print(
        "GRASP XY = continuous "
        "L-BFGS optimization"
    )
    print(
        "Spatial restarts = "
        "9 per task hypothesis"
    )
    print(
        "=" * 112
    )

    results = []
    flat_results = []

    for index, item in enumerate(
        test_sequences,
        start=1,
    ):
        result = invert_one(
            model=model,
            item=item,
            pca_mean=pca_mean,
            pca_components=
                pca_components,
            history_fraction=
                history_fraction,
            max_iter=
                args.max_iter,
        )

        results.append(
            result
        )

        flat_results.append(
            {
                key: value
                for key, value
                in result.items()
                if key
                not in {
                    "safe_restarts",
                    "pain_restarts",
                }
            }
        )

        true_name = (
            "PAIN"
            if result[
                "true_label"
            ] == 1
            else "SAFE"
        )

        pred_name = (
            "PAIN"
            if result[
                "predicted_label"
            ] == 1
            else "SAFE"
        )

        print(
            f"[{index:02d}/"
            f"{len(test_sequences):02d}] "
            f"{result['trajectory']} | "
            f"true={true_name:<4s} "
            f"pred={pred_name:<4s} | "
            f"frames="
            f"{result['n_observed_frames']}"
            f"/"
            f"{result['total_frames']} | "
            f"trueGrasp=("
            f"{result['true_grasp_x_m']:+.4f},"
            f"{result['true_grasp_y_m']:+.4f}) | "
            f"predGrasp=("
            f"{result['pred_grasp_x_m']:+.4f},"
            f"{result['pred_grasp_y_m']:+.4f}) | "
            f"err="
            f"{result['grasp_error_mm']:.2f} mm"
        )

    accuracy = float(
        np.mean(
            [
                item[
                    "classification_correct"
                ]
                for item
                in results
            ]
        )
    )

    errors = np.asarray(
        [
            item[
                "grasp_error_mm"
            ]
            for item
            in results
        ],
        dtype=np.float64,
    )

    summary = {
        "history_percent":
            float(
                args.history
            ),

        "n_test_trajectories":
            len(
                results
            ),

        "task_accuracy":
            accuracy,

        "mean_grasp_error_mm":
            float(
                np.mean(
                    errors
                )
            ),

        "median_grasp_error_mm":
            float(
                np.median(
                    errors
                )
            ),

        "max_grasp_error_mm":
            float(
                np.max(
                    errors
                )
            ),

        "results":
            results,
    }

    suffix = (
        f"{int(round(args.history))}pct"
    )

    if args.trajectory:
        suffix += (
            "_"
            + args.trajectory
        )

    json_path = (
        INVERSION_DIR
        / f"inversion_{suffix}.json"
    )

    csv_path = (
        INVERSION_DIR
        / f"inversion_{suffix}.csv"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding=
            "utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                flat_results[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            flat_results
        )

    print()
    print(
        "-" * 112
    )
    print(
        f"TASK accuracy       = "
        f"{accuracy * 100.0:.2f}%"
    )
    print(
        f"Mean grasp XY error = "
        f"{np.mean(errors):.2f} mm"
    )
    print(
        f"Median grasp error  = "
        f"{np.median(errors):.2f} mm"
    )
    print(
        f"Max grasp error     = "
        f"{np.max(errors):.2f} mm"
    )
    print()
    print(
        "JSON:"
    )
    print(
        json_path.resolve()
    )
    print()
    print(
        "CSV:"
    )
    print(
        csv_path.resolve()
    )
    print(
        "=" * 112
    )


if __name__ == "__main__":
    main()
