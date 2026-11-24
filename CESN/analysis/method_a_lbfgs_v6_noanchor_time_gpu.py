from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from CESN.core.cesn import ContextualESNTorch
from CESN.core.parameters_grasp_context_v6 import (
    LATENT_DATA_PATH,
    RGB_DATA_PATH,
    OUT_DIR,
    MODEL_PATH,
    STATIC_PCA_DIM,
    GRASP_X_SCALE,
    GRASP_Y_SCALE,
    SAFE_TARGET_LOCAL_X,
    PAIN_TARGET_LOCAL_X,
    TIME_INDEX_SCALE,
)


RESULT_DIR = (
    OUT_DIR
    / "joint_task_xy_v6"
)

HISTORIES = [
    5, 10, 15, 20, 25, 30,
    40, 50, 60, 70, 80, 90, 100,
]

QUICK_HISTORIES = [
    5, 10, 15, 20, 40, 70, 100,
]

SUMMARY_CSV = (
    RESULT_DIR
    / "joint_task_xy_v6_summary.csv"
)

SUMMARY_JSON = (
    RESULT_DIR
    / "joint_task_xy_v6_summary.json"
)

PLOT_PATH = (
    RESULT_DIR
    / "joint_task_xy_v6_vs_history.png"
)


def atanh_safe(
    x,
):
    x = torch.clamp(
        x,
        -0.999,
        0.999,
    )

    return 0.5 * torch.log(
        (1.0 + x)
        / (1.0 - x)
    )


def transform_static(
    z_static,
    mean,
    components,
):
    return (
        (
            z_static
            - mean
        )
        @ components.T
    ).astype(
        np.float32
    )


def task_restart_points(
    label,
):
    knife_xs = [
        -0.08,
        0.00,
        +0.08,
    ]

    knife_ys = [
        +0.08,
        0.00,
        -0.08,
    ]

    offset = (
        SAFE_TARGET_LOCAL_X
        if int(label) == 0
        else PAIN_TARGET_LOCAL_X
    )

    pts = []

    for y in knife_ys:
        for x in knife_xs:
            pts.append(
                (
                    x + offset,
                    y,
                )
            )

    return pts


def load_test_sequences(
    test_pairs,
):
    test_pairs = set(
        int(x)
        for x in test_pairs
    )

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

            if pair_id not in test_pairs:
                continue

            label = int(
                lg.attrs[
                    "pain_label"
                ]
            )

            obj_xyz = np.asarray(
                rgb_h5[
                    key
                ][
                    "object_xyz"
                ][
                    0
                ],
                dtype=np.float32,
            )

            knife_x = float(
                obj_xyz[
                    0
                ]
            )

            knife_y = float(
                obj_xyz[
                    1
                ]
            )

            true_x = (
                knife_x
                + (
                    SAFE_TARGET_LOCAL_X
                    if label == 0
                    else PAIN_TARGET_LOCAL_X
                )
            )

            true_y = (
                knife_y
            )

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

                    "true_grasp_x":
                        float(
                            true_x
                        ),

                    "true_grasp_y":
                        float(
                            true_y
                        ),
                }
            )

    return rows



# ============================================================
# FINAL V6 NO-ANCHOR + CAUSAL TIME EVALUATION
# ============================================================

RESULT_DIR = (
    OUT_DIR
    / "method_a_lbfgs"
)

SUMMARY_CSV = (
    RESULT_DIR
    / "method_a_lbfgs_summary.csv"
)

SUMMARY_JSON = (
    RESULT_DIR
    / "method_a_lbfgs_summary.json"
)

PLOT_TASK = (
    RESULT_DIR
    / "method_a_task_accuracy_vs_history.png"
)

PLOT_XY = (
    RESULT_DIR
    / "method_a_xy_error_vs_history.png"
)

HISTORIES = [
    5, 10, 15, 20, 25, 30,
    40, 50, 60, 70, 80, 90, 100,
]


def build_base_context_batch(
    label,
    z_static_pca,
    grasp_x,
    grasp_y,
    model,
):
    """
    Base context WITHOUT time:
      [task onehot(2), static PCA16, grasp XY(2)]
    -> 20 dimensions.

    Time is appended causally inside predict_batch().
    """
    B = int(
        grasp_x.shape[
            0
        ]
    )

    task_one = torch.tensor(
        [1.0, 0.0]
        if int(label) == 0
        else [0.0, 1.0],
        device=model.device,
        dtype=model.dtype,
    )

    task = task_one[
        None,
        :
    ].expand(
        B,
        -1,
    )

    static = torch.as_tensor(
        z_static_pca,
        device=model.device,
        dtype=model.dtype,
    )[
        None,
        :
    ].expand(
        B,
        -1,
    )

    grasp = torch.stack(
        [
            grasp_x
            / GRASP_X_SCALE,

            grasp_y
            / GRASP_Y_SCALE,
        ],
        dim=1,
    )

    base = torch.cat(
        [
            task,
            static,
            grasp,
        ],
        dim=1,
    )

    expected = (
        model.input_dim
        - 1
    )

    if int(
        base.shape[
            1
        ]
    ) != int(
        expected
    ):
        raise RuntimeError(
            f"Expected base context dim {expected}, "
            f"got {base.shape[1]}"
        )

    return base


def predict_batch(
    model,
    base_contexts,
    T,
):
    """
    Causal CESN rollout with explicit time index.

    At step t:
      time_index = t / TIME_INDEX_SCALE

    This never uses the unknown final trajectory length.
    """
    B = int(
        base_contexts.shape[
            0
        ]
    )

    state = torch.zeros(
        B,
        model.reservoir_size,
        device=model.device,
        dtype=model.dtype,
    )

    outputs = []

    for t in range(
        T
    ):
        time_col = torch.full(
            (
                B,
                1,
            ),
            float(
                t
            )
            / float(
                TIME_INDEX_SCALE
            ),
            device=model.device,
            dtype=model.dtype,
        )

        contexts = torch.cat(
            [
                base_contexts,
                time_col,
            ],
            dim=1,
        )

        pre = (
            contexts
            @ model.Win.T
            +
            state
            @ model.W.T
            +
            model.bias
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
            +
            model.leak_rate
            * candidate
        )

        ones = torch.ones(
            B,
            1,
            device=model.device,
            dtype=model.dtype,
        )

        features = torch.cat(
            [
                ones,
                contexts,
                state,
            ],
            dim=1,
        )

        y = (
            features
            @ model.Wout
        )

        outputs.append(
            y
        )

    return torch.stack(
        outputs,
        dim=1,
    )


def optimize_xy_batched(
    model,
    observed,
    z_static_pca,
    label,
    lbfgs_max_iter,
    lbfgs_history_size,
):
    """
    METHOD A - L-BFGS inversion.

    For ONE discrete task hypothesis (SAFE or PAIN):
      1) initialize 9 physically meaningful XY restart points
      2) optimize XY with L-BFGS
      3) keep the restart with the lowest reconstruction loss

    SAFE and PAIN are optimized independently by the caller.
    The final task hypothesis is selected by the lower minimum loss.

    Bounds are enforced through:
        x = tanh(raw_x) * GRASP_X_SCALE
        y = tanh(raw_y) * GRASP_Y_SCALE
    """
    obs = torch.as_tensor(
        observed,
        device=model.device,
        dtype=model.dtype,
    )

    T = int(
        obs.shape[
            0
        ]
    )

    starts = torch.tensor(
        task_restart_points(
            label
        ),
        device=model.device,
        dtype=model.dtype,
    )

    nx = (
        starts[
            :,
            0
        ]
        / GRASP_X_SCALE
    )

    ny = (
        starts[
            :,
            1
        ]
        / GRASP_Y_SCALE
    )

    raw = torch.stack(
        [
            atanh_safe(
                nx
            ),
            atanh_safe(
                ny
            ),
        ],
        dim=1,
    ).detach().clone()

    raw.requires_grad_(
        True
    )

    optimizer = torch.optim.LBFGS(
        [
            raw
        ],
        lr=1.0,
        max_iter=int(
            lbfgs_max_iter
        ),
        max_eval=int(
            lbfgs_max_iter
            * 2
        ),
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=int(
            lbfgs_history_size
        ),
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(
            set_to_none=True
        )

        grasp_x = (
            torch.tanh(
                raw[
                    :,
                    0
                ]
            )
            * GRASP_X_SCALE
        )

        grasp_y = (
            torch.tanh(
                raw[
                    :,
                    1
                ]
            )
            * GRASP_Y_SCALE
        )

        base_context = build_base_context_batch(
            label,
            z_static_pca,
            grasp_x,
            grasp_y,
            model,
        )

        pred = predict_batch(
            model,
            base_context,
            T,
        )

        losses = torch.mean(
            (
                pred
                - obs[
                    None,
                    :,
                    :
                ]
            ) ** 2,
            dim=(
                1,
                2,
            ),
        )

        # The nine restart candidates remain independent in the
        # reconstruction calculation. Summation gives one scalar
        # objective required by torch.optim.LBFGS.
        objective = losses.sum()

        objective.backward()

        return objective

    optimizer.step(
        closure
    )

    with torch.no_grad():
        grasp_x = (
            torch.tanh(
                raw[
                    :,
                    0
                ]
            )
            * GRASP_X_SCALE
        )

        grasp_y = (
            torch.tanh(
                raw[
                    :,
                    1
                ]
            )
            * GRASP_Y_SCALE
        )

        base_context = build_base_context_batch(
            label,
            z_static_pca,
            grasp_x,
            grasp_y,
            model,
        )

        pred = predict_batch(
            model,
            base_context,
            T,
        )

        losses = torch.mean(
            (
                pred
                - obs[
                    None,
                    :,
                    :
                ]
            ) ** 2,
            dim=(
                1,
                2,
            ),
        )

        idx = int(
            torch.argmin(
                losses
            ).item()
        )

        return (
            float(
                grasp_x[
                    idx
                ].item()
            ),
            float(
                grasp_y[
                    idx
                ].item()
            ),
            float(
                losses[
                    idx
                ].item()
            ),
        )

def fixed_xy_task_losses(
    model,
    observed,
    z_static_pca,
    true_x,
    true_y,
):
    """
    TASK discrimination with TRUE XY fixed.
    No continuous inversion here.
    """
    obs = torch.as_tensor(
        observed,
        device=model.device,
        dtype=model.dtype,
    )

    T = int(
        obs.shape[
            0
        ]
    )

    gx = torch.tensor(
        [
            true_x
        ],
        device=model.device,
        dtype=model.dtype,
    )

    gy = torch.tensor(
        [
            true_y
        ],
        device=model.device,
        dtype=model.dtype,
    )

    out = {}

    with torch.no_grad():
        for label in [
            0,
            1,
        ]:
            base_context = (
                build_base_context_batch(
                    label,
                    z_static_pca,
                    gx,
                    gy,
                    model,
                )
            )

            pred = predict_batch(
                model,
                base_context,
                T,
            )[
                0
            ]

            loss = torch.mean(
                (
                    pred
                    - obs
                ) ** 2
            )

            out[
                label
            ] = float(
                loss.item()
            )

    return (
        out[
            0
        ],
        out[
            1
        ],
    )


def xy_error_mm(
    pred_x,
    pred_y,
    true_x,
    true_y,
):
    return (
        math.sqrt(
            (
                pred_x
                - true_x
            ) ** 2
            +
            (
                pred_y
                - true_y
            ) ** 2
        )
        * 1000.0
    )


def summarize_errors(
    errors,
):
    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                np.mean(
                    errors
                )
            ),

        "median":
            float(
                np.median(
                    errors
                )
            ),

        "p90":
            float(
                np.percentile(
                    errors,
                    90,
                )
            ),

        "within_5":
            float(
                np.mean(
                    errors
                    <= 5.0
                )
            ),

        "within_10":
            float(
                np.mean(
                    errors
                    <= 10.0
                )
            ),

        "within_20":
            float(
                np.mean(
                    errors
                    <= 20.0
                )
            ),
    }


def evaluate_history(
    pct,
    lbfgs_max_iter,
    lbfgs_history_size,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model = ContextualESNTorch.from_state_dict(
        checkpoint[
            "model_state"
        ],
        device=device,
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

    sequences = load_test_sequences(
        checkpoint[
            "test_pairs"
        ]
    )

    known_xy_errors = []
    joint_xy_errors = []

    joint_task_correct = []
    true_xy_task_correct = []

    safe_joint_correct = []
    pain_joint_correct = []

    safe_truexy_correct = []
    pain_truexy_correct = []

    for item in sequences:
        T_full = int(
            item[
                "z_dynamic"
            ].shape[
                0
            ]
        )

        n_obs = max(
            2,
            int(
                np.ceil(
                    T_full
                    * pct
                    / 100.0
                )
            ),
        )

        n_obs = min(
            n_obs,
            T_full,
        )

        observed = item[
            "z_dynamic"
        ][
            :n_obs
        ]

        z_static_pca = transform_static(
            item[
                "z_static"
            ],
            pca_mean,
            pca_components,
        )

        true_label = int(
            item[
                "pain_label"
            ]
        )

        true_x = float(
            item[
                "true_grasp_x"
            ]
        )

        true_y = float(
            item[
                "true_grasp_y"
            ]
        )

        # ----------------------------------------------------
        # TEST 1: KNOWN TASK -> UNKNOWN XY
        # ----------------------------------------------------
        known_x, known_y, _ = (
            optimize_xy_batched(
                model,
                observed,
                z_static_pca,
                true_label,
                lbfgs_max_iter,
                lbfgs_history_size,
            )
        )

        known_xy_errors.append(
            xy_error_mm(
                known_x,
                known_y,
                true_x,
                true_y,
            )
        )

        # ----------------------------------------------------
        # TEST 2: UNKNOWN TASK + UNKNOWN XY
        # ----------------------------------------------------
        safe_x, safe_y, safe_loss = (
            optimize_xy_batched(
                model,
                observed,
                z_static_pca,
                0,
                lbfgs_max_iter,
                lbfgs_history_size,
            )
        )

        pain_x, pain_y, pain_loss = (
            optimize_xy_batched(
                model,
                observed,
                z_static_pca,
                1,
                lbfgs_max_iter,
                lbfgs_history_size,
            )
        )

        if safe_loss <= pain_loss:
            joint_label = 0
            joint_x = safe_x
            joint_y = safe_y
        else:
            joint_label = 1
            joint_x = pain_x
            joint_y = pain_y

        joint_ok = int(
            joint_label
            == true_label
        )

        joint_task_correct.append(
            joint_ok
        )

        if true_label == 0:
            safe_joint_correct.append(
                joint_ok
            )
        else:
            pain_joint_correct.append(
                joint_ok
            )

        joint_xy_errors.append(
            xy_error_mm(
                joint_x,
                joint_y,
                true_x,
                true_y,
            )
        )

        # ----------------------------------------------------
        # TEST 3: UNKNOWN TASK + TRUE XY FIXED
        # ----------------------------------------------------
        fixed_safe_loss, fixed_pain_loss = (
            fixed_xy_task_losses(
                model,
                observed,
                z_static_pca,
                true_x,
                true_y,
            )
        )

        fixed_label = (
            0
            if fixed_safe_loss
            <= fixed_pain_loss
            else 1
        )

        fixed_ok = int(
            fixed_label
            == true_label
        )

        true_xy_task_correct.append(
            fixed_ok
        )

        if true_label == 0:
            safe_truexy_correct.append(
                fixed_ok
            )
        else:
            pain_truexy_correct.append(
                fixed_ok
            )

    known = summarize_errors(
        known_xy_errors
    )

    joint = summarize_errors(
        joint_xy_errors
    )

    return {
        "history_percent":
            int(
                pct
            ),

        # TEST 1
        "known_task_xy_mean_mm":
            known[
                "mean"
            ],

        "known_task_xy_median_mm":
            known[
                "median"
            ],

        "known_task_xy_p90_mm":
            known[
                "p90"
            ],

        "known_task_xy_within_10":
            known[
                "within_10"
            ],

        "known_task_xy_within_20":
            known[
                "within_20"
            ],

        # TEST 2
        "joint_task_accuracy":
            float(
                np.mean(
                    joint_task_correct
                )
            ),

        "joint_safe_accuracy":
            float(
                np.mean(
                    safe_joint_correct
                )
            ),

        "joint_pain_accuracy":
            float(
                np.mean(
                    pain_joint_correct
                )
            ),

        "joint_xy_mean_mm":
            joint[
                "mean"
            ],

        "joint_xy_median_mm":
            joint[
                "median"
            ],

        "joint_xy_p90_mm":
            joint[
                "p90"
            ],

        "joint_xy_within_10":
            joint[
                "within_10"
            ],

        "joint_xy_within_20":
            joint[
                "within_20"
            ],

        # TEST 3
        "true_xy_task_accuracy":
            float(
                np.mean(
                    true_xy_task_correct
                )
            ),

        "true_xy_safe_accuracy":
            float(
                np.mean(
                    safe_truexy_correct
                )
            ),

        "true_xy_pain_accuracy":
            float(
                np.mean(
                    pain_truexy_correct
                )
            ),
    }


def worker(
    job,
):
    return evaluate_history(
        *job
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=80,
    )

    ap.add_argument(
        "--lbfgs-history-size",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 144
    )
    print(
        "METHOD A - L-BFGS | V6 NO-ANCHOR + CAUSAL TIME INDEX"
    )
    print(
        "=" * 144
    )
    print(
        "METHOD A:"
    )
    print(
        "SAFE hypothesis -> independently optimize X,Y with L-BFGS"
    )
    print(
        "PAIN hypothesis -> independently optimize X,Y with L-BFGS"
    )
    print(
        "Select hypothesis with the lowest reconstruction loss"
    )
    print(
        "Also reports known-TASK XY and TRUE-XY TASK diagnostics"
    )
    print(
        f"TIME INDEX: frame_index / {TIME_INDEX_SCALE:.0f} (causal)"
    )
    print(
        "Histories:",
        HISTORIES,
    )
    print(
        "=" * 144
    )

    jobs = [
        (
            pct,
            args.lbfgs_max_iter,
            args.lbfgs_history_size,
        )
        for pct in HISTORIES
    ]

    rows = []

    with ProcessPoolExecutor(
        max_workers=
            args.workers
    ) as executor:

        futures = {
            executor.submit(
                worker,
                job,
            ):
                job
            for job in jobs
        }

        for future in as_completed(
            futures
        ):
            row = future.result()

            rows.append(
                row
            )

            print(
                f"[DONE] "
                f"{row['history_percent']:>3d}%"
                f" | K-TASK XYmed="
                f"{row['known_task_xy_median_mm']:>7.2f}mm"
                f" P90="
                f"{row['known_task_xy_p90_mm']:>7.2f}"
                f" | JOINT TASK="
                f"{row['joint_task_accuracy']*100:>6.2f}%"
                f" XYmed="
                f"{row['joint_xy_median_mm']:>7.2f}mm"
                f" | TRUE-XY TASK="
                f"{row['true_xy_task_accuracy']*100:>6.2f}%"
            )

    rows.sort(
        key=lambda r:
            r[
                "history_percent"
            ]
    )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    SUMMARY_JSON.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    x = [
        r[
            "history_percent"
        ]
        for r in rows
    ]

    plt.figure(
        figsize=(
            9,
            5.5,
        )
    )

    plt.plot(
        x,
        [
            100.0
            * r[
                "joint_task_accuracy"
            ]
            for r in rows
        ],
        marker="o",
        label="Unknown TASK + unknown XY",
    )

    plt.plot(
        x,
        [
            100.0
            * r[
                "true_xy_task_accuracy"
            ]
            for r in rows
        ],
        marker="o",
        label="Unknown TASK + true XY",
    )

    plt.xlabel(
        "Observed trajectory (%)"
    )

    plt.ylabel(
        "TASK accuracy (%)"
    )

    plt.ylim(
        0,
        105,
    )

    plt.title(
        "Method A L-BFGS: TASK Accuracy"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_TASK,
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(
            9,
            5.5,
        )
    )

    plt.plot(
        x,
        [
            r[
                "known_task_xy_median_mm"
            ]
            for r in rows
        ],
        marker="o",
        label="Known TASK: median XY",
    )

    plt.plot(
        x,
        [
            r[
                "joint_xy_median_mm"
            ]
            for r in rows
        ],
        marker="o",
        label="Unknown TASK: median XY",
    )

    plt.xlabel(
        "Observed trajectory (%)"
    )

    plt.ylabel(
        "XY error (mm)"
    )

    plt.title(
        "Method A L-BFGS: XY Inversion"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_XY,
        dpi=200,
    )

    plt.close()

    print()
    print(
        "=" * 144
    )
    print(
        "Saved:",
        SUMMARY_CSV.resolve(),
    )
    print(
        "Saved:",
        SUMMARY_JSON.resolve(),
    )
    print(
        "=" * 144
    )


if __name__ == "__main__":
    main()
