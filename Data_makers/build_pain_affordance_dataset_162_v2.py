#!/usr/bin/env python3
"""
Final Pain Affordance dataset generator.

Grid:
    9 knife positions  (3x3)
    9 Panda start poses (3x3)
    2 variants:
        SAFE   -> grasp handle
        HAZARD -> grasp blade

Total:
    9 x 9 x 2 = 162 trajectories

Design:
    - knife yaw fixed
    - same object pose and same prepared Panda start inside each SAFE/HAZARD pair
    - same camera
    - same grasp orientation
    - preparation motion is NOT recorded
    - primary observation is RGB 256x256
    - qpos / tcp / object_xyz / actions are metadata/debug only
    - RRTConnect for horizontal motion and descent
    - screw motion for lift

Outputs:
    outputs/pain_affordance_final_162/
        pain_affordance_162.h5
        manifest.csv
        summary.json
        00_initial_states_9x9.png

Run from project root:
    python -m Data_makers.build_pain_affordance_dataset_162
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch
from PIL import Image, ImageDraw

from mani_skill.utils.structs.pose import Pose

import Data_makers.test_knife_paired_reaches as base


OUT_DIR = Path(
    "outputs/pain_affordance_final_162"
)

H5_PATH = (
    OUT_DIR
    / "pain_affordance_162.h5"
)

MANIFEST_PATH = (
    OUT_DIR
    / "manifest.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "summary.json"
)


SCENE_SEED = 123


# ============================================================
# Object grid
# ============================================================

OBJECT_X = (
    -0.08,
    0.00,
    +0.08,
)

OBJECT_Y = (
    +0.08,
    0.00,
    -0.08,
)


# ============================================================
# Panda prepared-start grid
# ============================================================

START_DX = (
    -0.06,
    0.00,
    +0.06,
)

START_DY = (
    +0.06,
    0.00,
    -0.06,
)


# ============================================================
# Affordances
# ============================================================

SAFE_TARGET_LOCAL = (
    base.SAFE_TARGET_LOCAL.copy()
)

HAZARD_TARGET_LOCAL = (
    base.HAZARD_TARGET_LOCAL.copy()
)


VARIANTS = (
    (
        "safe",
        0,
        SAFE_TARGET_LOCAL,
    ),
    (
        "hazard",
        1,
        HAZARD_TARGET_LOCAL,
    ),
)


# ============================================================
# Helpers
# ============================================================

def restore_exact_knife_pose(env_unwrapped, knife_xy):
    """
    Re-apply the exact commanded knife pose immediately before frame 0.

    Why:
        The unrecorded Panda preparation advances physics. In a few
        combinations the knife settled by a few micrometers differently
        between SAFE and HAZARD, which was enough to trip the strict
        paired-state check.

    This does NOT encode the class. SAFE and HAZARD receive the exact
    same object pose for the same pair.
    """
    yaw = np.deg2rad(
        float(base.KNIFE_YAW_DEG)
    )

    p = torch.tensor(
        [[
            float(knife_xy[0]),
            float(knife_xy[1]),
            float(env_unwrapped.object_support_z),
        ]],
        dtype=torch.float32,
        device=env_unwrapped.device,
    )

    q = torch.tensor(
        [[
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]],
        dtype=torch.float32,
        device=env_unwrapped.device,
    )

    env_unwrapped.obj.set_pose(
        Pose.create_from_pq(
            p=p,
            q=q,
        )
    )


def run_trajectory(
    env,
    *,
    knife_xy,
    start_offset_xy,
    variant,
    label,
    local_target,
    scene_seed,
):
    """
    Generate one trajectory.

    Preparation is not recorded.

    Recorded:
        1. horizontal movement toward affordance
        2. descent
        3. gripper close
        4. lift
    """

    env.reset(
        seed=int(scene_seed),
        options={
            "knife_xy": [
                float(
                    knife_xy[0]
                ),
                float(
                    knife_xy[1]
                ),
            ],
            "knife_yaw_deg":
                base.KNIFE_YAW_DEG,
        },
    )

    planner = base.new_planner(
        env,
        vis=False,
    )

    e = env.unwrapped

    try:
        object_p = np.asarray(
            e.obj.pose.sp.p,
            dtype=np.float64,
        )

        target_local = np.asarray(
            local_target,
            dtype=np.float64,
        ).copy()

        target_local[2] = float(
            e.local_center[2]
        )

        target_world = (
            base.local_to_world_point(
                object_p,
                base.KNIFE_YAW_DEG,
                target_local,
            )
        )

        approaching = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64,
        )

        closing = np.array(
            [
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )

        grasp_pose = (
            e.agent.build_grasp_pose(
                approaching,
                closing,
                target_world,
            )
        )

        grasp_p = np.asarray(
            grasp_pose.p,
            dtype=np.float64,
        )

        tcp0_p = np.asarray(
            e.agent.tcp.pose.sp.p,
            dtype=np.float64,
        )

        safe_z = max(
            0.30,
            float(
                tcp0_p[2]
            )
            + 0.10,
            float(
                grasp_p[2]
            )
            + 0.12,
        )

        dx = float(
            start_offset_xy[0]
        )

        dy = float(
            start_offset_xy[1]
        )

        # ----------------------------------------------------
        # PREPARATION — NOT RECORDED
        # ----------------------------------------------------

        prepared_start_pose = (
            sapien.Pose(
                p=np.array(
                    [
                        float(
                            tcp0_p[0]
                        )
                        + dx,
                        float(
                            tcp0_p[1]
                        )
                        + dy,
                        float(
                            safe_z
                        ),
                    ],
                    dtype=np.float64,
                ),
                q=grasp_pose.q,
            )
        )

        if not base.planner_step_ok(
            planner.move_to_pose_with_RRTConnect(
                prepared_start_pose
            )
        ):
            return (
                None,
                "preparation",
            )

        # Re-apply the exact knife pose after the unrecorded
        # preparation motion and BEFORE frame 0 is captured.
        restore_exact_knife_pose(
            e,
            knife_xy,
        )

        env.restart_capture_from_current_state()

        prepared_qpos = (
            base.get_qpos(
                env
            ).copy()
        )

        prepared_tcp = (
            base.get_tcp_xyz(
                env
            ).copy()
        )

        # ----------------------------------------------------
        # RECORDED 1 — HORIZONTAL MOVEMENT
        # ----------------------------------------------------

        aligned_pose = sapien.Pose(
            p=np.array(
                [
                    float(
                        grasp_p[0]
                    ),
                    float(
                        grasp_p[1]
                    ),
                    float(
                        safe_z
                    ),
                ],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )

        if not base.planner_step_ok(
            planner.move_to_pose_with_RRTConnect(
                aligned_pose
            )
        ):
            return (
                None,
                "horizontal_align",
            )

        # ----------------------------------------------------
        # RECORDED 2 — DESCENT
        # ----------------------------------------------------

        if not base.planner_step_ok(
            planner.move_to_pose_with_RRTConnect(
                grasp_pose
            )
        ):
            return (
                None,
                "descent",
            )

        # ----------------------------------------------------
        # RECORDED 3 — GRASP
        # ----------------------------------------------------

        planner.close_gripper()

        # ----------------------------------------------------
        # RECORDED 4 — LIFT
        # ----------------------------------------------------

        lift_pose = sapien.Pose(
            p=grasp_p
            + np.array(
                [
                    0.0,
                    0.0,
                    base.LIFT_DISTANCE,
                ],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )

        if not base.planner_step_ok(
            planner.move_to_pose_with_RRTConnect(
                lift_pose
            )
        ):
            return (
                None,
                "lift",
            )

        ep = env.episode()

        ep[
            "variant"
        ] = variant

        ep[
            "label"
        ] = int(
            label
        )

        ep[
            "knife_xy"
        ] = np.asarray(
            knife_xy,
            dtype=np.float32,
        )

        ep[
            "start_offset_xy"
        ] = np.asarray(
            start_offset_xy,
            dtype=np.float32,
        )

        ep[
            "target_local"
        ] = target_local.astype(
            np.float32
        )

        ep[
            "target_world"
        ] = target_world.astype(
            np.float32
        )

        ep[
            "prepared_qpos"
        ] = prepared_qpos.astype(
            np.float32
        )

        ep[
            "prepared_tcp"
        ] = prepared_tcp.astype(
            np.float32
        )

        return (
            ep,
            "ok",
        )

    finally:
        planner.close()


def write_episode_group(
    h5,
    *,
    trajectory_id,
    pair_id,
    object_id,
    start_id,
    object_row,
    object_col,
    start_row,
    start_col,
    ep,
):
    """
    Save one variable-length trajectory in its own HDF5 group.
    """

    name = (
        f"trajectory_"
        f"{trajectory_id:03d}"
    )

    g = h5.create_group(
        name
    )

    g.attrs[
        "trajectory_id"
    ] = int(
        trajectory_id
    )

    g.attrs[
        "pair_id"
    ] = int(
        pair_id
    )

    g.attrs[
        "object_id"
    ] = int(
        object_id
    )

    g.attrs[
        "start_id"
    ] = int(
        start_id
    )

    g.attrs[
        "object_row"
    ] = int(
        object_row
    )

    g.attrs[
        "object_col"
    ] = int(
        object_col
    )

    g.attrs[
        "start_row"
    ] = int(
        start_row
    )

    g.attrs[
        "start_col"
    ] = int(
        start_col
    )

    g.attrs[
        "variant"
    ] = str(
        ep[
            "variant"
        ]
    )

    g.attrs[
        "label"
    ] = int(
        ep[
            "label"
        ]
    )

    g.attrs[
        "frames"
    ] = int(
        len(
            ep[
                "rgb"
            ]
        )
    )

    # Primary model input.
    g.create_dataset(
        "rgb",
        data=ep[
            "rgb"
        ],
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )

    # Debug / validation metadata.
    g.create_dataset(
        "qpos",
        data=ep[
            "qpos"
        ],
        compression="gzip",
        compression_opts=4,
    )

    g.create_dataset(
        "tcp_xyz",
        data=ep[
            "tcp_xyz"
        ],
        compression="gzip",
        compression_opts=4,
    )

    g.create_dataset(
        "object_xyz",
        data=ep[
            "object_xyz"
        ],
        compression="gzip",
        compression_opts=4,
    )

    g.create_dataset(
        "actions",
        data=ep[
            "actions"
        ],
        compression="gzip",
        compression_opts=4,
    )

    g.create_dataset(
        "knife_xy",
        data=ep[
            "knife_xy"
        ],
    )

    g.create_dataset(
        "start_offset_xy",
        data=ep[
            "start_offset_xy"
        ],
    )

    g.create_dataset(
        "target_local",
        data=ep[
            "target_local"
        ],
    )

    g.create_dataset(
        "target_world",
        data=ep[
            "target_world"
        ],
    )

    g.create_dataset(
        "prepared_qpos",
        data=ep[
            "prepared_qpos"
        ],
    )

    g.create_dataset(
        "prepared_tcp",
        data=ep[
            "prepared_tcp"
        ],
    )


def save_initial_contact_sheet(
    pair_initial_frames,
):
    """
    9x9 contact sheet.

    Rows:
        object positions 0..8

    Columns:
        Panda start positions 0..8

    Each available cell is the REAL RGB frame 0 from the SAFE trajectory
    of an accepted SAFE/HAZARD pair.

    Missing/rejected pairs are explicitly marked.
    """

    thumb = 128
    header = 24

    canvas = Image.new(
        "RGB",
        (
            9 * thumb,
            9 * (
                thumb
                + header
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    lookup = {
        (
            int(item["object_id"]),
            int(item["start_id"]),
        ): item
        for item in pair_initial_frames
    }

    for object_id in range(9):
        for start_id in range(9):
            px = start_id * thumb
            py = object_id * (
                thumb
                + header
            )

            item = lookup.get(
                (
                    object_id,
                    start_id,
                )
            )

            if item is not None:
                frame = (
                    Image.fromarray(
                        item["rgb"]
                    )
                    .convert("RGB")
                    .resize(
                        (
                            thumb,
                            thumb,
                        )
                    )
                )

                canvas.paste(
                    frame,
                    (
                        px,
                        py,
                    ),
                )

                label = (
                    f"O{object_id}"
                    f"/S{start_id}"
                )

            else:
                draw.rectangle(
                    [
                        px,
                        py,
                        px + thumb - 1,
                        py + thumb - 1,
                    ],
                    outline="black",
                )

                draw.text(
                    (
                        px + 34,
                        py + 56,
                    ),
                    "MISSING",
                    fill="black",
                )

                label = (
                    f"O{object_id}"
                    f"/S{start_id}"
                    " rejected"
                )

            draw.text(
                (
                    px + 4,
                    py + thumb + 4,
                ),
                label,
                fill="black",
            )

    path = (
        OUT_DIR
        / "00_initial_states_9x9.png"
    )

    canvas.save(
        path
    )

    print(
        "Initial-state contact sheet:",
        path.resolve(),
    )



def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_env = gym.make(
        base.ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
        max_episode_steps=250,
    )

    env = base.CaptureEnv(
        base_env
    )

    manifest_rows = []

    pair_initial_frames = []

    trajectory_id = 0
    pair_id = 0

    successful_trajectories = 0
    accepted_pairs = 0
    failed_pairs = 0

    failure_counts = {}

    try:
        with h5py.File(
            H5_PATH,
            "w",
        ) as h5:

            # Root metadata.
            h5.attrs[
                "dataset_name"
            ] = (
                "pain_affordance_162"
            )

            h5.attrs[
                "primary_observation"
            ] = (
                "RGB 256x256"
            )

            h5.attrs[
                "label_0"
            ] = (
                "safe_handle"
            )

            h5.attrs[
                "label_1"
            ] = (
                "hazard_blade"
            )

            h5.attrs[
                "knife_yaw_deg"
            ] = float(
                base.KNIFE_YAW_DEG
            )

            h5.attrs[
                "object_locations"
            ] = 9

            h5.attrs[
                "robot_starts"
            ] = 9

            h5.attrs[
                "variants"
            ] = 2

            h5.attrs[
                "requested_trajectories"
            ] = 162

            object_id = 0

            for object_row, knife_y in enumerate(
                OBJECT_Y
            ):
                for object_col, knife_x in enumerate(
                    OBJECT_X
                ):
                    knife_xy = np.array(
                        [
                            knife_x,
                            knife_y,
                        ],
                        dtype=np.float64,
                    )

                    start_id = 0

                    for start_row, dy in enumerate(
                        START_DY
                    ):
                        for start_col, dx in enumerate(
                            START_DX
                        ):
                            start_offset_xy = np.array(
                                [
                                    dx,
                                    dy,
                                ],
                                dtype=np.float64,
                            )

                            print()
                            print(
                                "=" * 100
                            )

                            print(
                                f"PAIR {pair_id:02d}/80 | "
                                f"object={object_id} "
                                f"xy={knife_xy.tolist()} | "
                                f"start={start_id} "
                                f"offset={start_offset_xy.tolist()}"
                            )

                            print(
                                "=" * 100
                            )

                            pair_eps = {}

                            pair_success = True

                            for (
                                variant,
                                label,
                                local_target,
                            ) in VARIANTS:

                                print(
                                    f"  {variant.upper():6s}"
                                )

                                ep, status = (
                                    run_trajectory(
                                        env,
                                        knife_xy=
                                            knife_xy,
                                        start_offset_xy=
                                            start_offset_xy,
                                        variant=
                                            variant,
                                        label=
                                            label,
                                        local_target=
                                            local_target,
                                        scene_seed=
                                            SCENE_SEED,
                                    )
                                )

                                row = {
                                    "trajectory_id":
                                        trajectory_id,
                                    "pair_id":
                                        pair_id,
                                    "object_id":
                                        object_id,
                                    "object_row":
                                        object_row,
                                    "object_col":
                                        object_col,
                                    "knife_x":
                                        float(
                                            knife_x
                                        ),
                                    "knife_y":
                                        float(
                                            knife_y
                                        ),
                                    "start_id":
                                        start_id,
                                    "start_row":
                                        start_row,
                                    "start_col":
                                        start_col,
                                    "start_dx":
                                        float(
                                            dx
                                        ),
                                    "start_dy":
                                        float(
                                            dy
                                        ),
                                    "variant":
                                        variant,
                                    "label":
                                        int(
                                            label
                                        ),
                                    "success":
                                        ep
                                        is not None,
                                    "failure_stage":
                                        (
                                            ""
                                            if ep
                                            is not None
                                            else status
                                        ),
                                    "frames":
                                        (
                                            int(
                                                len(
                                                    ep[
                                                        "rgb"
                                                    ]
                                                )
                                            )
                                            if ep
                                            is not None
                                            else 0
                                        ),
                                    "initial_rgb_mae_pair":
                                        "",
                                    "initial_qpos_max_pair":
                                        "",
                                    "initial_object_max_pair":
                                        "",
                                }

                                manifest_rows.append(
                                    row
                                )

                                if ep is None:
                                    print(
                                        f"    FAILED at {status}"
                                    )

                                    pair_success = False

                                    failure_counts[
                                        status
                                    ] = (
                                        failure_counts.get(
                                            status,
                                            0,
                                        )
                                        + 1
                                    )

                                else:
                                    print(
                                        f"    OK | "
                                        f"frames="
                                        f"{len(ep['rgb'])}"
                                    )

                                    successful_trajectories += 1

                                    pair_eps[
                                        variant
                                    ] = ep

                                trajectory_id += 1

                            # --------------------------------------
                            # Pair anti-leakage validation
                            # --------------------------------------

                            if (
                                "safe"
                                in pair_eps
                                and
                                "hazard"
                                in pair_eps
                            ):
                                pair = (
                                    base.initial_pair_check(
                                        pair_eps[
                                            "safe"
                                        ],
                                        pair_eps[
                                            "hazard"
                                        ],
                                    )
                                )

                                print(
                                    "  PAIR CHECK | "
                                    f"RGB_MAE="
                                    f"{pair['initial_rgb_mae']:.8f} | "
                                    f"qpos_max="
                                    f"{pair['initial_qpos_max_abs']:.8e} | "
                                    f"obj_max="
                                    f"{pair['initial_object_xyz_max_abs']:.8e}"
                                )

                                for row in manifest_rows:
                                    if (
                                        row[
                                            "pair_id"
                                        ]
                                        == pair_id
                                    ):
                                        row[
                                            "initial_rgb_mae_pair"
                                        ] = (
                                            pair[
                                                "initial_rgb_mae"
                                            ]
                                        )

                                        row[
                                            "initial_qpos_max_pair"
                                        ] = (
                                            pair[
                                                "initial_qpos_max_abs"
                                            ]
                                        )

                                        row[
                                            "initial_object_max_pair"
                                        ] = (
                                            pair[
                                                "initial_object_xyz_max_abs"
                                            ]
                                        )

                                if (
                                    pair[
                                        "initial_rgb_mae"
                                    ]
                                    > 0.25
                                    or
                                    pair[
                                        "initial_qpos_max_abs"
                                    ]
                                    > 1e-5
                                    or
                                    pair[
                                        "initial_object_xyz_max_abs"
                                    ]
                                    > 1e-6
                                ):
                                    pair_success = False

                                    print(
                                        "  PAIR REJECTED: "
                                        "initial-state mismatch"
                                    )

                                else:
                                    pair_initial_frames.append(
                                        {
                                            "object_id":
                                                object_id,
                                            "start_id":
                                                start_id,
                                            "rgb":
                                                pair_eps[
                                                    "safe"
                                                ][
                                                    "rgb"
                                                ][
                                                    0
                                                ].copy(),
                                        }
                                    )

                            if (
                                pair_success
                                and
                                len(
                                    pair_eps
                                )
                                == 2
                            ):
                                # Only validated complete pairs are committed
                                # to the final HDF5 dataset.
                                for (
                                    accepted_variant,
                                    accepted_label,
                                    _,
                                ) in VARIANTS:
                                    accepted_ep = pair_eps[
                                        accepted_variant
                                    ]

                                    accepted_tid = (
                                        pair_id * 2
                                        + int(
                                            accepted_label
                                        )
                                    )

                                    write_episode_group(
                                        h5,
                                        trajectory_id=
                                            accepted_tid,
                                        pair_id=
                                            pair_id,
                                        object_id=
                                            object_id,
                                        start_id=
                                            start_id,
                                        object_row=
                                            object_row,
                                        object_col=
                                            object_col,
                                        start_row=
                                            start_row,
                                        start_col=
                                            start_col,
                                        ep=
                                            accepted_ep,
                                    )

                                accepted_pairs += 1

                                print(
                                    "  PAIR ACCEPTED + SAVED"
                                )

                            else:
                                failed_pairs += 1

                                print(
                                    "  PAIR FAILED"
                                )

                            pair_id += 1
                            start_id += 1

                    object_id += 1

            h5.attrs[
                "successful_trajectories"
            ] = int(
                successful_trajectories
            )

            h5.attrs[
                "accepted_pairs"
            ] = int(
                accepted_pairs
            )

            h5.attrs[
                "failed_pairs"
            ] = int(
                failed_pairs
            )

        # ====================================================
        # Manifest CSV
        # ====================================================

        with MANIFEST_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as fp:

            writer = csv.DictWriter(
                fp,
                fieldnames=
                    list(
                        manifest_rows[
                            0
                        ].keys()
                    ),
            )

            writer.writeheader()

            writer.writerows(
                manifest_rows
            )

        # ====================================================
        # Initial-state visual contact sheet
        # ====================================================

        save_initial_contact_sheet(
            pair_initial_frames
        )

        # ====================================================
        # Summary JSON
        # ====================================================

        frame_counts = [
            int(
                r[
                    "frames"
                ]
            )
            for r in manifest_rows
            if bool(
                r[
                    "success"
                ]
            )
        ]

        summary = {
            "dataset":
                "pain_affordance_162",
            "model_id":
                base.MODEL_ID,
            "primary_observation":
                "RGB 256x256",
            "qpos_role":
                "metadata/debug only",
            "object_grid_x_m":
                list(
                    OBJECT_X
                ),
            "object_grid_y_m":
                list(
                    OBJECT_Y
                ),
            "robot_start_dx_m":
                list(
                    START_DX
                ),
            "robot_start_dy_m":
                list(
                    START_DY
                ),
            "knife_yaw_deg":
                float(
                    base.KNIFE_YAW_DEG
                ),
            "safe_target_local_m":
                SAFE_TARGET_LOCAL.tolist(),
            "hazard_target_local_m":
                HAZARD_TARGET_LOCAL.tolist(),
            "object_locations":
                9,
            "robot_starts":
                9,
            "variants":
                2,
            "requested_trajectories":
                162,
            "successful_trajectories":
                successful_trajectories,
            "accepted_pairs":
                accepted_pairs,
            "failed_pairs":
                failed_pairs,
            "failure_counts":
                failure_counts,
            "frame_count_min":
                (
                    min(
                        frame_counts
                    )
                    if frame_counts
                    else None
                ),
            "frame_count_max":
                (
                    max(
                        frame_counts
                    )
                    if frame_counts
                    else None
                ),
            "frame_count_mean":
                (
                    float(
                        np.mean(
                            frame_counts
                        )
                    )
                    if frame_counts
                    else None
                ),
            "descent_planner":
                "RRTConnect",
            "horizontal_planner":
                "RRTConnect",
            "lift_planner":
                "RRTConnect",
            "h5_contains":
                "accepted complete SAFE/HAZARD pairs only",
            "h5_file":
                H5_PATH.name,
            "manifest_file":
                MANIFEST_PATH.name,
            "initial_state_contact_sheet":
                "00_initial_states_9x9.png",
        }

        SUMMARY_PATH.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # Final print
        # ====================================================

        print()
        print(
            "=" * 100
        )

        print(
            "FINAL PAIN AFFORDANCE DATASET SUMMARY"
        )

        print(
            "=" * 100
        )

        print(
            "Successful trajectories :",
            f"{successful_trajectories}/162",
        )

        print(
            "Accepted pairs          :",
            f"{accepted_pairs}/81",
        )

        print(
            "Failed pairs            :",
            f"{failed_pairs}/81",
        )

        print(
            "HDF5                    :",
            H5_PATH.resolve(),
        )

        print(
            "Manifest                :",
            MANIFEST_PATH.resolve(),
        )

        print(
            "Summary                 :",
            SUMMARY_PATH.resolve(),
        )

        print(
            "Initial-state sheet     :",
            (
                OUT_DIR
                / "00_initial_states_9x9.png"
            ).resolve(),
        )

        print(
            "=" * 100
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
