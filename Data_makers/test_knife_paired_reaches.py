#!/usr/bin/env python3
"""
Phase 1B — One knife pose, one Panda start, two paired reach variants.

SAFE:
    approach -> grasp handle -> small lift

HAZARD:
    approach -> grasp blade -> same small lift

The two trajectories use:
- the same knife pose
- the same Panda initial state
- the same camera
- the same motion structure

Only the local grasp target changes.

Primary observation for the future model: RGB.
qpos / tcp_xyz are saved only as metadata/debug information.
"""

from __future__ import annotations

from pathlib import Path
import json

import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image

import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


ENV_ID = "PainAffordanceKnifePair-v1"
MODEL_ID = "032_knife"

CAMERA_EYE = [0.60, -0.70, 0.60]
CAMERA_TARGET = [0.00, 0.00, 0.12]

# Phase 1B: one fixed object pose.
KNIFE_X = 0.0
KNIFE_Y = 0.0
KNIFE_YAW_DEG = 0.0

# Confirmed in Phase 1A:
# +X = red handle
# -X = black blade
SAFE_TARGET_LOCAL = np.array([+0.065, 0.0, 0.0], dtype=np.float64)
HAZARD_TARGET_LOCAL = np.array([-0.065, 0.0, 0.0], dtype=np.float64)

PREGRASP_HEIGHT = 0.070
LIFT_DISTANCE = 0.040

OUT_DIR = Path("outputs/pain_affordance_phase1b")


@register_env(
    ENV_ID,
    max_episode_steps=200,
    asset_download_ids=["ycb"],
)
class PainAffordanceKnifePairEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=CAMERA_EYE,
            target=CAMERA_TARGET,
        )
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=256,
                height=256,
                fov=1.0,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=CAMERA_EYE,
            target=CAMERA_TARGET,
        )
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{MODEL_ID}",
        )

        # Temporary build pose, used only so the actor can be constructed safely.
        builder.initial_pose = sapien.Pose(
            p=[0.0, 0.0, 0.20],
            q=[1.0, 0.0, 0.0, 0.0],
        )

        self.obj = builder.build(name=MODEL_ID)

    

    def _after_reconfigure(self, options: dict):
        mesh = self.obj.get_first_collision_mesh()

        bounds_world_at_build = np.asarray(
            mesh.bounding_box.bounds,
            dtype=np.float64,
        )

        # Actor was built at z=0.20. Convert the mesh bounds back to local Z.
        local_bounds = bounds_world_at_build.copy()
        local_bounds[:, 2] -= 0.20

        self.local_bounds = local_bounds
        self.local_center = 0.5 * (
            local_bounds[0] + local_bounds[1]
        )

        # Place lowest local collision point exactly on table z=0.
        self.object_support_z = float(
            -local_bounds[0, 2]
        )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            knife_xy = options.get(
                "knife_xy",
                [KNIFE_X, KNIFE_Y],
            )
            yaw_deg = float(
                options.get(
                    "knife_yaw_deg",
                    KNIFE_YAW_DEG,
                )
            )

            yaw = np.deg2rad(yaw_deg)

            q = torch.tensor(
                [[
                    np.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    np.sin(yaw / 2.0),
                ]],
                dtype=torch.float32,
                device=self.device,
            )

            p = torch.tensor(
                [[
                    float(knife_xy[0]),
                    float(knife_xy[1]),
                    float(self.object_support_z),
                ]],
                dtype=torch.float32,
                device=self.device,
            )

            self.obj.set_pose(
                Pose.create_from_pq(
                    p=p,
                    q=q,
                )
            )

    def evaluate(self):
        return {
            "success": torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            ),
            "fail": torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

    def compute_normalized_dense_reward(self, obs, action, info):
        return torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

    def _get_obs_extra(self, info: dict):
        return {}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def unbatch(x):
    x = to_numpy(x)
    if x.ndim > 0 and x.shape[0] == 1:
        x = x[0]
    return x


def observation_rgb(obs):
    x = unbatch(
        obs["sensor_data"]["base_camera"]["rgb"]
    )

    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating):
            if float(np.nanmax(x)) <= 1.0 + 1e-6:
                x = np.clip(x, 0.0, 1.0) * 255.0

        x = np.clip(
            x,
            0,
            255,
        ).astype(np.uint8)

    return x


def get_qpos(env):
    return unbatch(
        env.unwrapped.agent.robot.get_qpos()
    ).astype(np.float32)


def get_tcp_xyz(env):
    return unbatch(
        env.unwrapped.agent.tcp.pose.p
    ).astype(np.float32)


def get_object_xyz(env):
    return unbatch(
        env.unwrapped.obj.pose.p
    ).astype(np.float32)


class CaptureEnv(gym.Wrapper):
    """Record RGB as primary data plus kinematic metadata for validation."""

    def __init__(self, env):
        super().__init__(env)
        self.clear()

    def clear(self):
        self.rgb = []
        self.qpos = []
        self.tcp_xyz = []
        self.object_xyz = []
        self.actions = []

    def _append_observation(self, obs):
        self.rgb.append(
            observation_rgb(obs)
        )
        self.qpos.append(
            get_qpos(self)
        )
        self.tcp_xyz.append(
            get_tcp_xyz(self)
        )
        self.object_xyz.append(
            get_object_xyz(self)
        )

    def reset(self, **kwargs):
        self.clear()
        obs, info = self.env.reset(**kwargs)
        self._append_observation(obs)
        return obs, info

    def restart_capture_from_current_state(self):
        """
        Discard preparation motion and begin the recorded trajectory
        from the robot's current physical state.
        """
        self.clear()

        obs = self.env.unwrapped.get_obs()

        self._append_observation(obs)

        print(
            "recording restarted from prepared pose"
        )


    def step(self, action):
        action_np = unbatch(
            action
        ).astype(np.float32).reshape(-1)

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.actions.append(action_np)
        self._append_observation(obs)

        return obs, reward, terminated, truncated, info

    def episode(self):
        if len(self.rgb) == 0:
            raise RuntimeError("No episode was captured")

        return {
            "rgb": np.stack(self.rgb).astype(np.uint8),
            "qpos": np.stack(self.qpos).astype(np.float32),
            "tcp_xyz": np.stack(self.tcp_xyz).astype(np.float32),
            "object_xyz": np.stack(self.object_xyz).astype(np.float32),
            "actions": np.stack(self.actions).astype(np.float32),
        }


def new_planner(env, vis=False):
    return PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )


def planner_step_ok(result):
    if result is None:
        return False

    if isinstance(result, (int, np.integer)):
        return int(result) != -1

    return True


def local_to_world_point(
    object_position,
    yaw_deg,
    local_point,
):
    """
    Phase-1B transform.

    Knife has only a yaw rotation, so local XY can be converted explicitly.
    """
    theta = np.deg2rad(float(yaw_deg))

    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ],
        dtype=np.float64,
    )

    return (
        np.asarray(object_position, dtype=np.float64)
        + Rz @ np.asarray(local_point, dtype=np.float64)
    )


def run_reach_variant(
    env,
    *,
    variant,
    local_target,
    scene_seed=123,
    vis=False,
):
    """
    Paired reach policy.

    Both variants execute the same motion structure:
        1. rise vertically
        2. align above target
        3. descend continuously
        4. close gripper
        5. lift by the same fixed distance

    Only local_target differs.
    """
    env.reset(
        seed=int(scene_seed),
        options={
            "knife_xy": [KNIFE_X, KNIFE_Y],
            "knife_yaw_deg": KNIFE_YAW_DEG,
        },
    )

    planner = new_planner(
        env,
        vis=vis,
    )

    e = env.unwrapped

    try:
        object_p = np.asarray(
            e.obj.pose.sp.p,
            dtype=np.float64,
        )

        # The desired target sits around the middle of the knife thickness.
        target_local = np.asarray(
            local_target,
            dtype=np.float64,
        ).copy()

        target_local[2] = float(
            e.local_center[2]
        )

        target_world = local_to_world_point(
            object_p,
            KNIFE_YAW_DEG,
            target_local,
        )

        # Top-down grasp. Fingers close across knife local/world Y.
        approaching = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )

        closing = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

        grasp_pose = e.agent.build_grasp_pose(
            approaching,
            closing,
            target_world,
        )

        grasp_p = np.asarray(
            grasp_pose.p,
            dtype=np.float64,
        )

        tcp0_p = np.asarray(
            e.agent.tcp.pose.sp.p,
            dtype=np.float64,
        )

        tcp0_q = np.asarray(
            e.agent.tcp.pose.sp.q,
            dtype=np.float64,
        )

        safe_z = max(
            0.30,
            float(tcp0_p[2]) + 0.10,
            float(grasp_p[2]) + 0.12,
        )

        print()
        print("-" * 72)
        print(f"{variant.upper()} REACH")
        print("-" * 72)
        print("local target [m] :", np.round(target_local, 6))
        print("world target [m] :", np.round(target_world, 6))
        print("initial TCP [m]  :", np.round(tcp0_p, 6))
        print("safe Z [m]       :", round(float(safe_z), 6))

        # -------------------------------------------------------------
        # PREPARATION — NOT RECORDED
        #
        # Put the Panda at the common starting position with the wrist
        # already using the final grasp orientation.
        #
        # SAFE and HAZARD use exactly the same orientation.
        # -------------------------------------------------------------

        prepared_start_pose = sapien.Pose(
            p=np.array(
                [
                    float(tcp0_p[0]),
                    float(tcp0_p[1]),
                    float(safe_z),
                ],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_RRTConnect(
                prepared_start_pose
            )
        ):
            print(
                f"{variant}: FAILED at preparation"
            )
            return None

        # IMPORTANT:
        # Everything before this point is preparation and must not
        # become part of the RGB trajectory.
        env.restart_capture_from_current_state()

        prepared_tcp = np.asarray(
            e.agent.tcp.pose.sp.p,
            dtype=np.float64,
        )

        print(
            "prepared TCP [m] :",
            np.round(prepared_tcp, 6),
        )

        # 2) Horizontal alignment over the chosen affordance.
        aligned_pose = sapien.Pose(
            p=np.array(
                [
                    float(grasp_p[0]),
                    float(grasp_p[1]),
                    float(safe_z),
                ],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )
        if not planner_step_ok(
            planner.move_to_pose_with_RRTConnect(aligned_pose)
        ):
            print(f"{variant}: FAILED at horizontal-align")
            return None
        
        # 3) One continuous descent.
        if not planner_step_ok(
            planner.move_to_pose_with_screw(grasp_pose)
        ):
            print(f"{variant}: FAILED at descent")
            return None

        # 4) Same grasp action for both variants.
        planner.close_gripper()

        # 5) Same lift for both variants.
        lift_pose = sapien.Pose(
            p=grasp_p
            + np.array(
                [0.0, 0.0, LIFT_DISTANCE],
                dtype=np.float64,
            ),
            q=grasp_pose.q,
        )

        if not planner_step_ok(
            planner.move_to_pose_with_screw(lift_pose)
        ):
            print(f"{variant}: FAILED at lift")
            return None

        ep = env.episode()

        ep["variant"] = variant
        ep["target_local"] = target_local.astype(np.float32)
        ep["target_world"] = target_world.astype(np.float32)

        return ep

    finally:
        planner.close()


def save_gif(frames, path, fps=15):
    images = [
        Image.fromarray(frame)
        for frame in frames
    ]

    duration_ms = int(
        round(1000.0 / float(fps))
    )

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def save_contact_frames(ep, prefix):
    """
    Save a few representative RGB frames for easy visual checking.
    """
    T = len(ep["rgb"])

    ids = {
        "start": 0,
        "quarter": max(0, int(round(0.25 * (T - 1)))),
        "half": max(0, int(round(0.50 * (T - 1)))),
        "threequarter": max(0, int(round(0.75 * (T - 1)))),
        "final": T - 1,
    }

    for label, idx in ids.items():
        Image.fromarray(
            ep["rgb"][idx]
        ).save(
            OUT_DIR / f"{prefix}_{label}_{idx:04d}.png"
        )


def initial_pair_check(safe, hazard):
    rgb_mae = float(
        np.mean(
            np.abs(
                safe["rgb"][0].astype(np.float32)
                - hazard["rgb"][0].astype(np.float32)
            )
        )
    )

    qmax = float(
        np.max(
            np.abs(
                safe["qpos"][0]
                - hazard["qpos"][0]
            )
        )
    )

    object_max = float(
        np.max(
            np.abs(
                safe["object_xyz"][0]
                - hazard["object_xyz"][0]
            )
        )
    )

    return {
        "initial_rgb_mae": rgb_mae,
        "initial_qpos_max_abs": qmax,
        "initial_object_xyz_max_abs": object_max,
    }


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_env = gym.make(
        ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
        max_episode_steps=200,
    )

    env = CaptureEnv(base_env)

    try:
        safe = run_reach_variant(
            env,
            variant="safe",
            local_target=SAFE_TARGET_LOCAL,
            scene_seed=123,
        )

        hazard = run_reach_variant(
            env,
            variant="hazard",
            local_target=HAZARD_TARGET_LOCAL,
            scene_seed=123,
        )

        if safe is None or hazard is None:
            raise RuntimeError(
                "At least one reach failed. "
                "Inspect the printed failing stage before proceeding."
            )

        pair = initial_pair_check(
            safe,
            hazard,
        )

        print()
        print("=" * 72)
        print("PAIRED INITIAL-STATE CHECK")
        print("=" * 72)
        print(
            "initial RGB MAE       :",
            pair["initial_rgb_mae"],
        )
        print(
            "initial qpos max abs  :",
            pair["initial_qpos_max_abs"],
        )
        print(
            "initial object max abs:",
            pair["initial_object_xyz_max_abs"],
        )

        # Strict anti-shortcut checks.
        if pair["initial_rgb_mae"] > 0.25:
            raise RuntimeError(
                "Initial SAFE/HAZARD RGB differs too much."
            )

        if pair["initial_qpos_max_abs"] > 1e-5:
            raise RuntimeError(
                "Initial SAFE/HAZARD robot state differs."
            )

        if pair["initial_object_xyz_max_abs"] > 1e-6:
            raise RuntimeError(
                "Initial SAFE/HAZARD object pose differs."
            )

        # RGB outputs for human inspection.
        save_gif(
            safe["rgb"],
            OUT_DIR / "safe_handle_reach.gif",
        )

        save_gif(
            hazard["rgb"],
            OUT_DIR / "hazard_blade_reach.gif",
        )

        save_contact_frames(
            safe,
            "safe",
        )

        save_contact_frames(
            hazard,
            "hazard",
        )

        # Compact NPZ for this smoke test.
        np.savez_compressed(
            OUT_DIR / "phase1b_pair.npz",
            safe_rgb=safe["rgb"],
            hazard_rgb=hazard["rgb"],
            safe_qpos=safe["qpos"],
            hazard_qpos=hazard["qpos"],
            safe_tcp_xyz=safe["tcp_xyz"],
            hazard_tcp_xyz=hazard["tcp_xyz"],
            safe_object_xyz=safe["object_xyz"],
            hazard_object_xyz=hazard["object_xyz"],
            safe_target_local=safe["target_local"],
            hazard_target_local=hazard["target_local"],
            safe_target_world=safe["target_world"],
            hazard_target_world=hazard["target_world"],
        )

        summary = {
            "model_id": MODEL_ID,
            "knife_xy_m": [KNIFE_X, KNIFE_Y],
            "knife_yaw_deg": KNIFE_YAW_DEG,
            "safe_target_local_m": SAFE_TARGET_LOCAL.tolist(),
            "hazard_target_local_m": HAZARD_TARGET_LOCAL.tolist(),
            "pregrasp_height_m": PREGRASP_HEIGHT,
            "lift_distance_m": LIFT_DISTANCE,
            "safe_frames": int(len(safe["rgb"])),
            "hazard_frames": int(len(hazard["rgb"])),
            "primary_observation": "RGB",
            "qpos_role": "metadata/debug only",
            "paired_initial_check": pair,
        }

        (
            OUT_DIR / "phase1b_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print("PHASE 1B COMPLETE")
        print("=" * 72)
        print(
            "SAFE GIF   :",
            (OUT_DIR / "safe_handle_reach.gif").resolve(),
        )
        print(
            "HAZARD GIF :",
            (OUT_DIR / "hazard_blade_reach.gif").resolve(),
        )
        print(
            "NPZ        :",
            (OUT_DIR / "phase1b_pair.npz").resolve(),
        )
        print(
            "SUMMARY    :",
            (OUT_DIR / "phase1b_summary.json").resolve(),
        )
        print("=" * 72)

    finally:
        env.close()


if __name__ == "__main__":
    main()
