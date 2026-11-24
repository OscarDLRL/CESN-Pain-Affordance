#!/usr/bin/env python3
"""
Phase 1A — Visualize candidate local affordance targets on YCB 032_knife.

Purpose:
- Load Panda + table + fixed YCB 032_knife.
- Place two visual-only markers at local X = -0.065 m and +0.065 m.
- Render one image so we can identify which side is the handle and which side is blade.
- No robot motion yet.

Marker convention:
- GREEN = +X candidate
- BLUE  = -X candidate
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


ENV_ID = "InspectYCBKnifeTargets-v1"
MODEL_ID = "032_knife"

KNIFE_X = 0.0
KNIFE_Y = 0.0
KNIFE_YAW_DEG = 0.0

# Candidate target offsets in the knife's local longitudinal axis.
TARGET_OFFSET_X = 0.065

OUT_DIR = Path("outputs/pain_affordance_phase1")
OUT_IMAGE = OUT_DIR / "032_knife_candidate_targets.png"


@register_env(
    ENV_ID,
    max_episode_steps=20,
    asset_download_ids=["ycb"],
)
class InspectYCBKnifeTargetsEnv(BaseEnv):
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
            eye=[0.60, -0.70, 0.60],
            target=[0.00, 0.00, 0.12],
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
            eye=[0.60, -0.70, 0.60],
            target=[0.00, 0.00, 0.12],
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

        # Knife
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{MODEL_ID}",
        )
        builder.initial_pose = sapien.Pose(
            p=[0.0, 0.0, 0.20],
            q=[1.0, 0.0, 0.0, 0.0],
        )
        self.obj = builder.build(name=MODEL_ID)

        # Visual-only target markers.
        # Their final z is intentionally above the knife for easy inspection.
        marker_z = 0.035

        self.marker_plus_x = actors.build_sphere(
            scene=self.scene,
            radius=0.009,
            color=[0.0, 1.0, 0.0, 1.0],  # GREEN
            name="candidate_plus_x",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=[KNIFE_X + TARGET_OFFSET_X, KNIFE_Y, marker_z]
            ),
        )

        self.marker_minus_x = actors.build_sphere(
            scene=self.scene,
            radius=0.009,
            color=[0.0, 0.25, 1.0, 1.0],  # BLUE
            name="candidate_minus_x",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=[KNIFE_X - TARGET_OFFSET_X, KNIFE_Y, marker_z]
            ),
        )

    def _after_reconfigure(self, options: dict):
        mesh = self.obj.get_first_collision_mesh()

        bounds = np.asarray(
            mesh.bounding_box.bounds,
            dtype=np.float64,
        )

        self.world_bounds_at_build = bounds

        # Knife was built temporarily at z=0.20.
        initial_builder_z = 0.20
        local_min_z = float(bounds[0, 2] - initial_builder_z)

        self.object_support_z = -local_min_z

        # Convert displayed bounds into true local bounds for reporting.
        self.local_bounds = bounds.copy()
        self.local_bounds[:, 2] -= initial_builder_z

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            yaw = np.deg2rad(KNIFE_YAW_DEG)

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
                    KNIFE_X,
                    KNIFE_Y,
                    self.object_support_z,
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

    def _get_obs_extra(self, info: dict):
        return {}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
    )

    try:
        env.reset(seed=0)
        u = env.unwrapped

        pose_p = to_numpy(u.obj.pose.p).reshape(-1, 3)[0]
        bounds = np.asarray(u.local_bounds, dtype=np.float64)

        plus_world = np.array(
            [KNIFE_X + TARGET_OFFSET_X, KNIFE_Y, 0.035],
            dtype=np.float64,
        )
        minus_world = np.array(
            [KNIFE_X - TARGET_OFFSET_X, KNIFE_Y, 0.035],
            dtype=np.float64,
        )

        print("=" * 76)
        print("PHASE 1A — YCB KNIFE CANDIDATE TARGETS")
        print("=" * 76)
        print("model id               :", MODEL_ID)
        print("knife world position   :", np.round(pose_p, 6))
        print("knife yaw [deg]        :", KNIFE_YAW_DEG)
        print()
        print("TRUE LOCAL BOUNDS")
        print("min xyz [m]            :", np.round(bounds[0], 6))
        print("max xyz [m]            :", np.round(bounds[1], 6))
        print()
        print("CANDIDATE MARKERS")
        print("GREEN = local +X")
        print("  local target [m]     :", [TARGET_OFFSET_X, 0.0, 0.0])
        print("  marker world [m]     :", np.round(plus_world, 6))
        print("BLUE  = local -X")
        print("  local target [m]     :", [-TARGET_OFFSET_X, 0.0, 0.0])
        print("  marker world [m]     :", np.round(minus_world, 6))
        print()

        frame = to_numpy(env.render())

        if frame.ndim == 4:
            frame = frame[0]

        if frame.dtype != np.uint8:
            if float(np.nanmax(frame)) <= 1.0 + 1e-6:
                frame = np.clip(frame, 0.0, 1.0) * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        plt.imsave(OUT_IMAGE, frame)

        print("render saved           :", OUT_IMAGE.resolve())
        print("=" * 76)

    finally:
        env.close()


if __name__ == "__main__":
    main()
