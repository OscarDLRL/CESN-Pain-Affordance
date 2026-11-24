"""Loader for spatial-context CESN: joints + cube metadata + evaluation goal."""

from pathlib import Path
import h5py
import numpy as np


def _attr_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_group(group, key, split, joint_dim=9):
    for required in ("qpos", "cube_xyz", "goal_xyz"):
        if required not in group:
            raise KeyError(f"{key}: missing {required}")

    joints = np.asarray(group["qpos"][:], dtype=np.float32)
    cube = np.asarray(group["cube_xyz"][:], dtype=np.float32)
    goal = np.asarray(group["goal_xyz"][:], dtype=np.float32).reshape(3)

    if joints.ndim != 2 or joints.shape[1] != int(joint_dim):
        raise ValueError(f"{key}: expected qpos (T,{joint_dim}); got {joints.shape}")

    length = int(joints.shape[0])

    if cube.shape != (length, 3):
        raise ValueError(f"{key}: expected cube_xyz ({length},3); got {cube.shape}")

    if not np.all(np.isfinite(joints)):
        raise ValueError(f"{key}: qpos contains non-finite values")
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"{key}: cube_xyz contains non-finite values")
    if not np.all(np.isfinite(goal)):
        raise ValueError(f"{key}: goal_xyz contains non-finite values")

    label = int(group.attrs["label"])
    if label == 0:
        kind = "pick"
    elif label == 1:
        kind = "push"
    else:
        raise ValueError(f"{key}: invalid label {label}")

    return {
        "id": str(key),
        "kind": kind,
        "label": label,
        "split": str(split),
        "joints": joints,
        "cube_xyz": cube,
        "goal": goal,
        "context_region": _attr_text(
            group.attrs.get("context_region", "unknown")
        ),
        "length": length,
    }


def load_split(split, data_path, joint_dim=9):
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")

    path = Path(data_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []

    with h5py.File(path, "r") as handle:
        if "trajectories" not in handle:
            raise KeyError(f"{path}: expected /trajectories")

        root = handle["trajectories"]

        for key in sorted(root):
            group = root[key]
            group_split = _attr_text(group.attrs["split"])

            if group_split != split:
                continue

            rows.append(_read_group(group, key, split, joint_dim=joint_dim))

    if not rows:
        raise ValueError(f"No trajectories for split {split} in {path}")

    return rows


def load_trajectory(data_path, trajectory_id, joint_dim=9):
    path = Path(data_path).expanduser()
    name = str(trajectory_id)

    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        root = handle["trajectories"]

        if name not in root:
            raise KeyError(f"Trajectory '{name}' not found")

        group = root[name]
        split = _attr_text(group.attrs["split"])

        return _read_group(group, name, split, joint_dim=joint_dim)