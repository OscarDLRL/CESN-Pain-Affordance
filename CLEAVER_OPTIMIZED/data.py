"""Dataset access shared by autoencoder, latent export and CESN.

The reference frame (cfg.REFERENCE_FRAME) is applied only here and in
extract_latents, so training and latent export cannot disagree.
"""
import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg


def trajectory_keys(path=cfg.RGB):
    with h5py.File(path, "r") as f:
        return sorted(k for k in f if k.startswith("trajectory_"))


def split_keys(path=cfg.RGB):
    """{'train'|'validation'|'test': [trajectory keys]} with integrity checks."""
    groups = {"train": set(cfg.TRAIN), "validation": set(cfg.VALIDATION), "test": set(cfg.TEST)}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not groups[a] & groups[b], f"split overlap {a}/{b}"
    out = {k: [] for k in groups}
    labels = {}
    with h5py.File(path, "r") as f:
        for key in sorted(k for k in f if k.startswith("trajectory_")):
            g = f[key]
            pair = int(g.attrs["pair_id"])
            labels.setdefault(pair, []).append(int(g.attrs["label"]))
            assert len(g["rgb"]) == cfg.NUM_FRAMES, key
            for name, ids in groups.items():
                if pair in ids:
                    out[name].append(key)
    assert set(labels) == set.union(*groups.values()), "split does not cover the dataset"
    assert all(sorted(v) == [0, 1] for v in labels.values()), "every pair needs one SAFE and one PAIN"
    return out


def check_reference_is_settled(path=cfg.RGB, tol=1e-5):
    """The cleaver must not move between REFERENCE_FRAME and the next frame."""
    r = cfg.REFERENCE_FRAME
    with h5py.File(path, "r") as f:
        for key in (k for k in f if k.startswith("trajectory_")):
            xyz = f[key]["object_xyz"]
            drift = float(np.abs(xyz[r + 1] - xyz[r]).max())
            assert drift < tol, f"{key}: cleaver still moving at reference frame {r} ({drift:.2e} m)"


def load_rgb(keys, path=cfg.RGB):
    """uint8 cache [N, 223, 256, 256, 3]; never duplicated as float32."""
    rgb = np.empty((len(keys), cfg.NUM_FRAMES, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.uint8)
    with h5py.File(path, "r") as f:
        for i, k in enumerate(keys):
            ds = f[k]["rgb"]
            assert ds.shape == rgb[i].shape and ds.dtype == np.uint8, k
            ds.read_direct(rgb[i])
    print(f"RGB cache {rgb.nbytes / 2**30:.2f} GiB | {len(keys)} trajectories", flush=True)
    return rgb


def load_robot_state(keys, path=cfg.RGB):
    """[N, 223, 4] = TCP xyz (m) + gripper width (sum of finger joints). Training targets only."""
    out = np.empty((len(keys), cfg.NUM_FRAMES, 4), dtype=np.float32)
    with h5py.File(path, "r") as f:
        for i, k in enumerate(keys):
            out[i, :, :3] = f[k]["tcp_xyz"][:]
            out[i, :, 3] = f[k]["qpos"][:, 7:9].sum(1)
    return out


def to_tensor(a, device):
    """uint8 [B, H, W, 3] (numpy or torch) -> float [B, 3, H, W] in [0, 1], channels_last."""
    t = a if torch.is_tensor(a) else torch.from_numpy(np.ascontiguousarray(a))
    t = t.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return t.contiguous(memory_format=torch.channels_last)


def to_gpu_cache(rgb, device="cuda"):
    """Move a uint8 cache to the GPU (train+val ~5.5 GiB fits a 24 GiB card)."""
    t = torch.from_numpy(rgb).to(device)
    print(f"GPU cache {t.numel() / 2**30:.2f} GiB", flush=True)
    return t


def frame_indices(stride=1):
    """Target frames used with the reference frame (reference included: delta = 0)."""
    return np.arange(cfg.REFERENCE_FRAME, cfg.NUM_FRAMES, stride)
