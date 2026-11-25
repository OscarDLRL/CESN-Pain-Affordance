from __future__ import annotations

from pathlib import Path
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from CLEAVER_OPTIMIZED import config as cfg

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = cfg.RGB

SPLIT_SEED = 42
TRAIN_SEED = 42

STATIC_AUX_WEIGHT = 0.15
DYNAMIC_LOSS_WEIGHT = 2.0
FRAME_LOSS_WEIGHT = 1.0
MASK_LOSS_WEIGHT = 0.75
BACKGROUND_LOSS_WEIGHT = 2.0
CENTROID_LOSS_WEIGHT = 0.40
DYNAMIC_EDGE_WEIGHT = 0.50
FRAME_EDGE_WEIGHT = 0.50
MOTION_FOCUS = 30.0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_static_index(path, pair_ids):
    index = []
    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys()):
            if not str(key).startswith("trajectory_"):
                continue
            if int(h5[key].attrs["pair_id"]) in pair_ids:
                index.append(str(key))
    return index


def collect_dynamic_index(path, pair_ids):
    index = []
    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys()):
            if not str(key).startswith("trajectory_"):
                continue

            g = h5[key]
            pair_id = int(g.attrs["pair_id"])
            if pair_id not in pair_ids:
                continue

            T = int(g["rgb"].shape[0])
            for t in range(T):
                index.append((str(key), int(t)))

    return index


def image_gradients(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def edge_l1(pred, target):
    px, py = image_gradients(pred)
    tx, ty = image_gradients(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def motion_soft_mask(delta):
    mag = delta.abs().mean(dim=1, keepdim=True)
    return torch.sigmoid((mag - 0.035) * 60.0)


def motion_weight_map(delta):
    mag = delta.abs().mean(dim=1, keepdim=True)

    denom = (
        mag.flatten(1)
        .amax(dim=1, keepdim=True)
        .view(-1, 1, 1, 1)
        .clamp_min(1e-6)
    )

    normalized = mag / denom
    return 1.0 + MOTION_FOCUS * normalized


def weighted_mse(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()


def motion_centroid_target(delta):
    B, _, H, W = delta.shape
    mass = delta.abs().mean(dim=1) + 1e-6

    xs = torch.linspace(
        -1.0, 1.0, W,
        device=delta.device,
        dtype=delta.dtype,
    ).view(1, 1, W)

    ys = torch.linspace(
        -1.0, 1.0, H,
        device=delta.device,
        dtype=delta.dtype,
    ).view(1, H, 1)

    denom = mass.sum(dim=(1, 2)).clamp_min(1e-6)

    cx = (mass * xs).sum(dim=(1, 2)) / denom
    cy = (mass * ys).sum(dim=(1, 2)) / denom

    return torch.stack([cx, cy], dim=1)


def full_losses(outputs, frame0, frame_t):
    delta = outputs["delta"]

    mask_target = motion_soft_mask(delta)

    dynamic_weighted = weighted_mse(
        outputs["delta_recon"],
        delta,
        motion_weight_map(delta),
    )

    dynamic_edge = edge_l1(
        outputs["delta_recon"],
        delta,
    )

    frame_mse = F.mse_loss(
        outputs["frame_recon"],
        frame_t,
    )

    frame_edge = edge_l1(
        outputs["frame_recon"],
        frame_t,
    )

    mask_loss = F.binary_cross_entropy_with_logits(
        outputs["motion_mask_logits"],
        mask_target,
    )

    background_loss = (
        (1.0 - mask_target)
        * outputs["delta_recon"].abs().mean(
            dim=1,
            keepdim=True,
        )
    ).mean()

    centroid_target = motion_centroid_target(delta)

    centroid_loss = F.smooth_l1_loss(
        outputs["motion_centroid"],
        centroid_target,
    )

    if outputs.get("cached_static_mse") is not None:
        static_aux = outputs["cached_static_mse"].mean()
    else:
        static_aux = F.mse_loss(
            outputs["static_aux_recon"],
            frame0,
        )

    total = (
        STATIC_AUX_WEIGHT * static_aux
        + DYNAMIC_LOSS_WEIGHT
        * (
            dynamic_weighted
            + DYNAMIC_EDGE_WEIGHT * dynamic_edge
        )
        + FRAME_LOSS_WEIGHT
        * (
            frame_mse
            + FRAME_EDGE_WEIGHT * frame_edge
        )
        + MASK_LOSS_WEIGHT * mask_loss
        + BACKGROUND_LOSS_WEIGHT * background_loss
        + CENTROID_LOSS_WEIGHT * centroid_loss
    )

    return {
        "total": total,
        "static_aux": static_aux,
        "dynamic": dynamic_weighted,
        "frame": frame_mse,
        "mask": mask_loss,
        "background": background_loss,
        "centroid": centroid_loss,
    }


def rgb_to_device(x, device):
    x = x.to(device, non_blocking=True)
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    sums = {
        "total": 0.0,
        "static_aux": 0.0,
        "dynamic": 0.0,
        "frame": 0.0,
        "mask": 0.0,
        "background": 0.0,
        "centroid": 0.0,
    }

    count = 0

    for batch in loader:
        frame0 = rgb_to_device(
            batch["frame0"], device
        )
        frame_t = rgb_to_device(
            batch["frame_t"], device
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(frame0, frame_t)
            losses = full_losses(
                outputs,
                frame0,
                frame_t,
            )

        n = int(frame0.shape[0])

        for key in sums:
            sums[key] += float(losses[key].item()) * n

        count += n

    return {
        key: value / max(count, 1)
        for key, value in sums.items()
    }
