"""V6 objective (unchanged weights from the best V6Compact128 run)."""
import torch
import torch.nn.functional as F

STATIC_AUX_WEIGHT = 0.15
DYNAMIC_LOSS_WEIGHT = 2.0
FRAME_LOSS_WEIGHT = 1.0
MASK_LOSS_WEIGHT = 0.75
BACKGROUND_LOSS_WEIGHT = 2.0
CENTROID_LOSS_WEIGHT = 0.40
DYNAMIC_EDGE_WEIGHT = 0.50
FRAME_EDGE_WEIGHT = 0.50
MOTION_FOCUS = 30.0
MASK_THRESHOLD = 0.035
MASK_SHARPNESS = 60.0


def image_gradients(x):
    return x[:, :, :, 1:] - x[:, :, :, :-1], x[:, :, 1:, :] - x[:, :, :-1, :]


def edge_l1(pred, target):
    px, py = image_gradients(pred)
    tx, ty = image_gradients(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def motion_soft_mask(delta):
    """RGB-change target for the mask head (not a physical segmentation)."""
    mag = delta.abs().mean(1, keepdim=True)
    return torch.sigmoid((mag - MASK_THRESHOLD) * MASK_SHARPNESS)


def motion_weight_map(delta):
    # NOTE: normalised by the per-image maximum, so frames with almost no
    # motion get their noise amplified up to 1 + MOTION_FOCUS.
    mag = delta.abs().mean(1, keepdim=True)
    peak = mag.flatten(1).amax(1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return 1.0 + MOTION_FOCUS * mag / peak


def motion_centroid_target(delta):
    _, _, H, W = delta.shape
    mass = delta.abs().mean(1) + 1e-6
    xs = torch.linspace(-1, 1, W, device=delta.device, dtype=delta.dtype).view(1, 1, W)
    ys = torch.linspace(-1, 1, H, device=delta.device, dtype=delta.dtype).view(1, H, 1)
    total = mass.sum((1, 2)).clamp_min(1e-6)
    return torch.stack([(mass * xs).sum((1, 2)) / total, (mass * ys).sum((1, 2)) / total], 1)


def full_losses(out, reference, frame_t):
    delta = out["delta"]
    target_mask = motion_soft_mask(delta)

    dynamic = (motion_weight_map(delta) * (out["delta_recon"] - delta) ** 2).mean()
    dynamic_edge = edge_l1(out["delta_recon"], delta)
    frame = F.mse_loss(out["frame_recon"], frame_t)
    frame_edge = edge_l1(out["frame_recon"], frame_t)
    mask = F.binary_cross_entropy_with_logits(out["motion_mask_logits"], target_mask)
    background = ((1.0 - target_mask) * out["delta_recon"].abs().mean(1, keepdim=True)).mean()
    centroid = F.smooth_l1_loss(out["motion_centroid"], motion_centroid_target(delta))
    if out.get("cached_static_mse") is not None:
        static_aux = out["cached_static_mse"].mean()
    else:
        static_aux = F.mse_loss(out["static_aux_recon"], reference)

    total = (
        STATIC_AUX_WEIGHT * static_aux
        + DYNAMIC_LOSS_WEIGHT * (dynamic + DYNAMIC_EDGE_WEIGHT * dynamic_edge)
        + FRAME_LOSS_WEIGHT * (frame + FRAME_EDGE_WEIGHT * frame_edge)
        + MASK_LOSS_WEIGHT * mask
        + BACKGROUND_LOSS_WEIGHT * background
        + CENTROID_LOSS_WEIGHT * centroid
    )
    return {"total": total, "static_aux": static_aux, "dynamic": dynamic, "frame": frame,
            "mask": mask, "background": background, "centroid": centroid}


def temporal_smoothness(z_prev, z, z_next):
    """|acceleration|^2 / |velocity|^2 of z_dynamic over one frame (same quantity as
    the jitter probe). Velocity is detached, so the model cannot lower it by
    shrinking or speeding up z; it must remove frame-to-frame noise."""
    acc = (z_next - 2 * z + z_prev).float()
    vel = (0.5 * (z_next - z_prev)).detach().float()
    return acc.square().mean() / vel.square().mean().clamp_min(1e-8)


def state_loss(pred, target):
    """MSE on standardised robot state (TCP xyz, gripper width)."""
    return F.mse_loss(pred.float(), target)
