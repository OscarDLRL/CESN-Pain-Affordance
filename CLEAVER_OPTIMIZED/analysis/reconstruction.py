"""AE diagnostic: does the reconstruction change pixels that did not change?

static region : |frame_t - reference| < STATIC_TOL  (the reconstruction should leave it alone)
moving region : motion target mask > 0.5
settle region : pixels changed by the cleaver drop between frame 0 and 1, counted
                only while they are static at frame t (the "moving knife corners")
Reports per-frame errors on the validation split and panels for one pair.

    python -m CLEAVER_OPTIMIZED.analysis.reconstruction --checkpoint outputs/autoencoder/v6_ref1/autoencoder.pt
    # old checkpoint trained with frame 0 (from the backup):
    python -m CLEAVER_OPTIMIZED.analysis.reconstruction --checkpoint <path> --reference 0 --out outputs/analysis/old_ref0
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED import data
from CLEAVER_OPTIMIZED.autoencoder.losses import motion_soft_mask
from CLEAVER_OPTIMIZED.autoencoder.model import load_checkpoint
from CLEAVER_OPTIMIZED.cesn.model import csv_write

STATIC_TOL = 0.01


@torch.no_grad()
def evaluate(model, keys, reference, stride, batch=32):
    rows = []
    with h5py.File(cfg.RGB, "r") as f:
        for key in keys:
            rgb = f[key]["rgb"]
            ref = data.to_tensor(rgb[reference:reference + 1], "cuda")
            settled = data.to_tensor(rgb[cfg.REFERENCE_FRAME:cfg.REFERENCE_FRAME + 1], "cuda")
            settle = ((settled - data.to_tensor(rgb[0:1], "cuda")).abs().mean(1) > 0.035)[0]
            ts = np.arange(cfg.REFERENCE_FRAME, cfg.NUM_FRAMES, stride)
            for i in range(0, len(ts), batch):
                t = ts[i:i + batch]
                y = data.to_tensor(rgb[t], "cuda")
                out = model(ref.expand(len(t), -1, -1, -1), y)
                err = (out["frame_recon"].float() - y).abs().mean(1)
                change = (y - ref).abs().mean(1)
                knife_static = settle & ((y - settled).abs().mean(1) < STATIC_TOL)
                static = change < STATIC_TOL
                moving = motion_soft_mask(y - ref)[:, 0] > 0.5
                pred_mask = out["motion_mask"].float()[:, 0]
                for j, tt in enumerate(t):
                    s, m, k = static[j], moving[j], knife_static[j]
                    rows.append(dict(
                        trajectory=key, frame=int(tt),
                        mae=float(err[j].mean()),
                        static_mae=float(err[j][s].mean()) if s.any() else None,
                        static_false_change_pct=float(100 * (err[j][s] > 0.05).float().mean()) if s.any() else None,
                        settle_mae=float(err[j][k].mean()) if k.any() else None,
                        moving_mae=float(err[j][m].mean()) if m.any() else None,
                        mask_on_static_pct=float(100 * (pred_mask[j][s] > 0.5).float().mean()) if s.any() else None,
                        static_pixels=int(s.sum()), settle_pixels=int(k.sum()), moving_pixels=int(m.sum())))
    return rows


@torch.no_grad()
def panels(model, key, reference, frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with h5py.File(cfg.RGB, "r") as f:
        rgb = f[key]["rgb"]
        ref = data.to_tensor(rgb[reference:reference + 1], "cuda").expand(len(frames), -1, -1, -1)
        y = data.to_tensor(rgb[frames], "cuda")
    out = model(ref, y)
    error = (out["frame_recon"].float() - y).abs().mean(1, keepdim=True) * 5  # x5 for visibility
    rows = [("Original", y), ("Target mask", motion_soft_mask(y - ref)), ("Predicted mask", out["motion_mask"]),
            ("Reconstruction", out["frame_recon"]), ("|error| x5", error)]
    fig, ax = plt.subplots(len(rows), len(frames), figsize=(2 * len(frames), 2 * len(rows)))
    for r, (name, img) in enumerate(rows):
        a = img.float().clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        for c in range(len(frames)):
            gray = a.shape[-1] == 1
            ax[r, c].imshow(a[c, :, :, 0] if gray else a[c], cmap="gray" if gray else None, vmin=0, vmax=1)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"frame {frames[c]}", fontsize=8)
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", type=int, default=cfg.REFERENCE_FRAME, help="reference frame fed to the AE")
    ap.add_argument("--out", default=None, help="default: <checkpoint folder>/analysis")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--pair", type=int, default=cfg.VALIDATION[0])
    ap.add_argument("--frames", nargs="+", type=int, default=[10, 26, 60, 94, 114, 150, 201])
    a = ap.parse_args()
    ckpt_path = Path(a.checkpoint)
    out = Path(a.out) if a.out else ckpt_path.parent / "analysis"
    out = out if out.is_absolute() else cfg.BASE / out
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(ckpt_path if ckpt_path.is_absolute() else cfg.BASE / ckpt_path, "cuda")

    keys = data.split_keys()["validation"]
    rows = evaluate(model, keys, a.reference, a.stride)
    csv_write(out / "validation_frames.csv", rows)
    summary = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
               for k in ("mae", "static_mae", "static_false_change_pct", "settle_mae", "moving_mae", "mask_on_static_pct")}
    csv_write(out / "validation_summary.csv", [dict(reference=a.reference, **summary)])
    print("SUMMARY", summary, flush=True)

    with h5py.File(cfg.RGB, "r") as f:
        pair_keys = [k for k in keys if int(f[k].attrs["pair_id"]) == a.pair]
        labels = {k: int(f[k].attrs["label"]) for k in pair_keys}
    for key in pair_keys:
        name = "pain" if labels[key] else "safe"
        panels(model, key, a.reference, a.frames, out / f"pair{a.pair:03d}_{name}.png")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
