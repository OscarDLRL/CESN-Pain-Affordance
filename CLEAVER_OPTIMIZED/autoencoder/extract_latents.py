"""Export z_static / z_dynamic for every trajectory from a trained AE.

Sequences start at cfg.REFERENCE_FRAME, so z_dynamic has cfg.SEQ_LEN rows and
per-frame arrays (contact_impulses, ...) are sliced the same way.

    python -m CLEAVER_OPTIMIZED.autoencoder.extract_latents --run v6_ref1
    -> outputs/autoencoder/v6_ref1/latents.h5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED import data
from CLEAVER_OPTIMIZED.autoencoder.model import load_checkpoint

STATIC_FIELDS = ("target_world", "prepared_tcp", "knife_xy", "start_offset_xy")
PER_FRAME_FIELDS = ("contact_impulses",)


@torch.no_grad()
def export(model, src_path, dst_path, batch=64, reference=None):
    """Sequences always start at cfg.REFERENCE_FRAME; `reference` only changes the
    image the deltas are taken against (ablation of old frame-0 models)."""
    start = cfg.REFERENCE_FRAME
    ref_t = start if reference is None else int(reference)
    tmp = dst_path.with_suffix(".tmp")
    with h5py.File(src_path, "r") as src, h5py.File(tmp, "w") as dst:
        dst.attrs["reference_frame"] = ref_t
        dst.attrs["sequence_start"] = start
        keys = sorted(k for k in src if k.startswith("trajectory_"))
        for i, key in enumerate(keys):
            g = src[key]
            rgb = g["rgb"]
            ref = data.to_tensor(rgb[ref_t:ref_t + 1], "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                z_static = model.encode_static(ref)
            chunks = []
            for t in range(start, len(rgb), batch):
                y = data.to_tensor(rgb[t:t + batch], "cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    chunks.append(model.encode_dynamic(y - ref).float().cpu().numpy())
            z_dynamic = np.concatenate(chunks)
            assert z_dynamic.shape[0] == len(rgb) - start and np.isfinite(z_dynamic).all(), key
            out = dst.create_group(key)
            for name, value in g.attrs.items():
                out.attrs[name] = value
            out.attrs["pain_label"] = int(g.attrs["label"])
            out.create_dataset("z_static", data=z_static[0].float().cpu().numpy())
            out.create_dataset("z_dynamic", data=z_dynamic, compression="gzip")
            for name in STATIC_FIELDS:
                if name in g:
                    out.create_dataset(name, data=g[name][()])
            for name in PER_FRAME_FIELDS:
                if name in g:
                    out.create_dataset(name, data=g[name][start:])
            if (i + 1) % 20 == 0 or i + 1 == len(keys):
                print(f"LATENT {i + 1}/{len(keys)}", flush=True)
    tmp.replace(dst_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder under outputs/autoencoder/")
    ap.add_argument("--dataset", default=str(cfg.RGB))
    ap.add_argument("--name", default="latents.h5")
    ap.add_argument("--checkpoint", default=None, help="default: <run>/autoencoder.pt (e.g. an old backup checkpoint)")
    ap.add_argument("--reference", type=int, default=None, help="override reference image (ablation only)")
    a = ap.parse_args()
    folder = cfg.OUT / "autoencoder" / a.run
    folder.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_checkpoint(Path(a.checkpoint) if a.checkpoint else folder / "autoencoder.pt", "cuda")
    trained_ref = ckpt.get("reference_frame", 0)  # checkpoints older than this repo used frame 0
    if a.reference is None:
        assert trained_ref == cfg.REFERENCE_FRAME, "checkpoint trained with another reference frame: pass --reference"
    dst = folder / a.name
    export(model, Path(a.dataset), dst, reference=a.reference)
    print("DONE", dst, flush=True)


if __name__ == "__main__":
    main()
