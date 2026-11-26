"""Fit CESN C161 on exported latents and evaluate context inversion.

    python -m CLEAVER_OPTIMIZED.cesn.train --ae-run v6_ref1
    python -m CLEAVER_OPTIMIZED.cesn.train --ae-run v6_ref1 --splits validation test
-> outputs/autoencoder/<ae-run>/cesn_c161/
"""
import argparse
import json
import time

import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn
from CLEAVER_OPTIMIZED.perception import GoalPerception
from CLEAVER_OPTIMIZED.experiments.discrete_inference import history, perceived_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-run", required=True, help="folder under outputs/autoencoder/ holding latents.h5")
    ap.add_argument("--latents", default="latents.h5")
    ap.add_argument("--splits", nargs="+", default=["validation"], choices=["validation", "test"])
    ap.add_argument("--frames", nargs="+", type=int, default=cfg.INVERSION_FRAMES)
    ap.add_argument("--trajectory-batch", type=int, default=3)
    a = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(cfg.SEED)
    torch.set_float32_matmul_precision("high")
    folder = cfg.OUT / "autoencoder" / a.ae_run
    lat = folder / a.latents
    out = folder / "cesn_c161"
    out.mkdir(exist_ok=True)

    data = cesn.load_latents(lat)
    info, lo, hi = cesn.normalize(data, device)
    scale = info["dynamic_scale"]
    params = dict(reservoir=cfg.RESERVOIR_SIZE, radius=cfg.SPECTRAL_RADIUS, leak=cfg.LEAK_RATE,
                  feedback_gain=cfg.FEEDBACK_GAIN, ridge=cfg.RIDGE, seed=cfg.SEED)
    cesn.atomic_json(out / "manifest.json", dict(parameters=params, latent_sha256=cesn.digest(lat),
                                                  reference_frame=cfg.REFERENCE_FRAME, frames=a.frames))
    cesn.atomic_json(out / "normalization.json", info)

    model = cesn.build_c161(device)
    cesn.sync(device)
    tick = time.perf_counter()
    model.fit(data["train"], cfg.RIDGE)
    cesn.sync(device)
    fit_seconds = time.perf_counter() - tick
    torch.save(dict(model.state_dict(), normalization=info, parameters=params), out / "cesn.pt")

    perception=GoalPerception.fit(pairs=cfg.TRAIN,folder=out/'perception',note='train only')
    summaries,oracles=[],[]
    for split in a.splits:
        d=data[split]; o=model.oracle(d); oracles.append(dict(split=split,fit_seconds=fit_seconds,**o))
        xy,_=perceived_xy(d,perception,lo,hi); L=history(model,d,xy,1).cpu().numpy()
        for n in a.frames:
            summaries.append(dict(split=split,frames=n,accuracy_pct=100*float(np.mean(L[:,n-1].argmin(1)==d['label']))))
    cesn.csv_write(out/'summary.csv',summaries); cesn.csv_write(out/'oracle.csv',oracles)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
