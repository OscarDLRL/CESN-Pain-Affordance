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
                                                  reference_frame=cfg.REFERENCE_FRAME, frames=a.frames,
                                                  restarts=cfg.INVERSION_RESTARTS, max_iter=cfg.INVERSION_MAX_ITER))
    cesn.atomic_json(out / "normalization.json", info)

    model = cesn.build_c161(device)
    cesn.sync(device)
    tick = time.perf_counter()
    model.fit(data["train"], cfg.RIDGE)
    cesn.sync(device)
    fit_seconds = time.perf_counter() - tick
    torch.save(dict(model.state_dict(), normalization=info, parameters=params), out / "cesn.pt")

    starts = torch.tensor(np.random.default_rng(cfg.SEED).uniform(-0.95, 0.95, (cfg.INVERSION_RESTARTS, 2)),
                          device=device, dtype=torch.float32)
    starts[0] = 0
    summaries, oracles = [], []
    for split in a.splits:
        d = data[split]
        o = model.oracle(d)
        oracles.append(dict(split=split, fit_seconds=fit_seconds, **o))
        print("ORACLE", split, o, flush=True)
        rows = []
        for k in a.frames:
            for i in range(0, len(d["keys"]), a.trajectory_batch):
                ids = list(range(i, min(i + a.trajectory_batch, len(d["keys"]))))
                rows.extend(cesn.invert(model, d, ids, k, starts, lo, hi, scale))
            rr = [r for r in rows if r["frames"] == k]
            print(f"{split} frames={k:3d} accuracy={100 * np.mean([r['correct'] for r in rr]):5.1f}% "
                  f"XY={np.mean([r['xy_error_mm'] for r in rr]):6.2f} mm", flush=True)
        cesn.csv_write(out / f"{split}_per_trajectory.csv", rows)
        for r in cesn.summarize(rows):
            summaries.append(dict(split=split, **r, future_nrmse=None if r["future_rmse"] is None else r["future_rmse"] / scale))
        cesn.csv_write(out / "summary.csv", summaries)
        cesn.csv_write(out / "oracle.csv", oracles)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
