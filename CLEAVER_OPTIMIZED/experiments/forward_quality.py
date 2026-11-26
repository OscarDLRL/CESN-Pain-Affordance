"""Is the CESN forward model good enough to be inverted? (validation only)

1. Accuracy with the TRUE context, per phase, against simple references:
     label_mean : mean train trajectory of the same label (ignores XY)
     nearest    : train trajectory with same label + same robot start + nearest goal
   reported as NRMSE on normalised z (same units as cesn oracle).

    python -m CLEAVER_OPTIMIZED.experiments.forward_quality --ae-run v6_ref1
"""
import argparse
import math

import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn

PHASES = {"reach": (0, 111), "close": (111, 141), "lift": (141, cfg.SEQ_LEN), "all": (0, cfg.SEQ_LEN)}
WINDOWS = {"0-23": (0, 23), "0-67": (0, 67), "0-111 (reach)": (0, 111), "0-222 (all)": (0, cfg.SEQ_LEN),
           "111-222 (close+lift)": (111, cfg.SEQ_LEN)}


def nrmse(p, y, a, b):
    return float(np.sqrt(np.mean((p[:, a:b] - y[:, a:b]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-run", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    folder = cfg.OUT / "autoencoder" / a.ae_run
    out = folder / "forward_quality"
    out.mkdir(exist_ok=True)
    data = cesn.load_latents(folder / "latents.h5")
    info, lo, hi = cesn.normalize(data, device)
    with h5py.File(cfg.RGB, "r") as f:
        start = {k: int(f[k].attrs["start_id"]) for k in f if k.startswith("trajectory_")}
    model = cesn.build_c161(device)
    model.fit(data["train"], cfg.RIDGE)
    tr, va = data["train"], data["validation"]
    y_tr, y_va = tr["yn"].cpu().numpy(), va["yn"].cpu().numpy()

    # ---------------- 1. forward accuracy with the true context
    with torch.no_grad():
        c = model.context(torch.as_tensor(va["label"], device=device), va["static_n"], va["xyn"])
        preds = {"CESN teacher-forced": model.predict(c, cfg.SEQ_LEN, va["yn"]).cpu().numpy(),
                 "CESN autonomous": model.predict(c, cfg.SEQ_LEN).cpu().numpy()}
    preds["label_mean"] = np.stack([y_tr[tr["label"] == lab].mean(0) for lab in va["label"]])
    nn = []
    for i, k in enumerate(va["keys"]):
        cand = [j for j, kk in enumerate(tr["keys"]) if tr["label"][j] == va["label"][i] and start[kk] == start[k]]
        j = min(cand, key=lambda j: np.linalg.norm(tr["xy"][j] - va["xy"][i]))
        nn.append(y_tr[j])
    preds["nearest (same label+start)"] = np.stack(nn)
    rows = []
    for name, p in preds.items():
        row = dict(model=name, **{f"nrmse_{ph}": nrmse(p, y_va, *ab) for ph, ab in PHASES.items()})
        rows.append(row)
        print(f"{name:28s} " + "  ".join(f"{ph} {row[f'nrmse_{ph}']:.3f}" for ph in PHASES), flush=True)
    cesn.csv_write(out / "forward_accuracy.csv", rows)



if __name__ == "__main__":
    main()
