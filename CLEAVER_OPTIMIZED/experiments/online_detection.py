"""Offline replay of an online pain detector (validation only).

At every decision step n (every --decide-every frames) the CESN inverts the context
from the causal prefix z[0:n]. Only every k-th frame is a real camera observation;
between them the CESN feeds back its own prediction. A stop is triggered when the
PAIN score exceeds tau on `--consecutive` consecutive decisions.

Replay is exact for detection: frames before the stop are identical to those a
closed-loop run would see. Whether stopping avoids contact needs the simulator.

    python -m CLEAVER_OPTIMIZED.experiments.online_detection --ae-run v6_ref1
"""
import argparse
import time

import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn

TAUS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def pain_score(row):
    """> 0 means PAIN explains the prefix better; scale-free in [-1, 1]."""
    s, p = row["safe_loss"], row["pain_loss"]
    return (s - p) / max(s + p, 1e-12)


def first_alarm(scores, steps, tau, consecutive):
    run = 0
    for n, s in zip(steps, scores):
        run = run + 1 if s > tau else 0
        if run >= consecutive:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-run", required=True)
    ap.add_argument("--obs-every", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--decide-every", type=int, default=5)
    ap.add_argument("--max-step", type=int, default=150)
    ap.add_argument("--consecutive", type=int, default=2)
    a = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(cfg.SEED)
    torch.set_float32_matmul_precision("high")
    folder = cfg.OUT / "autoencoder" / a.ae_run
    out = folder / "online_detection"
    out.mkdir(exist_ok=True)
    data = cesn.load_latents(folder / "latents.h5")
    info, lo, hi = cesn.normalize(data, device)
    scale = info["dynamic_scale"]
    d = data["validation"]
    ids = list(range(len(d["keys"])))
    starts = torch.tensor(np.random.default_rng(cfg.SEED).uniform(-0.95, 0.95, (cfg.INVERSION_RESTARTS, 2)),
                          device=device, dtype=torch.float32)
    starts[0] = 0
    print("validation contact step (PAIN): min", int(d["contact"].min()), "max", int(d["contact"].max()), flush=True)

    trace, summary = [], []
    for k in a.obs_every:
        # decisions only right after a real observation
        steps = [n for n in range(a.decide_every, a.max_step + 1, a.decide_every)]
        for train_mode in (["teacher_forced"] if k == 1 else ["teacher_forced", "matched"]):
            model = cesn.build_c161(device)
            model.fit(d_train := data["train"], cfg.RIDGE, obs_every=1 if train_mode == "teacher_forced" else k)
            oracle = model.oracle(d)
            scores = {i: [] for i in ids}
            ms = []
            for n in steps:
                cesn.sync(device)
                t0 = time.perf_counter()
                rows = cesn.invert(model, d, ids, n, starts, lo, hi, scale, obs_every=k)
                cesn.sync(device)
                ms.append((time.perf_counter() - t0) * 1000 / len(ids))
                for i, r in zip(ids, rows):
                    s = pain_score(r)
                    scores[i].append(s)
                    trace.append(dict(obs_every=k, train=train_mode, trajectory=r["trajectory"], label=r["true_label"],
                                      step=n, contact_step=int(d["contact"][i]), pain_score=s, xy_error_mm=r["xy_error_mm"]))
            print(f"k={k:2d} {train_mode:14s} oracle={oracle} ms/traj/decision={np.mean(ms):.0f}", flush=True)
            for tau in TAUS:
                lead, missed, false_alarm = [], 0, 0
                for i in ids:
                    alarm = first_alarm(scores[i], steps, tau, a.consecutive)
                    if d["label"][i] == 1:
                        if alarm is None or alarm > d["contact"][i]:
                            missed += 1
                        else:
                            lead.append(int(d["contact"][i]) - alarm)
                    elif alarm is not None:
                        false_alarm += 1
                n_pain = int((d["label"] == 1).sum())
                n_safe = len(ids) - n_pain
                row = dict(obs_every=k, train=train_mode, tau=tau, pain_detected_before_contact_pct=100 * len(lead) / n_pain,
                           safe_false_alarm_pct=100 * false_alarm / n_safe,
                           median_lead_frames=float(np.median(lead)) if lead else None,
                           min_lead_frames=int(min(lead)) if lead else None,
                           ms_per_decision=float(np.mean(ms)), autonomous_nrmse=oracle["autonomous_nrmse"],
                           teacher_forced_nrmse=oracle["teacher_forced_nrmse"])
                summary.append(row)
                print("   ", {k_: (round(v, 1) if isinstance(v, float) else v) for k_, v in row.items()
                              if k_ not in ("obs_every", "train", "autonomous_nrmse", "teacher_forced_nrmse", "ms_per_decision")}, flush=True)
            cesn.csv_write(out / "summary.csv", summary)
            cesn.csv_write(out / "trace.csv", trace)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
