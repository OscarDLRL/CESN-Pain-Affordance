"""Final evaluation on the TEST split with the frozen configuration (config.py).

Everything is fitted on train+validation (7 cleaver locations); test = 2 unseen
locations x 9 robot starts x {SAFE, PAIN} = 36 trajectories.

Methods (label = SAFE/PAIN, XY = grasp goal):
  ours         CESN context inversion with the grasp XY from perception (frame 1);
               label = hypothesis with the lower prefix prediction error.
               XY refined with a perception prior (lambda = 1).
  cesn_free    CESN context inversion with label AND XY free (L-BFGS, 9 restarts).
  nearest      nearest training trajectory by prefix distance (label + its XY).
  linear       ridge on [z_static, last z_dynamic, prefix mean of z_dynamic],
               one model per prefix length (label by sign, XY by regression).
Ours is reported with observation every frame (k=1, same information as the
baselines) and every OBS_EVERY frames (deployed setting).

Protocols:
  heldout_grid  fit on train+validation (7 grid positions), test = 2 held-out grid positions
                -> outputs/final/results/{history, methods, cost}.csv
  offgrid       fit on all 9 grid positions, test = 4 off-grid positions (interpolation)
                -> outputs/final/interpolation/results/{history, methods}.csv

    python -m CLEAVER_OPTIMIZED.experiments.final_evaluation [--protocol offgrid]
"""
import argparse
import time

import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn
from CLEAVER_OPTIMIZED.detector import deployment_perception, fit_deployment_cesn
from CLEAVER_OPTIMIZED.experiments.perceived_xy_inversion import fixed_losses, prior_invert

import h5py

OUTS = {"heldout_grid": cfg.OUT / "final" / "results", "offgrid": cfg.OUT / "final" / "interpolation" / "results"}
SOURCES = {"heldout_grid": cfg.RGB, "offgrid": cfg.OFFGRID}
SIMS = {"heldout_grid": cfg.OUT / "sim_closed_loop", "offgrid": cfg.OUT / "sim_closed_loop_offgrid"}
FRAMES = [1, 2, 3, 5, 8, 10, 12, 15, 23, 34, 45, 67, 90, 111, 140, 170, 222]
PRIOR_LAMBDA = 1.0
RIDGE_BASELINE = 10.0


def sync():
    torch.cuda.synchronize()


def xy_err_mm(xy_pred_m, xy_true_m):
    return np.linalg.norm(np.asarray(xy_pred_m) - np.asarray(xy_true_m), axis=1) * 1000


def denorm(xyn, lo, hi):
    return (lo + (xyn + 1) * 0.5 * (hi - lo)).cpu().numpy()


def linear_features(d, n):
    y = d["yn"][:, :n].cpu().numpy()
    return np.c_[d["static_n"].cpu().numpy(), y[:, -1], y.mean(1), np.ones(len(y))]


def ridge(A, Y, lam):
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=list(OUTS), default="heldout_grid")
    a = ap.parse_args()
    OUT = OUTS[a.protocol]
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(cfg.SEED)
    torch.set_float32_matmul_precision("high")
    model, info, lo, hi, d = fit_deployment_cesn(device, protocol=a.protocol)
    tr, te = d["train"], d["test"]
    scale = info["dynamic_scale"]
    perception = deployment_perception(a.protocol)
    with h5py.File(SOURCES[a.protocol], "r") as f:
        xy_p = np.stack([perception.goal_xy(f[k]["rgb"][cfg.REFERENCE_FRAME])[0] for k in te["keys"]])
    xyn_p = torch.as_tensor(2 * (xy_p - lo.cpu().numpy()) / (hi - lo).cpu().numpy() - 1, device=device, dtype=torch.float32)
    labels = te["label"]
    true_xy = te["xy"]
    starts = torch.tensor(np.random.default_rng(cfg.SEED).uniform(-0.95, 0.95, (cfg.INVERSION_RESTARTS, 2)),
                          device=device, dtype=torch.float32)
    starts[0] = 0
    ids = list(range(len(te["keys"])))
    contact = te["contact"][labels == 1]
    print(f"[{a.protocol}] fit on {len(tr['keys'])} | test: {len(ids)} trajectories | perception XY error {xy_err_mm(xy_p, true_xy).mean():.1f} mm | "
          f"PAIN first edge contact step median {int(np.median(contact))}", flush=True)

    hist, methods = [], []
    for n in FRAMES:
        # ---- ours (k = 1 and deployed k)
        for k in (1, cfg.OBS_EVERY):
            L = fixed_losses(model, te, xyn_p, n, k)
            pred = L.argmin(1).cpu().numpy()
            p_lab, p_xy = prior_invert(model, te, xyn_p, n, PRIOR_LAMBDA, k)
            row = dict(frames=n, obs_every=k, accuracy_pct=100 * float(np.mean(pred == labels)),
                       pain_recall_pct=100 * float(np.mean(pred[labels == 1] == 1)),
                       safe_accuracy_pct=100 * float(np.mean(pred[labels == 0] == 0)),
                       xy_perception_mm=float(xy_err_mm(xy_p, true_xy).mean()),
                       xy_refined_mm=float(xy_err_mm(denorm(p_xy, lo, hi), true_xy).mean()))
            hist.append(row)
            if k == 1:
                methods.append(dict(method="ours", frames=n, accuracy_pct=row["accuracy_pct"], xy_mm=row["xy_refined_mm"]))
        # ---- CESN free inversion
        rr = cesn.invert(model, te, ids, n, starts, lo, hi, scale)
        methods.append(dict(method="cesn_free", frames=n, accuracy_pct=100 * float(np.mean([r["correct"] for r in rr])),
                            xy_mm=float(np.mean([r["xy_error_mm"] for r in rr]))))
        # ---- nearest neighbour on the prefix
        Yt, Yr = te["yn"][:, :n], tr["yn"][:, :n]
        dist = torch.cdist(Yt.reshape(len(Yt), -1), Yr.reshape(len(Yr), -1))
        j = dist.argmin(1).cpu().numpy()
        methods.append(dict(method="nearest", frames=n, accuracy_pct=100 * float(np.mean(tr["label"][j] == labels)),
                            xy_mm=float(xy_err_mm(tr["xy"][j], true_xy).mean())))
        # ---- linear ridge per prefix length
        W = ridge(linear_features(tr, n), np.c_[2 * tr["label"] - 1, tr["xy"]], RIDGE_BASELINE)
        P = linear_features(te, n) @ W
        methods.append(dict(method="linear", frames=n, accuracy_pct=100 * float(np.mean((P[:, 0] > 0) == labels)),
                            xy_mm=float(xy_err_mm(P[:, 1:], true_xy).mean())))
        print(f"frames {n:3d} | " + " | ".join(f"{m['method']} {m['accuracy_pct']:5.1f}% {m['xy_mm']:5.1f}mm"
                                             for m in methods if m["frames"] == n), flush=True)
    cesn.csv_write(OUT / "history.csv", hist)
    cesn.csv_write(OUT / "methods.csv", methods)
    if a.protocol != "heldout_grid":  # cost does not depend on the protocol
        print("DONE", OUT, flush=True)
        return

    # ---- computational cost: one trajectory, one decision, latents given (GPU, wall clock)
    cost = []
    one = {k: (v[:1] if not isinstance(v, list) else v[:1]) for k, v in te.items()}

    def timed(fn, reps=10):
        fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        return (time.perf_counter() - t0) * 1000 / reps

    for n in (23, 111):
        cost.append(dict(method="ours (batch recompute)", frames=n, ms=timed(lambda: fixed_losses(model, one, xyn_p[:1], n, 1))))
        cost.append(dict(method="ours + XY prior refinement", frames=n,
                         ms=timed(lambda: prior_invert(model, one, xyn_p[:1], n, PRIOR_LAMBDA, 1), reps=3)))
        cost.append(dict(method="cesn_free (L-BFGS)", frames=n,
                         ms=timed(lambda: cesn.invert(model, one, [0], n, starts, lo, hi, scale), reps=3)))
        Yr = tr["yn"][:, :n].reshape(len(tr["yn"]), -1)
        cost.append(dict(method="nearest", frames=n, ms=timed(lambda: torch.cdist(one["yn"][:, :n].reshape(1, -1), Yr).argmin())))
        W = ridge(linear_features(tr, n), np.c_[2 * tr["label"] - 1, tr["xy"]], RIDGE_BASELINE)
        cost.append(dict(method="linear", frames=n, ms=timed(lambda: linear_features(one, n) @ W)))
    sim = SIMS[a.protocol] / "summary.json"
    if sim.exists():
        import json
        s = json.loads(sim.read_text())
        cost.append(dict(method="ours online (per control step)", frames=-1, ms=s["ms_per_step"]))
        cost.append(dict(method="ours online (per observation, incl. AE)", frames=-1, ms=s["ms_per_observation"]))
    cesn.csv_write(OUT / "cost.csv", cost)
    for c in cost:
        print(f"COST {c['method']:42s} frames={c['frames']:4d} {c['ms']:9.3f} ms", flush=True)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
