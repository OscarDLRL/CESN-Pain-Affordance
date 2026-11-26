"""Context inversion with the grasp XY given by perception (frame REFERENCE_FRAME).

Leave-one-location-out CV over train+validation (test untouched). Per fold the knife
homography is fitted on the fold's training locations only.
Modes:
  free  : current inversion, (label, XY) by L-BFGS with restarts
  fixed : XY = perceived; only SAFE vs PAIN is compared (2 CESN rollouts, no optimiser)
  prior : L-BFGS from the perceived XY with penalty lam * |XY - XY_perceived|^2
Also: online detection (observation every k frames) with the fixed mode.

    python -m CLEAVER_OPTIMIZED.experiments.perceived_xy_inversion --ae-run v6_ref1
"""
import argparse

import cv2
import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as cesn
from CLEAVER_OPTIMIZED.experiments import knife_localization as kl
from CLEAVER_OPTIMIZED.experiments.forward_sweep_cv import build, pooled_folds

WINDOWS = [5, 10, 12, 23, 45, 67, 111, 222]
TAUS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def masked_loss(model, c, n, obs, obs_every):
    seen = torch.tensor([model.observed(t, obs_every) for t in range(n)], device=obs.device, dtype=torch.float32)
    p = model.predict(c, n, obs[:, :n], obs_every)
    return ((p - obs[:, :n]).square().mean(2) * seen).sum(1) / seen.sum()


@torch.no_grad()
def fixed_losses(model, d, xyn_p, n, obs_every):
    """[B, 2] prefix loss for SAFE / PAIN at the perceived XY."""
    B = len(d["keys"])
    labs = torch.arange(2, device=xyn_p.device).repeat_interleave(B)
    c = model.context(labs, d["static_n"].repeat(2, 1), xyn_p.repeat(2, 1))
    return masked_loss(model, c, n, d["yn"].repeat(2, 1, 1), obs_every).view(2, B).T


def prior_invert(model, d, xyn_p, n, lam, obs_every, max_iter=cfg.INVERSION_MAX_ITER):
    B = len(d["keys"])
    labs = torch.arange(2, device=xyn_p.device).repeat_interleave(B)
    static = d["static_n"].repeat(2, 1)
    anchor = xyn_p.repeat(2, 1)
    obs = d["yn"].repeat(2, 1, 1)
    xy = anchor.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([xy], lr=1.0, max_iter=max_iter, max_eval=2 * max_iter, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def total():
        return masked_loss(model, model.context(labs, static, xy), n, obs, obs_every) + lam * (xy - anchor).square().sum(1)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = total().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss = total().view(2, B).T
        pred = loss.argmin(1)
        chosen = xy.detach().view(2, B, 2)[pred, torch.arange(B, device=xy.device)]
    return pred, chosen


def to_mm(xyn, lo, hi):
    return (lo + (xyn + 1) * 0.5 * (hi - lo)) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-run", required=True)
    ap.add_argument("--leak", type=float, default=0.015)
    ap.add_argument("--radius", type=float, default=0.6)
    ap.add_argument("--ridge", type=float, default=30.0)
    ap.add_argument("--fb-gain", type=float, default=1.0)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.1, 1.0])
    ap.add_argument("--obs-every", type=int, default=5)
    a = ap.parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    folder = cfg.OUT / "autoencoder" / a.ae_run
    out = folder / "perceived_xy_inversion"
    out.mkdir(exist_ok=True)

    with h5py.File(cfg.RGB, "r") as f:
        k0 = next(k for k in sorted(f) if k.startswith("trajectory_") and int(f[k].attrs["pair_id"]) == cfg.TRAIN[0])
        tmpl = kl.template_from(f[k0]["rgb"][0], f[k0]["rgb"][cfg.REFERENCE_FRAME])
    percept = {r["key"]: r for r in kl.collect(cfg.RGB, tmpl)}
    starts = torch.tensor(np.random.default_rng(cfg.SEED).uniform(-0.95, 0.95, (cfg.INVERSION_RESTARTS, 2)),
                          device=device, dtype=torch.float32)
    starts[0] = 0

    rows, online = [], []
    for obj, data in pooled_folds(folder / "latents.h5"):
        info, lo, hi = cesn.normalize(data, device)
        scale = info["dynamic_scale"]
        H = kl.fit_h([percept[k] for k in data["train"]["keys"]])
        d = data["validation"]
        xy_p = kl.apply_h(H, [percept[k]["px"] for k in d["keys"]])
        xyn_p = torch.as_tensor(2 * (xy_p - lo.cpu().numpy()) / (hi - lo).cpu().numpy() - 1, device=device, dtype=torch.float32)
        true_mm = d["xy"] * 1000
        perc_err = float(np.linalg.norm(xy_p * 1000 - true_mm, axis=1).mean())
        model = build(device, 1.0, a.fb_gain, a.leak, a.radius)
        model.fit(data["train"], a.ridge)
        labels = torch.as_tensor(d["label"], device=device)
        for n in WINDOWS:
            for k in (1, a.obs_every):
                # fixed
                L = fixed_losses(model, d, xyn_p, n, k)
                acc = float((L.argmin(1) == labels).float().mean() * 100)
                rows.append(dict(object=obj, frames=n, obs_every=k, mode="fixed", accuracy=acc, xy_mm=perc_err))
                # prior
                for lam in a.lams:
                    pred, xy = prior_invert(model, d, xyn_p, n, lam, k)
                    err = float(np.linalg.norm(to_mm(xy, lo, hi).cpu().numpy() - true_mm, axis=1).mean())
                    rows.append(dict(object=obj, frames=n, obs_every=k, mode=f"prior{lam:g}",
                                     accuracy=float((pred == labels).float().mean() * 100), xy_mm=err))
                # free (current)
                rr = cesn.invert(model, d, list(range(len(d["keys"]))), n, starts, lo, hi, scale, obs_every=k)
                rows.append(dict(object=obj, frames=n, obs_every=k, mode="free",
                                 accuracy=100 * float(np.mean([r["correct"] for r in rr])),
                                 xy_mm=float(np.mean([r["xy_error_mm"] for r in rr]))))
        # online detection, fixed mode, decisions after each real observation
        steps = list(range(a.obs_every, 151, a.obs_every))
        scores = np.stack([(lambda L: ((L[:, 0] - L[:, 1]) / (L.sum(1) + 1e-12)).cpu().numpy())(
            fixed_losses(model, d, xyn_p, n, a.obs_every)) for n in steps], 1)  # [B, steps]
        for i in range(len(d["keys"])):
            online.append(dict(object=obj, label=int(d["label"][i]), contact=int(d["contact"][i]), scores=scores[i]))
        print(f"fold object {obj}: perception {perc_err:.1f} mm", flush=True)

    cesn.csv_write(out / "per_fold.csv", rows)
    print("\nMean over 7 held-out locations (accuracy % / XY mm):")
    modes = ["free", "fixed"] + [f"prior{lam:g}" for lam in a.lams]
    for k in (1, a.obs_every):
        print(f"  observation every {k} frame(s)")
        for m in modes:
            cells = []
            for n in WINDOWS:
                rr = [r for r in rows if r["mode"] == m and r["frames"] == n and r["obs_every"] == k]
                cells.append(f"{n}:{np.mean([r['accuracy'] for r in rr]):5.1f}%/{np.mean([r['xy_mm'] for r in rr]):4.0f}")
            print(f"    {m:9s} " + "  ".join(cells))

    print(f"\nOnline detection, fixed XY, observation every {a.obs_every} frames, 2 consecutive decisions:")
    summ = []
    for tau in TAUS:
        lead, miss, fa = [], 0, 0
        for o in online:
            run, alarm = 0, None
            for n, s in zip(steps, o["scores"]):
                run = run + 1 if s > tau else 0
                if run >= 2:
                    alarm = n
                    break
            if o["label"] == 1:
                if alarm is None or alarm > o["contact"]:
                    miss += 1
                else:
                    lead.append(o["contact"] - alarm)
            elif alarm is not None:
                fa += 1
        n_p = sum(o["label"] == 1 for o in online)
        n_s = len(online) - n_p
        row = dict(tau=tau, detected_pct=100 * len(lead) / n_p, false_alarm_pct=100 * fa / n_s,
                   median_lead=float(np.median(lead)) if lead else None, min_lead=min(lead) if lead else None)
        summ.append(row)
        print(f"  tau={tau:<5} detected {row['detected_pct']:5.1f}%  false alarms {row['false_alarm_pct']:5.1f}%  "
              f"lead median {row['median_lead']}  min {row['min_lead']}", flush=True)
    cesn.csv_write(out / "online_fixed.csv", summ)


if __name__ == "__main__":
    main()
