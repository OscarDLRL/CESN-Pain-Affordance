"""Online pain detector for closed-loop use (simulator or robot).

Per episode:
  reset(reference_rgb)  frame REFERENCE_FRAME: z_static (AE) + grasp XY (perception)
  step(rgb or None)     one call per control step. Two CESN reservoirs run in
                        parallel, one per hypothesis (SAFE, PAIN), with the
                        perceived XY. Every OBS_EVERY steps a camera frame is
                        encoded and fed back; in between each reservoir feeds back
                        its own prediction. The prefix error of each hypothesis is
                        accumulated on observed steps only.
Decision (after each observation): score = (L_safe - L_pain) / (L_safe + L_pain);
STOP when score > DETECT_TAU on DETECT_CONSECUTIVE consecutive observations.
Cost per control step: two reservoir updates (+ one AE encode every OBS_EVERY steps).

Deployment model: CESN B fitted on train+validation latents; perception fitted on
train+validation. Test locations are never used to fit anything.
"""
import json
import time

import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED import data
from CLEAVER_OPTIMIZED.autoencoder.model import load_checkpoint
from CLEAVER_OPTIMIZED.cesn import model as cesn
from CLEAVER_OPTIMIZED.perception import GoalPerception

FOLDER = cfg.OUT / "detector"


PROTOCOLS = ("heldout_grid", "offgrid")


def fit_deployment_cesn(device, ae_run=cfg.AE_RUN, protocol="heldout_grid"):
    """CESN B, normalisation from its own fitting data.
    heldout_grid: fit on train+validation (7 grid positions), test = 2 held-out grid positions
    offgrid     : fit on all 9 grid positions (162 trajectories), test = 4 off-grid positions"""
    assert protocol in PROTOCOLS
    lat = cfg.OUT / "autoencoder" / ae_run / "latents.h5"
    split = cesn.load_latents(lat)
    parts = ["train", "validation"] + (["test"] if protocol == "offgrid" else [])
    pooled = {k: (sum((split[p][k] for p in parts), []) if k == "keys" else np.concatenate([split[p][k] for p in parts]))
              for k in split["train"]}
    test = split["test"] if protocol == "heldout_grid" else cesn.load_latent_set(
        cfg.OUT / "autoencoder" / ae_run / "latents_offgrid.h5")
    d = {"train": pooled, "test": test}
    info, lo, hi = cesn.normalize(d, device)
    model = cesn.build_c161(device)
    model.fit(d["train"], cfg.RIDGE)
    folder = FOLDER / protocol
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model.state_dict(), normalization=info), folder / "cesn.pt")
    cesn.atomic_json(folder / "settings.json", dict(
        ae_run=ae_run, protocol=protocol, fitted_trajectories=len(pooled["keys"]), ridge=cfg.RIDGE, leak=cfg.LEAK_RATE, radius=cfg.SPECTRAL_RADIUS,
        feedback_gain=cfg.FEEDBACK_GAIN, obs_every=cfg.OBS_EVERY, tau=cfg.DETECT_TAU,
        consecutive=cfg.DETECT_CONSECUTIVE, latent_sha256=cesn.digest(lat)))
    return model, info, lo, hi, d


def deployment_perception(protocol):
    if protocol == "offgrid":
        return GoalPerception.fit(pairs=range(81), folder=cfg.OUT / "perception" / "all_grid", note="all 9 grid positions")
    return GoalPerception.fit()


class OnlinePainDetector:
    def __init__(self, ae, model, info, lo, hi, perception, device,
                 obs_every=cfg.OBS_EVERY, tau=cfg.DETECT_TAU, consecutive=cfg.DETECT_CONSECUTIVE):
        self.ae, self.m, self.perception, self.device = ae, model, perception, device
        self.obs_every, self.tau, self.consecutive = obs_every, tau, consecutive
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.s_mean, self.s_scale = t(info["static_mean"]), float(info["static_scale"])
        self.z_mean, self.z_scale = t(info["dynamic_mean"]), float(info["dynamic_scale"])
        self.lo, self.hi = lo, hi

    @classmethod
    def build(cls, device=torch.device("cuda"), protocol="heldout_grid"):
        ae, _ = load_checkpoint(cfg.OUT / "autoencoder" / cfg.AE_RUN / "autoencoder.pt", device)
        model, info, lo, hi, _ = fit_deployment_cesn(device, protocol=protocol)
        return cls(ae, model, info, lo, hi, deployment_perception(protocol), device)


    @torch.no_grad()
    def _encode(self, rgb):
        y = data.to_tensor(rgb[None], self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            z = self.ae.encode_dynamic(y - self.ref)
        return (z.float()[0] - self.z_mean) / self.z_scale

    @torch.no_grad()
    def reset(self, reference_rgb, goal_xy=None):
        """goal_xy overrides perception (for ablations only)."""
        self.ref = data.to_tensor(reference_rgb[None], self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            z_static = self.ae.encode_static(self.ref).float()[0]
        static_n = (z_static - self.s_mean) / self.s_scale
        if goal_xy is None:
            goal_xy, self.match_score = self.perception.goal_xy(reference_rgb)
        self.goal_xy = np.asarray(goal_xy, np.float64)
        xyn = 2 * (torch.as_tensor(self.goal_xy, dtype=torch.float32, device=self.device) - self.lo) / (self.hi - self.lo) - 1
        c = self.m.context(torch.arange(2, device=self.device), static_n.expand(2, -1), xyn.expand(2, -1))
        self.base = c @ self.m.Win + self.m.bias
        self.const, self.qr = self.m._readout_parts(c)
        self.r = torch.zeros(2, self.m.n, device=self.device)
        self.prev = torch.zeros(2, self.m.Wf.shape[0], device=self.device)
        self.sq = torch.zeros(2, device=self.device)
        self.n_obs, self.run, self.s = 0, 0, 0
        self.alarm_step, self.history = None, []

    @torch.no_grad()
    def step(self, rgb=None):
        """rgb: frame REFERENCE_FRAME + s (uint8 HxWx3); ignored on non-observation steps."""
        t0 = time.perf_counter()
        self.r = (1 - self.m.leak) * self.r + self.m.leak * torch.tanh(self.base + self.r @ self.m.W + self.prev @ self.m.Wf)
        y = self.const + self.r @ self.qr
        out = dict(step=self.s, observed=False, score=None, alarm=self.alarm_step is not None)
        if rgb is not None and self.s % self.obs_every == 0:
            z = self._encode(rgb)
            self.sq += (y - z).square().mean(1)
            self.n_obs += 1
            self.prev = z.expand(2, -1)
            L = self.sq / self.n_obs
            score = float((L[0] - L[1]) / (L.sum() + 1e-12))
            self.run = self.run + 1 if score > self.tau else 0
            if self.alarm_step is None and self.run >= self.consecutive:
                self.alarm_step = self.s
            out.update(observed=True, score=score, loss_safe=float(L[0]), loss_pain=float(L[1]),
                       alarm=self.alarm_step is not None)
        else:
            self.prev = y
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        out["ms"] = (time.perf_counter() - t0) * 1000
        self.history.append(out)
        self.s += 1
        return out

    def needs_frame(self):
        """True if the next step() is an observation step (render/encode only then)."""
        return self.s % self.obs_every == 0


