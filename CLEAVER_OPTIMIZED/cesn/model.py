"""Context Echo State Network, configuration C161 (CESN-Vision).

Context c = [pain one-hot (2), z_static (128, globally scaled), grasp XY (2, in [-1, 1])]
Drive     = c @ Wc + bias + r @ W + y_prev @ Wf          (no explicit time)
Readout   = [1, one-hot, XY, r] -> z_dynamic(t)
Training is teacher forced (y_prev = observed z_dynamic). Context inversion
compares SAFE/PAIN hypotheses given fixed spatial and visual inputs.

Random weights are generated in exactly the same order as the original
C161 implementation, so the same seed reproduces the same reservoir.
"""
import csv
import hashlib
import json
import math
import time

import h5py
import numpy as np
import torch
from scipy.sparse.linalg import eigs

from CLEAVER_OPTIMIZED import config as cfg

SPLITS = ("train", "validation", "test")


# ------------------------------------------------------------------ io helpers
def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ------------------------------------------------------------------ data
def load_latents(path):
    """Group latents by split. Contact frame indices are relative to the sequence start."""
    sets = {name: set(map(int, cfg.SPLIT[name])) for name in SPLITS}
    rows, contacts, pair_xy = [], {}, {}
    with h5py.File(path, "r") as f:
        start = int(f.attrs.get("sequence_start", f.attrs["reference_frame"]))
        assert start == cfg.REFERENCE_FRAME, "latent sequences start at another frame"
        for key in sorted(k for k in f if k.startswith("trajectory_")):
            g = f[key]
            p, lab = int(g.attrs["pair_id"]), int(g.attrs["pain_label"])
            s = np.asarray(g["z_static"], np.float32)
            z = np.asarray(g["z_dynamic"], np.float32)
            xy = np.asarray(g["target_world"], np.float32)[:2]
            imp = np.asarray(g["contact_impulses"])
            assert s.shape == (cfg.STATIC_LATENT_DIM,) and z.shape == (cfg.SEQ_LEN, cfg.DYNAMIC_LATENT_DIM), key
            assert np.isfinite(s).all() and np.isfinite(z).all() and np.isfinite(xy).all(), key
            hit = np.flatnonzero(imp[:, :2].sum(1) > 1e-4)
            assert bool(len(hit)) == bool(lab), f"{key}: contact does not match label"
            if lab:
                contacts[p] = int(hit[0])
            assert (p, lab) not in pair_xy
            pair_xy[p, lab] = xy
            rows.append((key, p, lab, s, z, xy))
    for p in set.union(*sets.values()):
        assert np.allclose(pair_xy[p, 0], pair_xy[p, 1], atol=1e-7, rtol=0), f"pair {p}: SAFE/PAIN goals differ"
    out = {}
    for name, ids in sets.items():
        rr = [r for r in rows if r[1] in ids]
        out[name] = dict(keys=[r[0] for r in rr], pair=np.array([r[1] for r in rr]), label=np.array([r[2] for r in rr]),
                         static=np.stack([r[3] for r in rr]), z=np.stack([r[4] for r in rr]),
                         xy=np.stack([r[5] for r in rr]), contact=np.array([contacts[r[1]] for r in rr]))
    return out


def load_latent_set(path):
    """All trajectories of a latent file as one set (e.g. the off-grid dataset)."""
    rows, contacts = [], {}
    with h5py.File(path, "r") as f:
        start = int(f.attrs.get("sequence_start", f.attrs["reference_frame"]))
        assert start == cfg.REFERENCE_FRAME, "latent sequences start at another frame"
        for key in sorted(k for k in f if k.startswith("trajectory_")):
            g = f[key]
            p, lab = int(g.attrs["pair_id"]), int(g.attrs["pain_label"])
            z = np.asarray(g["z_dynamic"], np.float32)
            assert z.shape == (cfg.SEQ_LEN, cfg.DYNAMIC_LATENT_DIM), key
            hit = np.flatnonzero(np.asarray(g["contact_impulses"])[:, :2].sum(1) > 1e-4)
            assert bool(len(hit)) == bool(lab), f"{key}: contact does not match label"
            if lab:
                contacts[p] = int(hit[0])
            rows.append((key, p, lab, np.asarray(g["z_static"], np.float32), z, np.asarray(g["target_world"], np.float32)[:2]))
    return dict(keys=[r[0] for r in rows], pair=np.array([r[1] for r in rows]), label=np.array([r[2] for r in rows]),
                static=np.stack([r[3] for r in rows]), z=np.stack([r[4] for r in rows]),
                xy=np.stack([r[5] for r in rows]), contact=np.array([contacts[r[1]] for r in rows]))


def normalize(data, device):
    """Train-split statistics. z_static: one global scale (top singular value / sqrt(n))."""
    tr = data["train"]
    s = tr["static"].astype(np.float64)
    sm = s.mean(0)
    ps = max(float(np.linalg.norm(s - sm, ord=2) / np.sqrt(len(s))), 1e-6)
    zm = tr["z"].astype(np.float64).mean((0, 1))
    zs = max(float((tr["z"] - zm).std()), 1e-6)
    lo, hi = tr["xy"].min(0), tr["xy"].max(0)
    assert np.all(hi > lo)

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    for d in data.values():
        d["static_n"] = t((d["static"] - sm) / ps)
        d["yn"] = t((d["z"] - zm) / zs)
        d["xyn"] = t(2 * (d["xy"] - lo) / (hi - lo) - 1)
        d["outside_bounds"] = np.any((d["xy"] < lo - 1e-6) | (d["xy"] > hi + 1e-6), axis=1)
    info = dict(static_mean=sm.tolist(), static_scale=ps, dynamic_mean=zm.tolist(), dynamic_scale=zs,
                xy_min=lo.tolist(), xy_max=hi.tolist())
    return info, t(lo), t(hi)


# ------------------------------------------------------------------ model
def random_weights(n, seed, device, static_dim=128, feedback_dim=128):
    """Unit-radius reservoir + input weights, generated in the original C161 order."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    w *= rng.random((n, n)) < 0.05
    try:
        rad = float(np.max(np.abs(eigs(w, k=1, which="LM", v0=np.ones(n), tol=1e-5, maxiter=5000, return_eigenvectors=False))))
    except Exception:
        rad = float(np.max(np.abs(np.linalg.eigvals(w))))
    if rad <= 0:
        raise RuntimeError("zero reservoir spectral radius")
    w *= 1.0 / rad
    wc = rng.uniform(-0.5, 0.5, (20, n)).astype(np.float32)  # rows 0-1: one-hot, rows 18-19: XY
    wf = rng.uniform(-0.5, 0.5, (feedback_dim, n)).astype(np.float32) / math.sqrt(feedback_dim)
    bias = rng.uniform(-0.1, 0.1, n).astype(np.float32)
    w_static = np.random.default_rng(seed + 1009).uniform(-0.5, 0.5, (static_dim, n)).astype(np.float32)
    win = np.concatenate([wc[:2], w_static, wc[18:20]], 0)
    return {k: torch.tensor(v, device=device) for k, v in dict(W=w, Win=win, Wf=wf, bias=bias).items()}


class CESN:
    def __init__(self, weights, radius, leak, feedback_gain):
        self.W = weights["W"] * radius
        self.Win = weights["Win"]
        self.Wf = weights["Wf"] * feedback_gain
        self.bias = weights["bias"]
        self.leak = float(leak)
        self.n = len(self.bias)
        self.C = 4  # readout sees [one-hot, XY], not z_static
        self.D = 1 + self.C + self.n
        self.readout = None

    @staticmethod
    def context(label, static, xy):
        one = torch.nn.functional.one_hot(label.long(), 2).float()
        return torch.cat([one, static, xy], -1)

    @staticmethod
    def context_out(c):
        return torch.cat([c[:, :2], c[:, -2:]], -1)

    def _step(self, base, r, prev):
        return (1 - self.leak) * r + self.leak * torch.tanh(base + r @ self.W + prev @ self.Wf)

    @staticmethod
    def observed(t, obs_every):
        """A camera observation is available at steps 0, k, 2k, ..."""
        return t % obs_every == 0

    def _readout_parts(self, c):
        q = self.readout
        return q[0] + self.context_out(c) @ q[1:1 + self.C], q[1 + self.C:]

    def features(self, c, y, obs_every=1):
        """Reservoir features [B, T, D]. Feedback = observed z on observed steps,
        the model's own prediction otherwise (requires an existing readout)."""
        B, T, Dy = y.shape
        r, prev = c.new_zeros(B, self.n), c.new_zeros(B, Dy)
        base, ones, co, f = c @ self.Win + self.bias, c.new_ones(B, 1), self.context_out(c), []
        if obs_every > 1:
            const, qr = self._readout_parts(c)
        for t in range(T):
            r = self._step(base, r, prev)
            f.append(torch.cat([ones, co, r], -1))
            prev = y[:, t] if self.observed(t, obs_every) else const + r @ qr
        return torch.stack(f, 1)

    def predict(self, c, T, obs=None, obs_every=1):
        """Feedback = observed z when a camera frame exists (within obs and every
        obs_every steps), own prediction otherwise."""
        B = len(c)
        r, prev = c.new_zeros(B, self.n), c.new_zeros(B, self.Wf.shape[0])
        base = c @ self.Win + self.bias
        const, qr = self._readout_parts(c)
        preds = []
        for t in range(T):
            r = self._step(base, r, prev)
            y = const + r @ qr
            preds.append(y)
            seen = obs is not None and t < obs.shape[1] and self.observed(t, obs_every)
            prev = obs[:, t] if seen else y
        return torch.stack(preds, 1)

    @torch.no_grad()
    def fit(self, d, ridge, obs_every=1, rounds=2):
        """Ridge readout. obs_every > 1: refit on features generated with the model's
        own predictions between observations (removes train/deploy mismatch)."""
        self._fit(d, ridge, 1)
        if obs_every > 1:
            for _ in range(rounds):
                self._fit(d, ridge, obs_every)

    @torch.no_grad()
    def _fit(self, d, ridge, obs_every):
        y = d["yn"]
        c = self.context(torch.as_tensor(d["label"], device=y.device), d["static_n"], d["xyn"])
        G = torch.zeros((self.D, self.D), device=y.device, dtype=torch.float64)
        H = torch.zeros((self.D, y.shape[-1]), device=y.device, dtype=torch.float64)
        for i in range(0, len(y), 8):
            f = self.features(c[i:i + 8], y[i:i + 8], obs_every).reshape(-1, self.D).double()
            G += f.T @ f
            H += f.T @ y[i:i + 8].reshape(-1, y.shape[-1]).double()
        G.diagonal().add_(ridge)
        G[0, 0] -= ridge  # bias column not regularised
        self.readout = torch.linalg.solve(G, H).float()
        assert torch.isfinite(self.readout).all()

    @torch.no_grad()
    def oracle(self, d):
        """Forward quality with the true context (normalised RMSE)."""
        y = d["yn"]
        c = self.context(torch.as_tensor(d["label"], device=y.device), d["static_n"], d["xyn"])
        out = {}
        for name, obs in (("teacher_forced", y), ("autonomous", None)):
            p = self.predict(c, y.shape[1], obs)
            out[f"{name}_nrmse"] = float((p.double() - y.double()).square().mean().sqrt())
            out[f"{name}_nonfinite"] = int((~torch.isfinite(p)).sum())
        return out

    def state_dict(self):
        return dict(W=self.W.cpu(), Win=self.Win.cpu(), Wf=self.Wf.cpu(), bias=self.bias.cpu(),
                    readout=self.readout.cpu(), leak=self.leak)


def build_c161(device, seed=cfg.SEED):
    w = random_weights(cfg.RESERVOIR_SIZE, seed, device, cfg.STATIC_LATENT_DIM, cfg.DYNAMIC_LATENT_DIM)
    return CESN(w, cfg.SPECTRAL_RADIUS, cfg.LEAK_RATE, cfg.FEEDBACK_GAIN)


