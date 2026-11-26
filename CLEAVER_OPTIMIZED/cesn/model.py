"""Context Echo State Network, configuration C161 (CESN-Vision).

Context c = [pain one-hot (2), z_static (128, globally scaled), grasp XY (2, in [-1, 1])]
Drive     = c @ Wc + bias + r @ W + y_prev @ Wf          (no explicit time)
Readout   = [1, one-hot, XY, r] -> z_dynamic(t)
Training is teacher forced (y_prev = observed z_dynamic). Context inversion
searches (label, XY) that best explain an observed z_dynamic prefix.

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


# ------------------------------------------------------------------ context inversion
def invert(model, d, ids, n, starts, lo, hi, scale, max_iter=cfg.INVERSION_MAX_ITER, obs_every=1):
    """For each trajectory, find (label, XY) minimising the error on the observed steps
    of the first n steps (every obs_every-th step is a real camera observation)."""
    device = lo.device
    B, R, T = len(ids), len(starts), d["yn"].shape[1]
    obs, static, full = d["yn"][ids, :n], d["static_n"][ids], d["yn"][ids]
    labs = torch.arange(2, device=device)[None, :, None].expand(B, 2, R).reshape(-1)
    S = static.shape[-1]
    ss = static[:, None, None, :].expand(B, 2, R, S).reshape(B * 2 * R, S)
    yy = obs[:, None, None].expand(B, 2, R, n, obs.shape[-1]).reshape(-1, n, obs.shape[-1])
    raw = torch.atanh(starts.clamp(-0.999, 0.999))[None, None].expand(B, 2, R, 2).reshape(-1, 2).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([raw], lr=1.0, max_iter=max_iter, max_eval=2 * max_iter, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    best_loss = torch.full((len(raw),), float("inf"), device=device)
    best_xy = torch.zeros_like(raw)
    calls = 0
    seen = torch.tensor([model.observed(t, obs_every) for t in range(n)], device=device, dtype=torch.float32)

    def losses():
        xy = raw.tanh()
        p = model.predict(model.context(labs, ss, xy), n, yy, obs_every)
        return ((p - yy).square().mean(2) * seen).sum(1) / seen.sum(), xy

    def remember(loss, xy):
        with torch.no_grad():
            better = loss.detach() < best_loss
            best_loss[better] = loss.detach()[better]
            best_xy[better] = xy.detach()[better]

    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        loss, xy = losses()
        remember(loss, xy)
        total = loss.sum()
        total.backward()
        calls += 1
        return total

    sync(device)
    tick = time.perf_counter()
    opt.step(closure)
    with torch.no_grad():
        loss, xy = losses()
        remember(loss, xy)
        calls += 1
        bl, bx = best_loss.view(B, 2, R), best_xy.view(B, 2, R, 2)
        v, j = bl.min(-1)
        x = bx.gather(2, j[:, :, None, None].expand(B, 2, 1, 2)).squeeze(2)
        pred = v.argmin(1)
        chosen = x[torch.arange(B, device=device), pred]
        sync(device)
        elapsed = (time.perf_counter() - tick) * 1000
        forecast = model.predict(model.context(pred, static, chosen), T, obs, obs_every)
        if n < T:
            future = (forecast[:, n:].double() - full[:, n:].double()).square().mean((1, 2)).sqrt() * scale
        else:
            future = torch.full((B,), float("nan"), device=device)
        world = (lo + (chosen + 1) * 0.5 * (hi - lo)).cpu().numpy()
        vv, pl, fut = v.cpu().numpy(), pred.cpu().numpy(), future.cpu().numpy()
    assert np.isfinite(vv).all() and np.isfinite(world).all()
    rows = []
    for j, i in enumerate(ids):
        lab = int(d["label"][i])
        good, bad = float(vv[j, lab]), float(vv[j, 1 - lab])
        rows.append(dict(
            trajectory=d["keys"][i], pair_id=int(d["pair"][i]), frames=n, observed_percent=100 * n / T,
            true_label=lab, pred_label=int(pl[j]), correct=int(lab == pl[j]),
            tie=int(abs(good - bad) <= 1e-10 * max(1, good, bad)),
            true_x_m=float(d["xy"][i, 0]), true_y_m=float(d["xy"][i, 1]),
            pred_x_m=float(world[j, 0]), pred_y_m=float(world[j, 1]),
            xy_error_mm=float(np.linalg.norm(world[j] - d["xy"][i]) * 1000),
            future_rmse=float(fut[j]) if np.isfinite(fut[j]) else None,
            prefix_rmse=float(math.sqrt(min(vv[j])) * scale), margin=(bad - good) / max(good + bad, 1e-12),
            safe_loss=float(vv[j, 0]), pain_loss=float(vv[j, 1]),
            first_contact_step=int(d["contact"][i]), precontact=int(n <= d["contact"][i]),
            outside_train_xy_bounds=int(d["outside_bounds"][i]),
            batch_inference_ms=elapsed, amortized_inference_ms=elapsed / B, closure_calls=calls))
    return rows


def summarize(rows):
    out = []

    def mean(rr, k):
        a = [float(r[k]) for r in rr if r[k] is not None]
        return float(np.mean(a)) if a else None

    for n in sorted({r["frames"] for r in rows}):
        allr = [r for r in rows if r["frames"] == n]
        for subset, rr in (("all", allr), ("precontact", [r for r in allr if r["precontact"]])):
            if not rr:
                continue
            safe = [r for r in rr if r["true_label"] == 0]
            pain = [r for r in rr if r["true_label"] == 1]
            xy = [r["xy_error_mm"] for r in rr]
            out.append(dict(
                frames=n, observed_percent=rr[0]["observed_percent"], subset=subset, count=len(rr),
                accuracy_pct=100 * mean(rr, "correct"),
                safe_accuracy_pct=100 * mean(safe, "correct") if safe else None,
                pain_recall_pct=100 * mean(pain, "correct") if pain else None,
                tie_pct=100 * mean(rr, "tie"), xy_mean_mm=float(np.mean(xy)), xy_median_mm=float(np.median(xy)),
                xy_p90_mm=float(np.quantile(xy, 0.9)), future_rmse=mean(rr, "future_rmse"),
                prefix_rmse=mean(rr, "prefix_rmse"), mean_margin=mean(rr, "margin"),
                amortized_inference_ms=mean(rr, "amortized_inference_ms")))
    return out
