"""Final matched CESN experiments for CLEAVER Lightweight128.

Runs E20/E22 baselines plus three final hypotheses:
E24 bounded incremental, E25 initial-state static memory, E26 hybrid absolute/incremental.
No explicit time input. Hyperparameters are selected on validation only;
test is evaluated once after selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED import run_incremental as base

OPTIONS = (1, 1, 1, 1, True, False, False)
EXPERIMENTS = [
    ("E20_Raw128_Absolute", "Absolute (Raw128)"),
    ("E22_Raw128_Increment", "Incremental (Raw128)"),
    ("E24_Bounded_Increment", "Bounded incremental"),
    ("E25_InitialState_Memory", "Initial-state memory"),
    ("E26_Hybrid", "Hybrid"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    # Different experiments can contribute different tuning columns
    # (e.g., clip_multiplier for E24 and alpha_absolute for E26).
    # Build a stable union of all keys and leave non-applicable cells blank.
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def raw_context_weights(common, n: int, seed: int, device):
    w, wc, wf, bias = common
    raw_w = torch.tensor(
        np.random.default_rng(seed + 1009)
        .uniform(-0.5, 0.5, (128, n)).astype(np.float32),
        device=device,
    )
    return w, torch.cat([wc[:2], raw_w, wc[18:20]], dim=0), wf, bias


class BoundedIncrementalCESN(base.CESN):
    """Incremental CESN with per-latent training-quantile increment limits."""

    def __init__(self, *args, **kwargs):
        kwargs["incremental"] = True
        super().__init__(*args, **kwargs)
        self.clip_vector = None
        self.clip_multiplier = 1.0

    @torch.no_grad()
    def fit(self, d, ridge):
        super().fit(d, ridge)
        y = d["yn"]
        increments = torch.cat([y[:, :1], y[:, 1:] - y[:, :-1]], dim=1)
        self.clip_vector = torch.quantile(
            increments.abs().reshape(-1, 128), 0.995, dim=0
        ).clamp_min(1e-4)

    def predict(self, c, T, obs=None):
        B = len(c)
        r = c.new_zeros(B, self.n)
        prev = c.new_zeros(B, 128)
        base_drive = c @ self.Wc + self.bias
        q = self.readout
        constant = q[0] + self.context_out(c) @ q[1:1 + self.C]
        qr = q[1 + self.C:1 + self.C + self.n]
        mod = self.modulation(c)
        preds = []
        limit = self.clip_vector * float(self.clip_multiplier)
        for t in range(T):
            drive = base_drive + r @ self.W + prev @ self.Wf
            r = (1 - self.leak) * r + self.leak * torch.tanh(drive)
            delta = constant + r @ qr
            if mod is not None:
                delta = delta + (r * mod) @ q[-self.n:]
            delta = torch.maximum(torch.minimum(delta, limit), -limit)
            y = prev + delta
            preds.append(y)
            prev = obs[:, t] if obs is not None and t < obs.shape[1] else y
        return torch.stack(preds, dim=1)


class InitialStateMemoryCESN:
    """Raw static image latent is written once into r(0), not re-fed each step."""

    def __init__(self, common, memory_weights, leak: float, memory_gain: float = 1.0):
        w, wc, wf, bias = common
        self.W = w
        self.Wc = torch.cat([wc[:2], wc[-2:]], dim=0)  # class(2) + XY(2)
        self.Wf = wf
        self.bias = bias
        self.Wmem = memory_weights
        self.leak = float(leak)
        self.memory_gain = float(memory_gain)
        self.n = len(bias)
        self.C = 4
        self.D = 1 + self.C + self.n
        self.pca = 128
        self.feedback = True
        self.incremental = False
        self.readout = None

    def context(self, label, static, xy):
        one = torch.nn.functional.one_hot(label.long(), 2).float()
        return torch.cat([one, static[:, :128], xy], dim=-1)

    def context_out(self, c):
        return torch.cat([c[:, :2], c[:, -2:]], dim=-1)

    def initial_state(self, c):
        static = c[:, 2:-2]
        return torch.tanh(self.memory_gain * (static @ self.Wmem))

    @torch.no_grad()
    def features(self, c, y):
        B, T, _ = y.shape
        r = self.initial_state(c)
        prev = c.new_zeros(B, 128)
        class_xy = self.context_out(c)
        base_drive = class_xy @ self.Wc + self.bias
        ones = c.new_ones(B, 1)
        feats = []
        for t in range(T):
            drive = base_drive + r @ self.W + prev @ self.Wf
            r = (1 - self.leak) * r + self.leak * torch.tanh(drive)
            feats.append(torch.cat([ones, class_xy, r], dim=-1))
            prev = y[:, t]
        return torch.stack(feats, dim=1)

    @torch.no_grad()
    def fit(self, d, ridge):
        y = d["yn"]
        c = self.context(torch.as_tensor(d["label"], device=y.device), d["pca"], d["xyn"])
        G = torch.zeros((self.D, self.D), device=y.device, dtype=torch.float64)
        H = torch.zeros((self.D, 128), device=y.device, dtype=torch.float64)
        for i in range(0, len(y), 8):
            f = self.features(c[i:i+8], y[i:i+8]).reshape(-1, self.D).double()
            target = y[i:i+8].reshape(-1, 128).double()
            G += f.T @ f
            H += f.T @ target
        G.diagonal().add_(ridge)
        G[0, 0] -= ridge
        self.readout = torch.linalg.solve(G, H).float()
        assert torch.isfinite(self.readout).all()

    def predict(self, c, T, obs=None):
        B = len(c)
        r = self.initial_state(c)
        prev = c.new_zeros(B, 128)
        class_xy = self.context_out(c)
        base_drive = class_xy @ self.Wc + self.bias
        q = self.readout
        constant = q[0] + class_xy @ q[1:1 + self.C]
        qr = q[1 + self.C:]
        ones = c.new_ones(B, 1)
        preds = []
        for t in range(T):
            drive = base_drive + r @ self.W + prev @ self.Wf
            r = (1 - self.leak) * r + self.leak * torch.tanh(drive)
            feat = torch.cat([ones, class_xy, r], dim=-1)
            y = constant + r @ qr
            preds.append(y)
            prev = obs[:, t] if obs is not None and t < obs.shape[1] else y
        return torch.stack(preds, dim=1)

    def oracle(self, d, scale):
        return base.CESN.oracle(self, d, scale)


class HybridCESN(base.CESN):
    """Shared reservoir and feedback state, with absolute and increment readouts."""

    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        kwargs["incremental"] = False
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.readout_abs = None
        self.readout_inc = None

    @torch.no_grad()
    def fit(self, d, ridge):
        y = d["yn"]
        c = self.context(torch.as_tensor(d["label"], device=y.device), d["pca"], d["xyn"])
        inc = torch.cat([y[:, :1], y[:, 1:] - y[:, :-1]], dim=1)
        G = torch.zeros((self.D, self.D), device=y.device, dtype=torch.float64)
        H_abs = torch.zeros((self.D, 128), device=y.device, dtype=torch.float64)
        H_inc = torch.zeros((self.D, 128), device=y.device, dtype=torch.float64)
        for i in range(0, len(y), 8):
            f = self.features(c[i:i+8], y[i:i+8]).reshape(-1, self.D).double()
            ya = y[i:i+8].reshape(-1, 128).double()
            yi = inc[i:i+8].reshape(-1, 128).double()
            G += f.T @ f
            H_abs += f.T @ ya
            H_inc += f.T @ yi
        G.diagonal().add_(ridge)
        G[0, 0] -= ridge
        self.readout_abs = torch.linalg.solve(G, H_abs).float()
        self.readout_inc = torch.linalg.solve(G, H_inc).float()
        self.readout = self.readout_abs
        assert torch.isfinite(self.readout_abs).all() and torch.isfinite(self.readout_inc).all()

    def predict(self, c, T, obs=None):
        B = len(c)
        r = c.new_zeros(B, self.n)
        prev = c.new_zeros(B, 128)
        base_drive = c @ self.Wc + self.bias
        co = self.context_out(c)
        ones = c.new_ones(B, 1)
        mod = self.modulation(c)
        preds = []
        qa = self.readout_abs
        qi = self.readout_inc
        for t in range(T):
            drive = base_drive + r @ self.W + prev @ self.Wf
            r = (1 - self.leak) * r + self.leak * torch.tanh(drive)
            feat = torch.cat([ones, co, r], dim=-1)
            if mod is not None:
                feat = torch.cat([feat, r * mod], dim=-1)
            y_abs = feat @ qa
            delta = feat @ qi
            y_inc = prev + delta
            y = self.alpha * y_abs + (1.0 - self.alpha) * y_inc
            preds.append(y)
            prev = obs[:, t] if obs is not None and t < obs.shape[1] else y
        return torch.stack(preds, dim=1)


def tune_bounded(model, validation, scale):
    rows = []
    best = None
    for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        model.clip_multiplier = multiplier
        result = model.oracle(validation, scale)
        row = {"experiment": "E24_Bounded_Increment", "clip_multiplier": multiplier, **result}
        rows.append(row)
        score = result["autonomous_rmse"]
        if math.isfinite(score) and (best is None or score < best[0]):
            best = (score, multiplier)
    if best is None:
        raise RuntimeError("No finite validation setting for bounded incremental model")
    model.clip_multiplier = best[1]
    return rows, {"clip_multiplier": best[1], "validation_autonomous_rmse": best[0]}


def tune_hybrid(model, validation, scale):
    rows = []
    best = None
    for alpha in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
        model.alpha = alpha
        result = model.oracle(validation, scale)
        row = {"experiment": "E26_Hybrid", "alpha_absolute": alpha, **result}
        rows.append(row)
        score = result["autonomous_rmse"]
        if math.isfinite(score) and (best is None or score < best[0]):
            best = (score, alpha)
    if best is None:
        raise RuntimeError("No finite validation setting for hybrid model")
    model.alpha = best[1]
    return rows, {"alpha_absolute": best[1], "validation_autonomous_rmse": best[0]}


def run_inversion(model, data, split_name, folder, args, starts, lo, hi, scale):
    d = data[split_name]
    part = folder / split_name
    part.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sorted(set(args.frames)):
        for i in range(0, len(d["keys"]), args.trajectory_batch):
            ids = list(range(i, min(i + args.trajectory_batch, len(d["keys"]))))
            cache = part / f"prefix_{n:03d}_batch_{i:03d}.json"
            if cache.exists():
                batch_rows = json.loads(cache.read_text())
            else:
                batch_rows = base.invert(model, d, ids, n, starts, args, lo, hi, scale)
                atomic_json(cache, batch_rows)
            rows.extend(batch_rows)
        current = [r for r in rows if r["frames"] == n]
        acc = 100 * np.mean([r["correct"] for r in current])
        xy = np.mean([r["xy_error_mm"] for r in current])
        print(f"{n:3d} frames | acc={acc:6.2f}% | XY={xy:7.2f} mm", flush=True)
        write_csv(part / "per_trajectory.csv", rows)
    summary = base.summarize(rows)
    write_csv(part / "summary.csv", summary)
    return rows, summary


def make_memory_weights(n, seed, device):
    rng = np.random.default_rng(seed + 2009)
    arr = rng.normal(0.0, 1.0 / math.sqrt(128), (128, n)).astype(np.float32)
    return torch.tensor(arr, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reservoir", type=int, default=1500)
    p.add_argument("--ridge", type=float, default=0.5)
    p.add_argument("--radius", type=float, default=0.95)
    p.add_argument("--leak", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=9)
    p.add_argument("--max-iter", type=int, default=40)
    p.add_argument("--trajectory-batch", type=int, default=3)
    p.add_argument("--frames", nargs="+", type=int, default=[1, 2, 3, 5, 10, 12, 23, 45, 67, 112, 157, 223])
    p.add_argument("--device", default="cuda")
    p.add_argument("--validation", choices=["grouped", "original"], default="grouped")
    p.add_argument("--run-name", default="final_cesn_round3_lightweight128")
    p.add_argument("--memory-gain", type=float, default=1.0)
    p.add_argument("--skip-test", action="store_true")
    args = p.parse_args()

    assert args.reservoir > 0 and args.ridge > 0 and args.restarts > 0 and args.max_iter > 0
    assert 0 < args.leak <= 1 and args.radius > 0 and args.memory_gain > 0
    assert all(1 <= x <= 223 for x in args.frames)
    assert Path(args.run_name).name == args.run_name and args.run_name not in (".", "..")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    ae_dir = cfg.OUT / "autoencoder_lightweight128"
    latent_path = ae_dir / "cleaver_lightweight_latent128.h5"
    split_path = ae_dir / "split.json"
    assert latent_path.is_file(), latent_path
    assert split_path.is_file(), split_path
    out = cfg.OUT / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    data, split = base.load(latent_path, split_path)
    if args.validation == "grouped":
        data, split = base.grouped_validation(data, split)
    info, lo, hi = base.normalize(data, device)
    scale = info["dynamic_scale"]
    for d in data.values():
        d["pca"] = d["raw_global"]  # Raw128; PCA is not used by these final models.

    manifest = {
        "args": vars(args),
        "latent_sha256": digest(latent_path),
        "split": split,
        "script_sha256": digest(Path(__file__)),
        "protocol": "validation-only hyperparameter selection; one final test evaluation",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        # A reporting-only script patch may change the file hash without
        # changing the data, split, or experiment arguments. Permit cache
        # reuse only when every non-code manifest field is identical.
        ignored = {"script_sha256", "previous_script_sha256"}
        previous_core = {
            k: v for k, v in previous_manifest.items() if k not in ignored
        }
        current_core = {
            k: v for k, v in manifest.items() if k not in ignored
        }
        if previous_core != current_core:
            raise RuntimeError(
                "Run manifest differs in data, split, or arguments. "
                "Choose a new --run-name."
            )
        old_code_hash = previous_manifest.get("script_sha256")
        if old_code_hash and old_code_hash != manifest["script_sha256"]:
            manifest["previous_script_sha256"] = old_code_hash
        elif previous_manifest.get("previous_script_sha256"):
            manifest["previous_script_sha256"] = previous_manifest[
                "previous_script_sha256"
            ]
        atomic_json(manifest_path, manifest)
    else:
        atomic_json(manifest_path, manifest)
    atomic_json(out / "normalization.json", info)
    atomic_json(out / "split_used.json", split)

    print("Pairs:", {k: len(v["keys"]) for k, v in data.items()}, flush=True)
    for name, d in data.items():
        print(name, "outside train XY bounds:", int(d["outside_bounds"].sum()), flush=True)

    common = base.common_weights(args.reservoir, args.seed, args.radius, device)
    common_raw = raw_context_weights(common, args.reservoir, args.seed, device)
    rng = np.random.default_rng(args.seed)
    starts = torch.tensor(rng.uniform(-0.95, 0.95, (args.restarts, 2)), device=device, dtype=torch.float32)
    starts[0] = 0.0

    models = {}
    models["E20_Raw128_Absolute"] = base.CESN(common_raw, 128, True, args.leak, OPTIONS, incremental=False)
    models["E22_Raw128_Increment"] = base.CESN(common_raw, 128, True, args.leak, OPTIONS, incremental=True)
    models["E24_Bounded_Increment"] = BoundedIncrementalCESN(common_raw, 128, True, args.leak, OPTIONS)
    models["E25_InitialState_Memory"] = InitialStateMemoryCESN(
        common_raw, make_memory_weights(args.reservoir, args.seed, device), args.leak, args.memory_gain
    )
    models["E26_Hybrid"] = HybridCESN(common_raw, 128, True, args.leak, OPTIONS, alpha=0.5)

    all_rows = []
    oracle_rows = []
    tuning_rows = []
    chosen = {}

    for name, model in models.items():
        print("\n" + "=" * 88 + "\n" + name + "\n" + "=" * 88, flush=True)
        t0 = time.perf_counter()
        model.fit(data["train"], args.ridge)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter() - t0
        print("fit seconds:", round(fit_seconds, 3), flush=True)

        if name == "E24_Bounded_Increment":
            rows, selection = tune_bounded(model, data["validation"], scale)
            tuning_rows.extend(rows)
            chosen[name] = selection
        elif name == "E26_Hybrid":
            rows, selection = tune_hybrid(model, data["validation"], scale)
            tuning_rows.extend(rows)
            chosen[name] = selection
        elif name == "E25_InitialState_Memory":
            chosen[name] = {"memory_gain": args.memory_gain}
        else:
            chosen[name] = {}

        folder = out / name
        folder.mkdir(exist_ok=True)
        atomic_json(folder / "selected_hyperparameters.json", chosen[name])
        for split_name in (("validation",) if args.skip_test else ("validation", "test")):
            d = data[split_name]
            oracle = model.oracle(d, scale)
            oracle_rows.append({"experiment": name, "split": split_name, "fit_seconds": fit_seconds, **oracle})
            print(split_name, "oracle", oracle, flush=True)
            _, summary = run_inversion(model, data, split_name, folder, args, starts, lo, hi, scale)
            all_rows.extend({"experiment": name, "split": split_name, **r} for r in summary)
            write_csv(out / "comparison.csv", all_rows)
            write_csv(out / "oracle_forward.csv", oracle_rows)
            write_csv(out / "hyperparameter_selection.csv", tuning_rows)

    atomic_json(out / "selected_hyperparameters.json", chosen)
    write_csv(out / "comparison.csv", all_rows)
    write_csv(out / "oracle_forward.csv", oracle_rows)
    write_csv(out / "hyperparameter_selection.csv", tuning_rows)

    from CLEAVER_OPTIMIZED.plot_final_figures import make_figures
    make_figures(out)
    print("\nDONE", out.resolve(), flush=True)
    print("Figures:", (out / "paper_figures").resolve(), flush=True)


if __name__ == "__main__":
    main()
