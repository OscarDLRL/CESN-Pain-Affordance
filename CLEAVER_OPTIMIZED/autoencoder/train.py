"""Train the V6 dual-encoder AE in two stages (recipe of the best V6Compact128 run).

Stage 1: static encoder + aux decoder on reference images only, then frozen.
Stage 2: dynamic encoder/decoder on (reference, frame_t) pairs.
Model selection on the validation split only.

    python -m CLEAVER_OPTIMIZED.autoencoder.train --run v6_ref1
    python -m CLEAVER_OPTIMIZED.autoencoder.train --run v6_ref1 --resume
    python -m CLEAVER_OPTIMIZED.autoencoder.train --run smoke --smoke
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED import data
from CLEAVER_OPTIMIZED.autoencoder.losses import full_losses, state_loss, temporal_smoothness
from CLEAVER_OPTIMIZED.autoencoder.model import V6DualEncoder


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def amp():
    return torch.autocast("cuda", dtype=torch.float16)


def pair_batches(n_traj, frames, batch, rng=None):
    """Yield (trajectory index array, frame index array)."""
    ix = np.arange(n_traj * len(frames))
    if rng is not None:
        rng.shuffle(ix)
    for i in range(0, len(ix), batch):
        a = ix[i:i + batch]
        yield torch.as_tensor(a // len(frames), device="cuda"), torch.as_tensor(frames[a % len(frames)], device="cuda")


@torch.no_grad()
def cache_static(model, rgb):
    model.eval()
    z = torch.zeros((len(rgb), model.static_latent_dim), device="cuda", dtype=torch.float16)
    mse = torch.zeros(len(rgb), device="cuda")
    for i in range(0, len(rgb), 16):
        x = data.to_tensor(rgb[i:i + 16, cfg.REFERENCE_FRAME], "cuda")
        with amp():
            s = model.encode_static(x)
            rec = model.decode_static_aux(s)
        z[i:i + 16] = s
        mse[i:i + 16] = (rec.float() - x).square().flatten(1).mean(1)
    return z, mse


def run_epoch(model, rgb, frames, cache, batch, opt=None, scaler=None, rng=None, state=None, w_state=0.0, w_smooth=0.0):
    """state: standardised robot state [N, 223, 4] on GPU (training-only target).
    w_smooth > 0 also encodes frames t-1 and t+1 (frames must exclude the ends)."""
    training = opt is not None
    model.train(training)
    model.static_encoder.eval()
    model.static_aux_decoder.eval()
    z_static, static_mse = cache
    sums, count = {}, 0
    for ix, t in pair_batches(len(rgb), frames, batch, rng if training else None):
        ref = data.to_tensor(rgb[ix, cfg.REFERENCE_FRAME], "cuda")
        y = data.to_tensor(rgb[ix, t], "cuda")
        if training:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), amp():
            z_dyn = None
            if w_smooth:
                # t, t-1, t+1 in ONE encoder call: BatchNorm must normalise them together
                B = len(ix)
                frames3 = data.to_tensor(torch.cat([rgb[ix, t], rgb[ix, t - 1], rgb[ix, t + 1]]), "cuda")
                z3 = model.encode_dynamic(frames3 - ref.repeat(3, 1, 1, 1))
                z_dyn = z3[:B]
            out = model(ref, y, z_static[ix], static_mse[ix], z_dynamic=z_dyn)
            losses = full_losses(out, ref, y)
            if w_state:
                losses["state"] = state_loss(out["state"], state[ix, t])
                losses["total"] = losses["total"] + w_state * losses["state"]
            if w_smooth:
                losses["smooth"] = temporal_smoothness(z3[B:2 * B], z_dyn, z3[2 * B:])
                losses["total"] = losses["total"] + w_smooth * losses["smooth"]
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.step(opt)
            scaler.update()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach()) * len(ix)
        count += len(ix)
    return {k: v / count for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output folder under outputs/autoencoder/")
    ap.add_argument("--static-epochs", type=int, default=15)
    ap.add_argument("--joint-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--static-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--stride", type=int, default=2, help="training frame stride (validation uses 1)")
    ap.add_argument("--detail-size", type=int, default=cfg.DETAIL_SIZE)
    ap.add_argument("--state-weight", type=float, default=0.0, help="training-only TCP/gripper head; 0 disables")
    ap.add_argument("--smooth-weight", type=float, default=0.0, help="latent acceleration penalty; 0 disables")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 pairs per split, 1 epoch each")
    a = ap.parse_args()
    assert min(a.static_epochs, a.joint_epochs, a.batch_size, a.static_batch, a.patience, a.stride) > 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    out = cfg.OUT / "autoencoder" / a.run
    out.mkdir(parents=True, exist_ok=True)
    latest, best = out / "latest.pt", out / "autoencoder.pt"
    if not a.resume and (latest.exists() or best.exists()):
        raise RuntimeError(f"{out} already has checkpoints: use --resume or a new --run")

    seed_everything(cfg.SEED)
    data.check_reference_is_settled()
    keys = data.split_keys()
    if a.smoke:
        keys = {k: v[:4] for k, v in keys.items()}
        a.static_epochs = a.joint_epochs = 1
        a.stride = 50
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rgb_train = data.to_gpu_cache(data.load_rgb(keys["train"]))
    rgb_val = data.to_gpu_cache(data.load_rgb(keys["validation"]))
    train_frames = data.frame_indices(a.stride)
    val_frames = data.frame_indices(1 if not a.smoke else 50)
    if a.smooth_weight:  # need t-1 and t+1 inside the sequence
        train_frames = train_frames[(train_frames > cfg.REFERENCE_FRAME) & (train_frames < cfg.NUM_FRAMES - 1)]
        val_frames = val_frames[(val_frames > cfg.REFERENCE_FRAME) & (val_frames < cfg.NUM_FRAMES - 1)]
    state_dim = 4 if a.state_weight else 0
    state_train = state_val = None
    state_stats = None
    if state_dim:
        s_tr, s_va = data.load_robot_state(keys["train"]), data.load_robot_state(keys["validation"])
        mu, sd = s_tr.reshape(-1, 4).mean(0), s_tr.reshape(-1, 4).std(0) + 1e-6
        state_stats = dict(mean=mu.tolist(), std=sd.tolist(), fields=["tcp_x", "tcp_y", "tcp_z", "gripper_width"])
        state_train = torch.as_tensor((s_tr - mu) / sd, device="cuda")
        state_val = torch.as_tensor((s_va - mu) / sd, device="cuda")
    aux = dict(w_state=a.state_weight, w_smooth=a.smooth_weight)

    settings = dict(args=vars(a), dataset=str(cfg.RGB), split=cfg.SPLIT, reference_frame=cfg.REFERENCE_FRAME,
                    static_latent_dim=cfg.STATIC_LATENT_DIM, dynamic_latent_dim=cfg.DYNAMIC_LATENT_DIM)
    signature = {k: v for k, v in settings.items() if k != "args"} | {
        k: getattr(a, k) for k in ("static_epochs", "batch_size", "static_batch", "lr", "stride", "detail_size",
                                   "state_weight", "smooth_weight")}
    write_json(out / "settings.json", settings)

    model = V6DualEncoder(cfg.STATIC_LATENT_DIM, cfg.DYNAMIC_LATENT_DIM, a.detail_size, state_dim).cuda().to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters:", n_params, "| pairs", {k: len(v) // 2 for k, v in keys.items()}, flush=True)

    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(cfg.SEED)
    static_done = joint_done = stale = 0
    best_loss = float("inf")
    history = []
    resume = None
    if a.resume:
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        assert resume["signature"] == signature, "resume configuration mismatch"
        model.load_state_dict(resume["model_state_dict"])
        static_done, joint_done = resume["static_done"], resume["joint_done"]
        best_loss, stale, history = resume["best_loss"], resume["stale"], resume["history"]
        rng.bit_generator.state = resume["rng"]
        scaler.load_state_dict(resume["scaler"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])

    def save_latest(opt, stage):
        atomic_save(dict(model_state_dict=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                         stage=stage, static_done=static_done, joint_done=joint_done, best_loss=best_loss,
                         stale=stale, history=history, rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         signature=signature), latest)

    # ---------------- stage 1: static branch on reference images
    static_params = list(model.static_encoder.parameters()) + list(model.static_aux_decoder.parameters())
    opt = torch.optim.AdamW(static_params, lr=a.lr, weight_decay=1e-5)
    if resume and resume["stage"] == "static":
        opt.load_state_dict(resume["optimizer"])
    for epoch in range(static_done + 1, a.static_epochs + 1):
        model.train()
        order = rng.permutation(len(rgb_train))
        total, start = 0.0, time.perf_counter()
        for i in range(0, len(order), a.static_batch):
            ix = torch.as_tensor(order[i:i + a.static_batch], device="cuda")
            x = data.to_tensor(rgb_train[ix, cfg.REFERENCE_FRAME], "cuda")
            opt.zero_grad(set_to_none=True)
            with amp():
                loss = F.mse_loss(model.decode_static_aux(model.encode_static(x)), x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(ix)
        static_done = epoch
        save_latest(opt, "static")
        print(f"STATIC {epoch:03d} loss={total / len(rgb_train):.6f} sec={time.perf_counter() - start:.1f}", flush=True)

    # ---------------- stage 2: dynamic branch, static frozen
    for m in (model.static_encoder, model.static_aux_decoder):
        m.requires_grad_(False)
        m.eval()
    cache_train, cache_val = cache_static(model, rgb_train), cache_static(model, rgb_val)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-5)
    if resume and resume["stage"] == "joint":
        opt.load_state_dict(resume["optimizer"])
    for epoch in range(joint_done + 1, a.joint_epochs + 1):
        if stale >= a.patience:
            break
        start = time.perf_counter()
        tm = run_epoch(model, rgb_train, train_frames, cache_train, a.batch_size, opt, scaler, rng, state_train, **aux)
        vm = run_epoch(model, rgb_val, val_frames, cache_val, a.batch_size, state=state_val, **aux)
        if not np.isfinite(vm["total"]):
            raise RuntimeError("non-finite validation loss")
        joint_done = epoch
        history.append(dict(epoch=epoch, train=tm, validation=vm, seconds=time.perf_counter() - start))
        if vm["total"] < best_loss - 1e-5 or not best.exists():
            best_loss, stale = vm["total"], 0
            atomic_save(dict(model_state_dict=model.state_dict(), architecture="v6_dual_encoder",
                             static_latent_dim=cfg.STATIC_LATENT_DIM, dynamic_latent_dim=cfg.DYNAMIC_LATENT_DIM,
                             detail_size=a.detail_size, state_dim=state_dim, state_stats=state_stats,
                             reference_frame=cfg.REFERENCE_FRAME, epoch=epoch,
                             split=cfg.SPLIT), best)
        else:
            stale += 1
        save_latest(opt, "joint")
        write_json(out / "history.json", history)
        print(f"JOINT {epoch:03d} train={tm['total']:.6f} val={vm['total']:.6f} "
              f"sec={history[-1]['seconds']:.1f} patience={stale}/{a.patience}", flush=True)

    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = run_epoch(model, rgb_val, val_frames, cache_static(model, rgb_val), a.batch_size, state=state_val, **aux)
    write_json(out / "training_summary.json", dict(best_epoch=ckpt["epoch"], validation_metrics=metrics,
                                                   parameters=n_params, history=history))
    print("VALIDATION", metrics, flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
