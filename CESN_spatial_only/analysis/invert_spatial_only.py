
from __future__ import annotations

import argparse
import json
import h5py
import numpy as np
import pandas as pd
import torch

from CESN.core.cesn import ContextualESNTorch
from CESN_spatial_only.core.parameters_spatial_only import (
    LATENT_DATA_PATH, RGB_DATA_PATH, OUT_DIR, MODEL_PATH,
    GRASP_X_SCALE, GRASP_Y_SCALE,
    SAFE_TARGET_LOCAL_X, PAIN_TARGET_LOCAL_X,
    TIME_INDEX_SCALE,
)

HISTORIES = [5,10,15,20,25,30,40,50,60,70,80,90,100]
RESTARTS_XY = [
    (-0.145,-0.080),(-0.145,0.000),(-0.145,+0.080),
    (0.000,-0.080),(0.000,0.000),(0.000,+0.080),
    (+0.145,-0.080),(+0.145,0.000),(+0.145,+0.080),
]


def transform_static(z_static, mean, components):
    return ((z_static - mean) @ components.T).astype(np.float32)


def load_test_sequences(test_pairs):
    test_pairs = set(int(x) for x in test_pairs)
    rows = []

    with h5py.File(LATENT_DATA_PATH, "r") as lh5, \
         h5py.File(RGB_DATA_PATH, "r") as rh5:
        for key in sorted(lh5.keys()):
            if not str(key).startswith("trajectory_"):
                continue

            lg = lh5[key]
            pair_id = int(lg.attrs["pair_id"])
            if pair_id not in test_pairs:
                continue

            rg = rh5[key]
            label = int(lg.attrs["pain_label"])
            xyz = np.asarray(rg["object_xyz"][0], dtype=np.float32)
            knife_x = float(xyz[0])
            knife_y = float(xyz[1])

            true_x = knife_x + (
                SAFE_TARGET_LOCAL_X if label == 0 else PAIN_TARGET_LOCAL_X
            )
            true_y = knife_y

            rows.append({
                "key": str(key),
                "pair_id": pair_id,
                "pain_label": label,
                "knife_x": knife_x,
                "knife_y": knife_y,
                "true_grasp_x": float(true_x),
                "true_grasp_y": float(true_y),
                "z_static": np.asarray(lg["z_static"], dtype=np.float32),
                "z_dynamic": np.asarray(lg["z_dynamic"], dtype=np.float32),
            })
    return rows


def build_context_sequence(static_pca, x, y, T, device):
    B = x.shape[0]

    static = torch.as_tensor(
        static_pca, dtype=torch.float32, device=device
    ).view(1,-1).expand(B,-1)

    grasp = torch.stack([
        x / GRASP_X_SCALE,
        y / GRASP_Y_SCALE,
    ], dim=1)

    base = torch.cat([static, grasp], dim=1)
    base_seq = base[:,None,:].expand(B,T,base.shape[1])

    time = (
        torch.arange(T, dtype=torch.float32, device=device)
        / float(TIME_INDEX_SCALE)
    ).view(1,T,1).expand(B,-1,-1)

    return torch.cat([base_seq,time], dim=2)


def predict_batch(model, U):
    B,T,_ = U.shape
    state = torch.zeros(
        (B, model.reservoir_size),
        device=model.device,
        dtype=model.dtype,
    )
    outputs = []

    for t in range(T):
        u_t = U[:,t,:]
        pre = (
            u_t @ model.Win.T
            + state @ model.W.T
            + model.bias[None,:]
        )
        candidate = torch.tanh(pre)
        state = (
            (1.0-model.leak_rate)*state
            + model.leak_rate*candidate
        )

        ones = torch.ones(
            (B,1), device=model.device, dtype=model.dtype
        )
        X = torch.cat([ones,u_t,state], dim=1)
        outputs.append(X @ model.Wout)

    return torch.stack(outputs, dim=1)


def optimize_xy(model, observed, static_pca, max_iter, history_size):
    device = model.device
    observed_t = torch.as_tensor(
        observed, dtype=torch.float32, device=device
    )

    raw0 = []
    for x0,y0 in RESTARTS_XY:
        nx = np.clip(x0/GRASP_X_SCALE, -0.999999, 0.999999)
        ny = np.clip(y0/GRASP_Y_SCALE, -0.999999, 0.999999)
        raw0.append([np.arctanh(nx),np.arctanh(ny)])

    raw = torch.tensor(
        raw0, dtype=torch.float32, device=device, requires_grad=True
    )

    optimizer = torch.optim.LBFGS(
        [raw],
        lr=1.0,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    T = int(observed_t.shape[0])

    def closure():
        optimizer.zero_grad(set_to_none=True)

        x = torch.tanh(raw[:,0]) * GRASP_X_SCALE
        y = torch.tanh(raw[:,1]) * GRASP_Y_SCALE

        U = build_context_sequence(
            static_pca, x, y, T, device
        )
        pred = predict_batch(model, U)
        losses = torch.mean(
            (pred-observed_t[None,:,:])**2,
            dim=(1,2),
        )
        objective = losses.sum()
        objective.backward()
        return objective

    optimizer.step(closure)

    with torch.no_grad():
        x = torch.tanh(raw[:,0]) * GRASP_X_SCALE
        y = torch.tanh(raw[:,1]) * GRASP_Y_SCALE

        U = build_context_sequence(
            static_pca, x, y, T, device
        )
        pred = predict_batch(model, U)
        losses = torch.mean(
            (pred-observed_t[None,:,:])**2,
            dim=(1,2),
        )

        idx = int(torch.argmin(losses).item())
        return (
            float(x[idx].item()),
            float(y[idx].item()),
            float(losses[idx].item()),
        )


def xy_error_mm(px,py,tx,ty):
    return 1000.0 * float(np.sqrt((px-tx)**2 + (py-ty)**2))


def summarize(values):
    v = np.asarray(values,dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v,ddof=1)) if len(v)>1 else 0.0,
        "median": float(np.median(v)),
        "p90": float(np.percentile(v,90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbfgs-max-iter", type=int, default=80)
    ap.add_argument("--lbfgs-history-size", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA was not detected.")

    checkpoint = torch.load(MODEL_PATH,map_location="cpu")
    model = ContextualESNTorch.from_state_dict(
        checkpoint["model_state"], device=device
    )

    pca_mean = checkpoint["pca_mean"].cpu().numpy().astype(np.float32)
    pca_components = checkpoint["pca_components"].cpu().numpy().astype(np.float32)

    sequences = load_test_sequences(checkpoint["test_pairs"])

    result_dir = OUT_DIR / "inversion"
    result_dir.mkdir(parents=True,exist_ok=True)

    print()
    print("="*112)
    print("SPATIAL-ONLY INVERSION | NO TASK CONTEXT")
    print("="*112)
    print("Infer only grasp XY, then derive affordance from geometry.")
    print("="*112)

    rows = []

    for pct in HISTORIES:
        errors = []
        correct = []

        for item in sequences:
            T = int(item["z_dynamic"].shape[0])
            n_obs = max(2,int(np.ceil(T*pct/100.0)))
            n_obs = min(n_obs,T)

            observed = item["z_dynamic"][:n_obs]
            static_pca = transform_static(
                item["z_static"], pca_mean, pca_components
            )

            px,py,loss = optimize_xy(
                model, observed, static_pca,
                args.lbfgs_max_iter,
                args.lbfgs_history_size,
            )

            err = xy_error_mm(
                px,py,item["true_grasp_x"],item["true_grasp_y"]
            )

            relative_x = px - item["knife_x"]
            pred_label = 0 if relative_x >= 0.0 else 1
            is_correct = int(pred_label == item["pain_label"])

            errors.append(err)
            correct.append(is_correct)

            rows.append({
                "history_percent": pct,
                "trajectory": item["key"],
                "pair_id": item["pair_id"],
                "true_label": item["pain_label"],
                "pred_label": pred_label,
                "correct": is_correct,
                "pred_x": px,
                "pred_y": py,
                "knife_x": item["knife_x"],
                "knife_y": item["knife_y"],
                "relative_grasp_x": relative_x,
                "true_x": item["true_grasp_x"],
                "true_y": item["true_grasp_y"],
                "xy_error_mm": err,
                "latent_loss": loss,
                "n_obs": n_obs,
                "T_full": T,
            })

        s = summarize(errors)
        acc = 100.0 * float(np.mean(correct))

        print(
            f"[DONE] {pct:3d}%"
            f" | AFFORDANCE={acc:6.2f}%"
            f" | XY mean={s['mean']:7.2f} ± {s['std']:6.2f} mm"
            f" | median={s['median']:6.2f}"
            f" | P90={s['p90']:6.2f}"
        )

    df = pd.DataFrame(rows)
    detail_path = result_dir/"spatial_only_per_trajectory.csv"
    df.to_csv(detail_path,index=False)

    summary_rows = []
    for pct in HISTORIES:
        g = df[df["history_percent"]==pct]
        s = summarize(g["xy_error_mm"].to_numpy())
        summary_rows.append({
            "history_percent": pct,
            "affordance_accuracy_pct": 100.0*float(g["correct"].mean()),
            "xy_mean_mm": s["mean"],
            "xy_std_mm": s["std"],
            "xy_median_mm": s["median"],
            "xy_p90_mm": s["p90"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = result_dir/"spatial_only_summary.csv"
    summary_df.to_csv(summary_csv,index=False)

    summary_json = result_dir/"spatial_only_summary.json"
    summary_json.write_text(
        json.dumps({
            "method": "CESN spatial-only inversion",
            "task_as_context": False,
            "affordance_rule": "relative_grasp_x >= 0 -> SAFE; < 0 -> PAIN",
            "summary": summary_rows,
        },indent=2),
        encoding="utf-8",
    )

    print("Saved:",detail_path.resolve())
    print("Saved:",summary_csv.resolve())
    print("Saved:",summary_json.resolve())


if __name__ == "__main__":
    main()
