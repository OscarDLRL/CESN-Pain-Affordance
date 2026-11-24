#!/usr/bin/env python3
"""Figure 11: L-BFGS evaluation paths on the frozen CESN XY loss landscape.

Run from repository root:
    python3 -m CESN.analysis.final.loss_landscape --device cuda --verify

The background loss surface is for the Pick hypothesis.

The optimizer uses Method A:
    - Discrete Pick/Push hypotheses
    - Four XY initializations
    - Batched L-BFGS with shared curvature history
    - Maximum 60 iterations

The plotted paths represent L-BFGS closure evaluations, including
line-search trial points, not exclusively accepted optimizer steps.

No retraining. No Adam.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess

from pathlib import Path

import numpy as np
import torch

from CESN.core import cesn, parametersCESN as p
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.inversion.infer_context import infer_one


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUT = (
    ROOT / "CESN/outputs/final_lbfgs/final_report/figures"
)

MODEL = (
    ROOT / "CESN/outputs/models/N_900/run_01/model.npz"
)

TRAJECTORY = "pick_interpolation_0080"


def xy_of(row):
    cube = np.asarray(row["cube_xyz"], dtype=np.float64)

    if cube.ndim == 2:
        cube = cube[0]

    return cube[:2].copy()


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--trajectory",
        default=TRAJECTORY,
    )

    parser.add_argument(
        "--K",
        type=int,
        default=30,
        help="Exact number of observed frames",
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--grid-batch",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.035,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------

    if (
        args.restarts < 1
        or args.K < 1
        or args.max_iter < 1
        or args.grid_size < 10
        or args.grid_batch < 1
    ):
        parser.error("Invalid experiment parameters")

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA is not available. Use --device cpu."
        )

    if not args.model.is_file():
        parser.error(
            f"Frozen CESN model not found: {args.model}"
        )

    if not Path(p.DATA_PATH).is_file():
        parser.error(
            f"Dataset not found: {p.DATA_PATH}"
        )

    # ------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------

    out = args.out_dir.resolve()

    stem = (
        f"11_loss_landscape_lbfgs_runs_K{args.K}"
    )

    image_path = out / f"{stem}.png"

    csv_path = out / f"{stem}_evaluations.csv"

    config_path = out / f"{stem}_config.json"

    existing = [
        path
        for path in (
            image_path,
            csv_path,
            config_path,
        )
        if path.exists()
    ]

    if existing and not args.overwrite:
        parser.error(
            "Output already exists. "
            "Use --overwrite or a different --out-dir."
        )

    # ------------------------------------------------------------
    # Load test trajectory
    # ------------------------------------------------------------

    test_rows = load_split(
        p.TEST_SPLIT,
        p.DATA_PATH,
        joint_dim=p.JOINT_DIM,
    )

    hits = [
        row
        for row in test_rows
        if str(row["id"]) == args.trajectory
    ]

    if len(hits) != 1:
        parser.error(
            f"Expected one trajectory "
            f"{args.trajectory!r}, found {len(hits)}"
        )

    row = hits[0]

    task_name = str(row["kind"]).lower()

    if task_name != "pick":
        parser.error(
            "This figure uses a Pick loss landscape. "
            "Choose a Pick trajectory."
        )

    q = np.asarray(
        row["joints"],
        dtype=np.float64,
    )

    T = len(q)

    if args.K > T:
        parser.error(
            f"K={args.K} exceeds trajectory length T={T}"
        )

    prefix = q[:args.K].copy()

    # ------------------------------------------------------------
    # Load training contexts
    # ------------------------------------------------------------

    train = []

    for split in p.TRAIN_SPLITS:
        train.extend(
            load_split(
                split,
                p.DATA_PATH,
                joint_dim=p.JOINT_DIM,
            )
        )

    if not train:
        parser.error(
            "Training split not found."
        )

    train_xy = np.stack(
        [xy_of(row) for row in train]
    )

    low = train_xy.min(axis=0)
    high = train_xy.max(axis=0)

    # ------------------------------------------------------------
    # Generate initial XY contexts
    # ------------------------------------------------------------

    rng = np.random.default_rng(args.seed)

    starts = np.column_stack(
        (
            rng.uniform(
                low[0],
                high[0],
                args.restarts,
            ),
            rng.uniform(
                low[1],
                high[1],
                args.restarts,
            ),
        )
    )

    true_xy = xy_of(row)

    xlo, ylo = (
        np.minimum(low, true_xy) - args.margin
    )

    xhi, yhi = (
        np.maximum(high, true_xy) + args.margin
    )

    # ------------------------------------------------------------
    # Load frozen CESN
    # ------------------------------------------------------------

    model = cesn.load_model(args.model)

    dtype = (
        torch.float32
        if args.dtype == "float32"
        else torch.float64
    )

    runner = GPUBatchedCESN(
        model,
        p.LEAK_RATE,
        device=args.device,
        dtype=dtype,
    )

    device = runner.device

    # ------------------------------------------------------------
    # Construct XY loss landscape
    # ------------------------------------------------------------

    xs = np.linspace(
        xlo,
        xhi,
        args.grid_size,
    )

    ys = np.linspace(
        ylo,
        yhi,
        args.grid_size,
    )

    grid_x, grid_y = np.meshgrid(xs, ys)

    grid_xy = np.column_stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
        ]
    )

    task_pick = np.asarray(
        p.TASK_LABELS["pick"],
        dtype=np.float64,
    )

    grid_loss_parts = []

    print("Computing loss landscape...")

    with torch.no_grad():

        for start in range(
            0,
            len(grid_xy),
            args.grid_batch,
        ):

            chunk = grid_xy[
                start:start + args.grid_batch
            ]

            task_batch = np.repeat(
                task_pick[None],
                len(chunk),
                axis=0,
            )

            contexts = np.column_stack(
                [task_batch, chunk]
            )

            c = torch.tensor(
                contexts,
                dtype=dtype,
                device=device,
            )

            target = torch.tensor(
                np.repeat(
                    prefix[None],
                    len(chunk),
                    axis=0,
                ),
                dtype=dtype,
                device=device,
            )

            ts = torch.full(
                (len(chunk),),
                T,
                dtype=torch.long,
                device=device,
            )

            ks = torch.full(
                (len(chunk),),
                args.K,
                dtype=torch.long,
                device=device,
            )

            losses = runner.prefix_loss(
                c,
                target,
                ts,
                ks,
            )

            grid_loss_parts.append(
                losses.detach().cpu().numpy()
            )

    surface = np.concatenate(
        grid_loss_parts
    ).reshape(grid_x.shape)

    if not np.isfinite(surface).all():
        raise RuntimeError(
            "Non-finite values in loss landscape."
        )

    # ------------------------------------------------------------
    # Method A: discrete Pick / Push hypotheses
    # ------------------------------------------------------------

    candidates = np.repeat(
        starts,
        2,
        axis=0,
    )

    hypothesis_names = [
        "pick" if j % 2 == 0 else "push"
        for j in range(2 * args.restarts)
    ]

    task_codes = np.stack(
        [
            p.TASK_LABELS[name]
            for name in hypothesis_names
        ]
    ).astype(np.float64)

    xy = torch.tensor(
        candidates,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    task = torch.tensor(
        task_codes,
        dtype=dtype,
        device=device,
    )

    targets = torch.tensor(
        np.repeat(
            prefix[None],
            len(candidates),
            axis=0,
        ),
        dtype=dtype,
        device=device,
    )

    ts = torch.full(
        (len(candidates),),
        T,
        dtype=torch.long,
        device=device,
    )

    ks = torch.full(
        (len(candidates),),
        args.K,
        dtype=torch.long,
        device=device,
    )

    # ------------------------------------------------------------
    # Configure L-BFGS
    # ------------------------------------------------------------

    optimizer = torch.optim.LBFGS(
        [xy],
        lr=args.lr,
        max_iter=args.max_iter,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    traces = []

    def losses():

        contexts = torch.cat(
            [task, xy],
            dim=1,
        )

        return runner.prefix_loss(
            contexts,
            targets,
            ts,
            ks,
        )

    def record(step, loss_values):

        points = (
            xy.detach()
            .cpu()
            .numpy()
        )

        values = (
            loss_values.detach()
            .cpu()
            .numpy()
        )

        for i, (point, value) in enumerate(
            zip(points, values)
        ):

            traces.append(
                {
                    "evaluation": int(step),
                    "restart": int(i // 2),
                    "hypothesis": hypothesis_names[i],
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "prefix_loss": float(value),
                }
            )

    # Record initialization
    with torch.no_grad():
        record(0, losses())

    counter = [0]

    def closure():

        optimizer.zero_grad(
            set_to_none=True
        )

        per_candidate_loss = losses()

        if not torch.isfinite(
            per_candidate_loss
        ).all().item():

            raise FloatingPointError(
                "Non-finite loss during L-BFGS"
            )

        counter[0] += 1

        record(
            counter[0],
            per_candidate_loss,
        )

        # Shared-curvature batched L-BFGS
        total_loss = per_candidate_loss.sum()

        total_loss.backward()

        return total_loss

    # ------------------------------------------------------------
    # Execute L-BFGS
    # ------------------------------------------------------------

    print("Running Method A + L-BFGS...")

    optimizer.step(closure)

    with torch.no_grad():

        final_loss = (
            losses()
            .detach()
            .cpu()
            .numpy()
        )

        final_xy = (
            xy.detach()
            .cpu()
            .numpy()
        )

    winner = int(
        np.argmin(final_loss)
    )

    win_task = hypothesis_names[winner]

    win_xy = final_xy[
        winner
    ].astype(float)

    print()

    # ------------------------------------------------------------
    # Compare true context against recovered context
    # ------------------------------------------------------------

    true_context = np.concatenate(
        [
            np.asarray(
                p.TASK_LABELS[task_name],
                dtype=np.float64,
            ),
            true_xy,
        ]
    )

    true_context_tensor = torch.tensor(
        true_context[None, :],
        dtype=dtype,
        device=device,
    )

    true_target = torch.tensor(
        prefix[None, :, :],
        dtype=dtype,
        device=device,
    )

    true_T = torch.tensor(
        [T],
        dtype=torch.long,
        device=device,
    )

    true_K = torch.tensor(
        [args.K],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        true_loss = runner.prefix_loss(
            true_context_tensor,
            true_target,
            true_T,
            true_K,
        ).item()

    selected_loss = float(final_loss[winner])

    xy_error_mm = float(
        np.linalg.norm(win_xy - true_xy) * 1000.0
    )

    print()
    print("=" * 55)
    print("TRUE CONTEXT VS RECOVERED CONTEXT")
    print("=" * 55)

    print(f"True XY:       {true_xy}")
    print(f"Recovered XY:  {win_xy}")

    print(f"XY error:      {xy_error_mm:.3f} mm")

    print(f"True loss:     {true_loss:.10f}")
    print(f"Selected loss: {selected_loss:.10f}")

    print(
        "Loss difference (true - selected): "
        f"{true_loss - selected_loss:.10f}"
    )

    print("=" * 55)
    print("Selected context:")
    print("  Task:", win_task)
    print("  XY:", win_xy)
    print(
        "  Prefix loss:",
        float(final_loss[winner]),
    )
    print(
        "  Closure evaluations:",
        counter[0],
    )

    # ------------------------------------------------------------
    # Verify against official inversion engine
    # ------------------------------------------------------------

    if args.verify:

        selected = infer_one(
            runner,
            prefix,
            T,
            starts,
            method="discrete",
            optimizer="lbfgs",
            lbfgs_max_iter=args.max_iter,
            lbfgs_lr=args.lr,
        )

        delta = np.max(
            np.abs(
                selected.estimated_xy - win_xy
            )
        )

        loss_delta = abs(
            selected.prefix_loss
            - float(final_loss[winner])
        )

        print()
        print("Parity verification:")
        print(
            "  XY max difference:",
            delta,
        )
        print(
            "  Loss difference:",
            loss_delta,
        )
        print(
            "  Task:",
            selected.predicted_task,
        )

        if (
            selected.predicted_task != win_task
            or delta > 1e-6
            or loss_delta > 1e-7
        ):
            raise AssertionError(
                "L-BFGS result does not match infer_one"
            )

        print("PASS: L-BFGS parity verified")

    # ------------------------------------------------------------
    # Plot landscape
    # ------------------------------------------------------------

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(
        figsize=(8.8, 6.8)
    )

    positive = np.maximum(
        surface,
        1e-14,
    )

    floor = max(
        float(np.quantile(positive, 0.001)),
        1e-12,
    )

    upper = max(
        float(positive.max()),
        floor * (1 + 1e-6),
    )

    levels = np.geomspace(
        floor,
        upper,
        30,
    )

    filled = ax.contourf(
        grid_x,
        grid_y,
        positive,
        levels=levels,
        norm=LogNorm(
            vmin=floor,
            vmax=upper,
        ),
        cmap="viridis",
        extend="min",
    )

    fig.colorbar(
        filled,
        ax=ax,
        label=(
            "Pick-hypothesis prefix MSE "
            "(log scale)"
        ),
    )

    palette = plt.get_cmap("tab10")

    for j in range(args.restarts):

        # Plot Pick-hypothesis evaluations only
        path = [
            z
            for z in traces
            if (
                z["restart"] == j
                and z["hypothesis"] == "pick"
            )
        ]

        px = np.array(
            [z["x"] for z in path]
        )

        py = np.array(
            [z["y"] for z in path]
        )

        color = palette(j % 10)

        ax.plot(
            px,
            py,
            "-",
            color=color,
            linewidth=1.35,
            alpha=0.85,
            label=f"L-BFGS Pick restart {j + 1}",
        )

        # Starting point
        ax.scatter(
            [px[0]],
            [py[0]],
            color=color,
            s=30,
            marker="o",
            zorder=5,
        )

        # Last evaluated point
        ax.scatter(
            [px[-1]],
            [py[-1]],
            color=color,
            s=45,
            marker="x",
            zorder=6,
        )

    # True cube position
    ax.scatter(
        [true_xy[0]],
        [true_xy[1]],
        marker="*",
        color="magenta",
        edgecolor="white",
        linewidth=0.5,
        s=210,
        zorder=8,
        label="True cube XY",
    )

    # Selected context
    ax.scatter(
        [win_xy[0]],
        [win_xy[1]],
        marker="D",
        color="white",
        edgecolor="black",
        linewidth=0.8,
        s=60,
        zorder=8,
        label=f"Selected {win_task} XY",
    )

    ax.set(
        xlabel="cube x [m]",
        ylabel="cube y [m]",
        title=(
            f"CESN context loss — Pick | "
            f"{args.trajectory} | "
            f"K={args.K}/{T}"
        ),
    )

    ax.text(
        0.01,
        0.01,
        (
            "Lines: L-BFGS closure evaluations "
            "(includes line-search trials)\n"
            "Optimization includes both "
            "Pick and Push hypotheses"
        ),
        transform=ax.transAxes,
        fontsize=8.1,
        va="bottom",
        bbox=dict(
            facecolor="white",
            alpha=0.82,
            edgecolor="none",
        ),
    )

    ax.grid(alpha=0.18)

    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.95,
    )

    fig.tight_layout()

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        image_path,
        dpi=200,
    )

    plt.close(fig)

    save_csv(
        csv_path,
        traces,
    )

    try:

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        commit = "unavailable"

    config = {
        "trajectory": args.trajectory,
        "K": args.K,
        "T": T,
        "true_task": task_name,
        "true_xy": true_xy.tolist(),
        "selected_task": win_task,
        "selected_xy": win_xy.tolist(),
        "selected_loss": float(
            final_loss[winner]
        ),
        "optimizer": "batched_lbfgs_strong_wolfe",
        "method": "A_discrete",
        "restarts": args.restarts,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "lbfgs_lr": args.lr,
        "lbfgs_history": 20,
        "dtype": args.dtype,
        "device": args.device,
        "batch_cases": 1,
        "batch_candidates": 2 * args.restarts,
        "grid_size": args.grid_size,
        "closure_evaluations": counter[0],
        "git_commit": commit,
        "model_path": str(args.model),
        "model_sha256": sha256_file(
            args.model
        ),
    }

    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Figure 11 generated successfully!")
    print()
    print("PNG:", image_path)
    print("CSV:", csv_path)
    print("JSON:", config_path)


if __name__ == "__main__":
    main()