"""Train spatial-context CESN: [task, cube_x, cube_y] + time -> joints."""

from pathlib import Path
import argparse
import importlib
import sys
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


p = importlib.import_module(
    "CESN.core.parametersCESN"
)

cesn = importlib.import_module(
    "CESN.core.cesn"
)

from CESN.core.data import load_split


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--neurons",
        type=int,
        nargs="+",
    )

    ap.add_argument(
        "--runs",
        type=int,
    )

    args = ap.parse_args()

    neurons = (
        tuple(args.neurons)
        if args.neurons
        else tuple(p.NEURON_COUNTS)
    )

    runs = int(
        args.runs
        or p.RUNS
    )

    # ============================================================
    # Load training data
    # ============================================================

    train = []

    for split in p.TRAIN_SPLITS:
        rows = load_split(
            split,
            p.DATA_PATH,
            joint_dim=p.JOINT_DIM,
        )

        print(
            f"{split.upper()}="
            f"{len(rows)}"
        )

        train.extend(rows)

    if not train:
        raise RuntimeError(
            "No training trajectories were loaded."
        )

    # ============================================================
    # Validate dataset
    # ============================================================

    for i, row in enumerate(train):

        missing = [
            k
            for k in (
                "joints",
                "cube_xyz",
            )
            if k not in row
        ]

        if missing:
            raise KeyError(
                f"Row {i} missing fields {missing}. "
                f"Available: {sorted(row.keys())}"
            )

        task = row.get(
            "task",
            row.get("kind"),
        )

        if task is None:
            raise KeyError(
                f"Row {i} has no 'task' or 'kind'. "
                f"Available: {sorted(row.keys())}"
            )

        task = str(task).lower()

        if task not in p.TASK_LABELS:
            raise ValueError(
                f"Row {i}: unsupported task={task!r}. "
                f"Expected {tuple(p.TASK_LABELS)}"
            )

    # ============================================================
    # Dataset summary
    # ============================================================

    lengths = [
        len(r["joints"])
        for r in train
    ]

    tasks = {}
    cube_xy_values = []

    for row in train:
        task = str(
            row.get("task", row.get("kind"))
        ).lower()

        tasks[task] = tasks.get(task, 0) + 1

        cube = np.asarray(
            row["cube_xyz"],
            dtype=np.float64,
        )
        cube0 = (
            cube[0]
            if cube.ndim == 2
            else cube
        ).reshape(3)

        cube_xy_values.append(cube0[:2])

    cube_xy_values = np.asarray(
        cube_xy_values,
        dtype=np.float64,
    )

    print(
        f"\nARCH={cesn.ARCHITECTURE}"
        f"\nTOTAL TRAIN={len(train)}"
        f"\nFRAMES={min(lengths)}..{max(lengths)}"
        f"\nTASKS={tasks}"
        f"\nCONTEXT DIM={p.CONTEXT_DIM}"
        f"\nTASK DIM={p.TASK_DIM}"
        f"\nCUBE DIM={p.CUBE_DIM}"
        f"\nCUBE X RANGE="
        f"{cube_xy_values[:,0].min():.6f}..{cube_xy_values[:,0].max():.6f}"
        f"\nCUBE Y RANGE="
        f"{cube_xy_values[:,1].min():.6f}..{cube_xy_values[:,1].max():.6f}"
        f"\nTIME INDEX={p.USE_TIME_INDEX}"
        f"\nJOINT DIM={p.JOINT_DIM}\n"
    )

    # ============================================================
    # Seeds
    # ============================================================

    seeds = np.random.SeedSequence(
        p.RANDOM_SEED
    ).spawn(runs)

    # ============================================================
    # Train
    # ============================================================

    for n in neurons:

        for run, seq in enumerate(
            seeds,
            1,
        ):
            seed = int(
                seq.generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )

            print(
                "=" * 72
            )

            print(
                f"N={n} | "
                f"RUN={run}/{runs} | "
                f"SEED={seed}"
            )

            print(
                "=" * 72
            )

            model = cesn.fit_model(
                train,
                n_reservoir=int(n),
                seed=seed,
            )

            model_path = (
                p.MODEL_DIR
                / f"N_{int(n)}"
                / f"run_{run:02d}"
                / "model.npz"
            )

            cesn.save_model(
                model,
                model_path,
            )

            print(
                "\nSaved:",
                model_path,
            )


if __name__ == "__main__":
    main()