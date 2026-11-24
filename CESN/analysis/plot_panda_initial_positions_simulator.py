#!/usr/bin/env python3
from pathlib import Path
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = (
    ROOT
    / "datasets"
    / "pain_affordance_162"
    / "pain_affordance_162.h5"
)

DEFAULT_OUTPUT = (
    ROOT
    / "CESN"
    / "outputs"
    / "paper_figures"
    / "panda_initial_positions_3x3.png"
)

# Exact 3x3 Panda-start grid used in the dataset.
# Order: top row -> middle row -> bottom row.
PANDA_START_GRID = [
    (-0.06, +0.06),
    ( 0.00, +0.06),
    (+0.06, +0.06),

    (-0.06,  0.00),
    ( 0.00,  0.00),
    (+0.06,  0.00),

    (-0.06, -0.06),
    ( 0.00, -0.06),
    (+0.06, -0.06),
]


def trajectory_name(index):
    return f"trajectory_{index:03d}"


def load_nine_initial_frames(h5_path, object_index=0):
    """
    Dataset layout:
        9 knife positions
        x 9 Panda initial positions
        x 2 tasks (SAFE, PAIN)
        = 162 trajectories.

    For one fixed knife position:
        pair index = object_index * 9 + panda_index

    Each pair contains two trajectories:
        even trajectory = first member of pair
        odd trajectory  = second member of pair

    Because paired SAFE/PAIN trajectories share exactly the same frame0,
    we use the even member of each pair.
    """
    frames = []
    keys = []

    with h5py.File(h5_path, "r") as h5:
        for panda_index in range(9):
            pair_index = object_index * 9 + panda_index
            traj_index = 2 * pair_index

            key = trajectory_name(traj_index)

            if key not in h5:
                raise KeyError(
                    f"{key} not found in dataset. "
                    "Dataset ordering may differ from the expected "
                    "9 knife x 9 Panda x 2 task layout."
                )

            g = h5[key]

            if "rgb" not in g:
                raise KeyError(f"{key} has no 'rgb' dataset.")

            rgb = np.asarray(g["rgb"][0], dtype=np.uint8)

            frames.append(rgb)
            keys.append(key)

    return frames, keys


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create a 3x3 simulator-image figure showing the nine "
            "initial Panda positions used in the pain-affordance dataset."
        )
    )

    ap.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    ap.add_argument(
        "--object-index",
        type=int,
        default=0,
        choices=range(9),
        metavar="[0-8]",
        help=(
            "Which fixed knife position to use for all nine Panda starts. "
            "Default: 0."
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    ap.add_argument(
        "--show-trajectory-id",
        action="store_true",
        help="Also print the source trajectory ID inside each panel.",
    )

    args = ap.parse_args()

    frames, keys = load_nine_initial_frames(
        args.dataset,
        object_index=args.object_index,
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10.5, 8.5),
    )

    for i, ax in enumerate(axes.flat):
        dx, dy = PANDA_START_GRID[i]

        ax.imshow(frames[i])
        ax.set_xticks([])
        ax.set_yticks([])

        title = (
            f"P{i+1}: "
            f"$\\Delta x={dx:+.2f}$ m, "
            f"$\\Delta y={dy:+.2f}$ m"
        )

        if args.show_trajectory_id:
            title += f"\n{keys[i]}"

        ax.set_title(
            title,
            fontsize=10,
        )

    fig.suptitle(
        (
            "Nine Initial Panda Positions\n"
            f"Fixed knife position #{args.object_index + 1}"
        ),
        fontsize=14,
    )

    plt.tight_layout(
        rect=(0, 0, 1, 0.94)
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 88)
    print("PANDA INITIAL-POSITION FIGURE CREATED")
    print("=" * 88)
    print("Dataset      :", args.dataset)
    print("Knife index  :", args.object_index)
    print("Output       :", args.output)
    print()
    print("Panda offsets:")
    for i, ((dx, dy), key) in enumerate(
        zip(PANDA_START_GRID, keys),
        start=1,
    ):
        print(
            f"  P{i}: "
            f"dx={dx:+.2f} m, dy={dy:+.2f} m"
            f"  <- {key}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
