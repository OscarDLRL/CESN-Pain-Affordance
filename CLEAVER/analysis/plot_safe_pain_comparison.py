from pathlib import Path
from collections import defaultdict
import argparse
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "CLEAVER/outputs/figures/dataset"


def get_dataset_path():
    from CLEAVER.config import RGB
    candidates = [
        Path(RGB),
        ROOT / "datasets/cleaver_extended_workspace_162/cleaver_same_goal_162.h5",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError("No se encontró el dataset RGB.")


def get_scalar(group, key, default):
    if key in group:
        return int(np.asarray(group[key][()]).reshape(-1)[0])
    return int(group.attrs.get(key, default))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", type=int, default=None)
    args = parser.parse_args()

    dataset = get_dataset_path()
    print("Dataset:", dataset)

    OUT.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset, "r") as f:

        pairs = defaultdict(dict)

        for name in f.keys():
            if not re.fullmatch(r"trajectory_\d+", name):
                continue

            group = f[name]
            pair_id = int(group.attrs["pair_id"])
            label = int(group.attrs["label"])

            pairs[pair_id][label] = name

        valid = [
            pid for pid, trajectories in pairs.items()
            if 0 in trajectories and 1 in trajectories
        ]

        if not valid:
            raise RuntimeError("No se encontraron pares SAFE/PAIN.")

        if args.pair_id is None:
            pair_id = valid[0]
        else:
            pair_id = args.pair_id
            if pair_id not in valid:
                raise ValueError(f"Pair {pair_id} no está completo.")

        safe = f[pairs[pair_id][0]]
        pain = f[pairs[pair_id][1]]

        fig, axes = plt.subplots(
            2, 4,
            figsize=(10, 5.5)
        )

        for row, (group, label) in enumerate([
            (safe, "SAFE"),
            (pain, "PAIN"),
        ]):

            rgb = np.asarray(group["rgb"])

            if rgb.dtype != np.uint8:
                if rgb.max() <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            n = len(rgb)

            reach = get_scalar(group, "reach_end", n // 3)
            close = get_scalar(group, "close_end", 2 * n // 3)

            indices = [
                0,
                reach,
                close,
                n - 1
            ]

            titles = [
                "Initial",
                "Reach",
                "Grasp",
                "Final"
            ]

            for col, (idx, title) in enumerate(
                zip(indices, titles)
            ):

                idx = np.clip(idx, 0, n - 1)

                axes[row, col].imshow(rgb[idx])
                axes[row, col].set_title(
                    f"{title}\nFrame {idx}",
                    fontsize=9
                )
                axes[row, col].axis("off")

            axes[row, 0].text(
                -0.12, 0.5, label,
                transform=axes[row, 0].transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=11,
                fontweight="bold"
            )

        fig.suptitle(
            f"SAFE vs PAIN — Pair {pair_id}",
            fontsize=13
        )

        plt.tight_layout()

        output = OUT / "03_safe_pain_comparison.png"

        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved:", output)


if __name__ == "__main__":
    main()
