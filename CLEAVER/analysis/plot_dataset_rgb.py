import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from CLEAVER.config import RGB, OUT

FIG_DIR = OUT / "figures" / "dataset"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Central reference configurations
FIXED_OBJECT = 4
FIXED_START = 4

robot_images = {}
object_images = {}

# ==================================================
# EXTRACT ORIGINAL SIMULATOR FRAMES
# ==================================================

with h5py.File(RGB, "r") as h5:

    for key in sorted(h5.keys()):

        if not key.startswith("trajectory_"):
            continue

        g = h5[key]

        if int(g.attrs["label"]) != 0:
            continue

        sid = int(g.attrs["start_id"])
        oid = int(g.attrs["object_id"])

        if oid == FIXED_OBJECT:

            robot_images[sid] = {
                "rgb": g["rgb"][0],
                "xyz": np.asarray(g["prepared_tcp"][:])
            }

        if sid == FIXED_START:

            object_images[oid] = {
                "rgb": g["rgb"][0],
                "xyz": np.asarray(g["object_xyz"][0])[:3]
            }

assert len(robot_images) == 9
assert len(object_images) == 9


# ==================================================
# CREATE 3x3 RGB FIGURES
# ==================================================

def plot_grid(images, prefix, filename):

    fig, axes = plt.subplots(
        3, 3,
        figsize=(7.2, 7.2)
    )

    for ax, idx in zip(axes.flat, sorted(images)):

        item = images[idx]

        ax.imshow(item["rgb"])

        x, y, z = item["xyz"] * 100

        ax.set_title(
            f"{prefix}{idx + 1} "
            f"({x:.1f}, {y:.1f}) cm",
            fontsize=10
        )

        ax.axis("off")

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.97,
        wspace=0.03,
        hspace=0.13
    )

    for ext in ["png"]:

        path = FIG_DIR / f"{filename}.{ext}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03
        )

        print("SAVED:", path)

    plt.close(fig)


# FIGURE 1: ROBOT POSITIONS

plot_grid(
    robot_images,
    "R",
    "01_robot_initial_rgb"
)


# FIGURE 2: KNIFE POSITIONS

plot_grid(
    object_images,
    "K",
    "02_knife_positions_rgb"
)

print("\nDATASET FIGURES COMPLETE")

