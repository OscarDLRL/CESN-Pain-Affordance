import argparse
import h5py
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from CLEAVER.config import RGB, AE_MODEL, OUT, TEST

from CLEAVER.core.autoencoder import (
    PainStaticDynamicAutoencoderV6Native256Latent200,
)

from CLEAVER.core.autoencoder_training_utils import (
    motion_soft_mask,
)


# ==================================================
# ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pair",
    type=int,
    default=min(TEST)
)

parser.add_argument(
    "--frames",
    nargs="+",
    type=int,
    default=[25, 75, 125, 175]
)

args = parser.parse_args()

assert args.pair in TEST, "Select a held-out test pair."


# ==================================================
# CONFIGURATION
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FIG_DIR = OUT / "figures" / "autoencoder"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
})


# ==================================================
# LOAD FROZEN AUTOENCODER
# ==================================================

checkpoint = torch.load(
    AE_MODEL,
    map_location=DEVICE
)

model = (
    PainStaticDynamicAutoencoderV6Native256Latent200(
        static_latent_dim=128,
        dynamic_latent_dim=200,
    )
    .to(DEVICE)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Autoencoder loaded")
print("Device:", DEVICE)
print("Test pair:", args.pair)


# ==================================================
# RGB CONVERSION
# ==================================================

def to_tensor(rgb):

    rgb = np.asarray(rgb)

    return (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(DEVICE)
    )


def to_rgb(tensor):

    return (
        tensor[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        .clip(0, 1)
    )


def to_mask(tensor):

    return (
        tensor[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


# ==================================================
# RECONSTRUCTION FIGURES
# ==================================================

def plot_reconstruction(group, label):

    rgb = group["rgb"]

    frame0 = to_tensor(rgb[0])

    indices = args.frames

    assert all(
        0 <= t < len(rgb)
        for t in indices
    )

    fig, axes = plt.subplots(
        4,
        len(indices),
        figsize=(2.0 * len(indices), 7.5),
        squeeze=False,
    )

    row_names = [
        "Original RGB",
        "Target mask",
        "Predicted mask",
        "Reconstruction",
    ]

    frame_errors = []

    for col, t in enumerate(indices):

        frame_t = to_tensor(rgb[t])

        with torch.inference_mode():

            outputs = model(frame0, frame_t)

            target_mask = motion_soft_mask(
                frame_t - frame0
            )

            pred_mask = outputs["motion_mask"]

            reconstructed = outputs["frame_recon"]

        mse = torch.mean(
            (reconstructed - frame_t) ** 2
        ).item()

        frame_errors.append(mse)

        # Original RGB
        axes[0, col].imshow(
            to_rgb(frame_t)
        )

        # Target motion mask
        axes[1, col].imshow(
            to_mask(target_mask),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        # Predicted motion mask
        axes[2, col].imshow(
            to_mask(pred_mask),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        # Reconstructed RGB
        axes[3, col].imshow(
            to_rgb(reconstructed)
        )

        axes[0, col].set_title(
            f"Frame {t}",
            fontsize=10
        )

        for row in range(4):

            ax = axes[row, col]

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if col == 0:

                ax.set_ylabel(
                    row_names[row],
                    fontsize=10,
                    labelpad=8
                )

        print(
            f"{label} | Frame {t:03d}"
            f" | MSE={mse:.8f}"
        )

    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.95,
        bottom=0.02,
        wspace=0.025,
        hspace=0.05,
    )

    filename = (
        FIG_DIR /
        f"autoencoder_{label.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plt.close(fig)

    print("Saved:", filename)
    print("Selected-frame mean MSE:",
          np.mean(frame_errors))


# ==================================================
# USE SAME TEST PAIR FOR SAFE AND PAIN
# ==================================================

with h5py.File(RGB, "r") as h5:

    safe_key = f"trajectory_{args.pair * 2:03d}"

    pain_key = f"trajectory_{args.pair * 2 + 1:03d}"

    plot_reconstruction(
        h5[safe_key],
        "SAFE"
    )

    plot_reconstruction(
        h5[pain_key],
        "PAIN"
    )

print("\nAUTOENCODER FIGURES COMPLETE")

