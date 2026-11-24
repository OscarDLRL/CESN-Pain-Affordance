from __future__ import annotations

from pathlib import Path
import argparse
import json
import random

import h5py
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from CESN.core.autoencoder_v5 import (
    PainStaticDynamicAutoencoderV5,
)


# ============================================================
# GPU
# ============================================================

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision(
    "high"
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "datasets"
    / "pain_affordance_162"
    / "pain_affordance_162.h5"
)

OUT_DIR = (
    ROOT
    / "CESN"
    / "outputs"
    / "pain_affordance_spatial_motion_autoencoder_v5"
)

MODEL_PATH = (
    OUT_DIR
    / "best_spatial_motion_autoencoder_v5.pt"
)

SUMMARY_PATH = (
    OUT_DIR
    / "training_summary.json"
)

RECON_PATH = (
    OUT_DIR
    / "reconstruction_examples.png"
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 128

STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 128

TRAIN_PAIRS = 60
VAL_PAIRS = 10
TEST_PAIRS = 11

SPLIT_SEED = 42
TRAIN_SEED = 42

# Loss weights
STATIC_AUX_WEIGHT = 0.15

DYNAMIC_LOSS_WEIGHT = 2.0
FRAME_LOSS_WEIGHT = 1.0
MASK_LOSS_WEIGHT = 0.75
BACKGROUND_LOSS_WEIGHT = 2.0
CENTROID_LOSS_WEIGHT = 0.40

DYNAMIC_EDGE_WEIGHT = 0.50
FRAME_EDGE_WEIGHT = 0.50

MOTION_FOCUS = 30.0


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Split
# ============================================================

def build_pair_splits(
    path,
):
    pair_ids = set()

    with h5py.File(
        path,
        "r",
    ) as h5:
        for key in h5.keys():
            if str(
                key
            ).startswith(
                "trajectory_"
            ):
                pair_ids.add(
                    int(
                        h5[
                            key
                        ].attrs[
                            "pair_id"
                        ]
                    )
                )

    pair_ids = np.asarray(
        sorted(
            pair_ids
        ),
        dtype=np.int64,
    )

    if len(pair_ids) != 81:
        raise ValueError(
            f"Expected 81 pairs, got "
            f"{len(pair_ids)}"
        )

    rng = np.random.default_rng(
        SPLIT_SEED
    )

    rng.shuffle(
        pair_ids
    )

    train = set(
        pair_ids[
            :TRAIN_PAIRS
        ].tolist()
    )

    validation = set(
        pair_ids[
            TRAIN_PAIRS:
            TRAIN_PAIRS
            + VAL_PAIRS
        ].tolist()
    )

    test = set(
        pair_ids[
            TRAIN_PAIRS
            + VAL_PAIRS:
        ].tolist()
    )

    return {
        "train":
            train,

        "validation":
            validation,

        "test":
            test,
    }


def collect_static_index(
    path,
    pair_ids,
):
    index = []

    with h5py.File(
        path,
        "r",
    ) as h5:
        for key in sorted(
            h5.keys()
        ):
            if not str(
                key
            ).startswith(
                "trajectory_"
            ):
                continue

            if int(
                h5[
                    key
                ].attrs[
                    "pair_id"
                ]
            ) in pair_ids:
                index.append(
                    str(
                        key
                    )
                )

    return index


def collect_dynamic_index(
    path,
    pair_ids,
):
    index = []

    with h5py.File(
        path,
        "r",
    ) as h5:
        for key in sorted(
            h5.keys()
        ):
            if not str(
                key
            ).startswith(
                "trajectory_"
            ):
                continue

            g = h5[
                key
            ]

            pair_id = int(
                g.attrs[
                    "pair_id"
                ]
            )

            if pair_id not in pair_ids:
                continue

            T = int(
                g[
                    "rgb"
                ].shape[
                    0
                ]
            )

            for t in range(
                T
            ):
                index.append(
                    (
                        str(
                            key
                        ),
                        int(
                            t
                        ),
                    )
                )

    return index


# ============================================================
# Data
# ============================================================

def prepare_rgb(
    rgb,
):
    x = (
        torch.from_numpy(
            np.asarray(
                rgb,
                dtype=np.uint8,
            )
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        / 255.0
    )

    return F.interpolate(
        x.unsqueeze(
            0
        ),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    ).squeeze(
        0
    )


class H5StaticDataset(
    Dataset
):
    def __init__(
        self,
        path,
        index,
    ):
        self.path = str(
            path
        )

        self.index = list(
            index
        )

        self._h5 = None

    def _file(
        self,
    ):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.path,
                "r",
            )

        return self._h5

    def __len__(
        self,
    ):
        return len(
            self.index
        )

    def __getitem__(
        self,
        idx,
    ):
        key = self.index[
            idx
        ]

        return prepare_rgb(
            self._file()[
                key
            ][
                "rgb"
            ][
                0
            ]
        )


class H5DynamicDataset(
    Dataset
):
    def __init__(
        self,
        path,
        index,
    ):
        self.path = str(
            path
        )

        self.index = list(
            index
        )

        self._h5 = None

    def _file(
        self,
    ):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.path,
                "r",
            )

        return self._h5

    def __len__(
        self,
    ):
        return len(
            self.index
        )

    def __getitem__(
        self,
        idx,
    ):
        key, t = self.index[
            idx
        ]

        g = self._file()[
            key
        ]

        return {
            "frame0":
                prepare_rgb(
                    g[
                        "rgb"
                    ][
                        0
                    ]
                ),

            "frame_t":
                prepare_rgb(
                    g[
                        "rgb"
                    ][
                        t
                    ]
                ),
        }


# ============================================================
# Loss helpers
# ============================================================

def image_gradients(
    x,
):
    dx = (
        x[
            :,
            :,
            :,
            1:
        ]
        -
        x[
            :,
            :,
            :,
            :-1
        ]
    )

    dy = (
        x[
            :,
            :,
            1:,
            :
        ]
        -
        x[
            :,
            :,
            :-1,
            :
        ]
    )

    return (
        dx,
        dy,
    )


def edge_l1(
    pred,
    target,
):
    px, py = image_gradients(
        pred
    )

    tx, ty = image_gradients(
        target
    )

    return (
        F.l1_loss(
            px,
            tx,
        )
        +
        F.l1_loss(
            py,
            ty,
        )
    )


def motion_soft_mask(
    delta,
):
    """
    Soft target mask derived ONLY from image motion.

    No robot-state or grasp labels are used.
    """
    mag = delta.abs().mean(
        dim=1,
        keepdim=True,
    )

    # ~0 below small photometric changes,
    # rapidly approaches 1 on meaningful motion.
    return torch.sigmoid(
        (
            mag
            - 0.035
        )
        * 60.0
    )


def motion_weight_map(
    delta,
):
    mag = delta.abs().mean(
        dim=1,
        keepdim=True,
    )

    denom = (
        mag
        .flatten(
            1
        )
        .amax(
            dim=1,
            keepdim=True,
        )
        .view(
            -1,
            1,
            1,
            1,
        )
        .clamp_min(
            1e-6
        )
    )

    normalized = (
        mag
        / denom
    )

    return (
        1.0
        + MOTION_FOCUS
        * normalized
    )


def weighted_mse(
    pred,
    target,
    weight,
):
    return (
        weight
        * (
            pred
            - target
        ) ** 2
    ).mean()


def motion_centroid_target(
    delta,
):
    """
    Self-supervised target from visual motion distribution.
    Output normalized image coordinates in [-1, 1].
    """
    B, _, H, W = delta.shape

    mass = delta.abs().mean(
        dim=1
    )

    mass = (
        mass
        + 1e-6
    )

    xs = torch.linspace(
        -1.0,
        1.0,
        W,
        device=delta.device,
        dtype=delta.dtype,
    ).view(
        1,
        1,
        W,
    )

    ys = torch.linspace(
        -1.0,
        1.0,
        H,
        device=delta.device,
        dtype=delta.dtype,
    ).view(
        1,
        H,
        1,
    )

    denom = mass.sum(
        dim=(
            1,
            2,
        )
    ).clamp_min(
        1e-6
    )

    cx = (
        mass
        * xs
    ).sum(
        dim=(
            1,
            2,
        )
    ) / denom

    cy = (
        mass
        * ys
    ).sum(
        dim=(
            1,
            2,
        )
    ) / denom

    return torch.stack(
        [
            cx,
            cy,
        ],
        dim=1,
    )


def full_losses(
    outputs,
    frame0,
    frame_t,
):
    delta = (
        frame_t
        - frame0
    )

    mask_target = motion_soft_mask(
        delta
    )

    dynamic_weighted = weighted_mse(
        outputs[
            "delta_recon"
        ],
        delta,
        motion_weight_map(
            delta
        ),
    )

    dynamic_edge = edge_l1(
        outputs[
            "delta_recon"
        ],
        delta,
    )

    frame_mse = F.mse_loss(
        outputs[
            "frame_recon"
        ],
        frame_t,
    )

    frame_edge = edge_l1(
        outputs[
            "frame_recon"
        ],
        frame_t,
    )

    mask_loss = F.binary_cross_entropy_with_logits(
        outputs[
            "motion_mask_logits"
        ],
        mask_target,
    )

    # Explicitly penalize ghost residuals where target motion is low.
    background_loss = (
        (
            1.0
            - mask_target
        )
        * outputs[
            "delta_recon"
        ].abs().mean(
            dim=1,
            keepdim=True,
        )
    ).mean()

    centroid_target = motion_centroid_target(
        delta
    )

    centroid_loss = F.smooth_l1_loss(
        outputs[
            "motion_centroid"
        ],
        centroid_target,
    )

    static_aux = F.mse_loss(
        outputs[
            "static_aux_recon"
        ],
        frame0,
    )

    total = (
        STATIC_AUX_WEIGHT
        * static_aux
        +
        DYNAMIC_LOSS_WEIGHT
        * (
            dynamic_weighted
            +
            DYNAMIC_EDGE_WEIGHT
            * dynamic_edge
        )
        +
        FRAME_LOSS_WEIGHT
        * (
            frame_mse
            +
            FRAME_EDGE_WEIGHT
            * frame_edge
        )
        +
        MASK_LOSS_WEIGHT
        * mask_loss
        +
        BACKGROUND_LOSS_WEIGHT
        * background_loss
        +
        CENTROID_LOSS_WEIGHT
        * centroid_loss
    )

    return {
        "total":
            total,

        "static_aux":
            static_aux,

        "dynamic":
            dynamic_weighted,

        "frame":
            frame_mse,

        "mask":
            mask_loss,

        "background":
            background_loss,

        "centroid":
            centroid_loss,
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    sums = {
        "total": 0.0,
        "static_aux": 0.0,
        "dynamic": 0.0,
        "frame": 0.0,
        "mask": 0.0,
        "background": 0.0,
        "centroid": 0.0,
    }

    count = 0

    for batch in loader:
        frame0 = batch[
            "frame0"
        ].to(
            device,
            non_blocking=True,
        )

        frame_t = batch[
            "frame_t"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type
                == "cuda"
            ),
        ):
            outputs = model(
                frame0,
                frame_t,
            )

            losses = full_losses(
                outputs,
                frame0,
                frame_t,
            )

        n = int(
            frame0.shape[
                0
            ]
        )

        for key in sums:
            sums[
                key
            ] += (
                float(
                    losses[
                        key
                    ].item()
                )
                * n
            )

        count += n

    return {
        key:
            value
            / max(
                count,
                1,
            )
        for key, value
        in sums.items()
    }


# ============================================================
# Visualization
# ============================================================

def tensor_to_pil(
    x,
):
    x = (
        x.detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    arr = (
        x.permute(
            1,
            2,
            0,
        )
        .numpy()
        * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        arr
    )


def delta_to_pil(
    x,
):
    x = (
        x.detach()
        .float()
        .cpu()
        .clamp(
            -1.0,
            1.0,
        )
    )

    arr = (
        (
            x
            * 0.5
            + 0.5
        )
        .permute(
            1,
            2,
            0,
        )
        .numpy()
        * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        arr
    )


def mask_to_pil(
    x,
):
    x = (
        x.detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    arr = (
        x[
            0
        ].numpy()
        * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        arr,
        mode="L",
    ).convert(
        "RGB"
    )


@torch.no_grad()
def save_examples(
    model,
    dataset,
    device,
):
    model.eval()

    ids = (
        np.linspace(
            0,
            len(
                dataset
            )
            - 1,
            min(
                8,
                len(
                    dataset
                ),
            ),
        )
        .round()
        .astype(
            int
        )
    )

    labels = [
        "frame0",
        "frame t",
        "delta true",
        "motion mask",
        "delta recon",
        "frame recon",
    ]

    cell = IMAGE_SIZE
    label_h = 25

    canvas = Image.new(
        "RGB",
        (
            len(
                labels
            )
            * cell,
            len(
                ids
            )
            * (
                cell
                + label_h
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for row, idx in enumerate(
        ids
    ):
        sample = dataset[
            int(
                idx
            )
        ]

        frame0 = sample[
            "frame0"
        ].unsqueeze(
            0
        ).to(
            device
        )

        frame_t = sample[
            "frame_t"
        ].unsqueeze(
            0
        ).to(
            device
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type
                == "cuda"
            ),
        ):
            out = model(
                frame0,
                frame_t,
            )

        delta_true = (
            frame_t[
                0
            ]
            - frame0[
                0
            ]
        )

        images = [
            tensor_to_pil(
                frame0[
                    0
                ]
            ),

            tensor_to_pil(
                frame_t[
                    0
                ]
            ),

            delta_to_pil(
                delta_true
            ),

            mask_to_pil(
                out[
                    "motion_mask"
                ][
                    0
                ]
            ),

            delta_to_pil(
                out[
                    "delta_recon"
                ][
                    0
                ]
            ),

            tensor_to_pil(
                out[
                    "frame_recon"
                ][
                    0
                ]
            ),
        ]

        py = row * (
            cell
            + label_h
        )

        for col, image in enumerate(
            images
        ):
            px = col * cell

            canvas.paste(
                image,
                (
                    px,
                    py,
                ),
            )

            draw.text(
                (
                    px + 4,
                    py + cell + 4,
                ),
                labels[
                    col
                ],
                fill="black",
            )

    canvas.save(
        RECON_PATH
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--static-epochs",
        type=int,
        default=15,
    )

    ap.add_argument(
        "--joint-epochs",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=8e-4,
    )

    args = ap.parse_args()

    seed_everything(
        TRAIN_SEED
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_pair_splits(
        DATA_PATH
    )

    train_static_ds = H5StaticDataset(
        DATA_PATH,
        collect_static_index(
            DATA_PATH,
            splits[
                "train"
            ],
        ),
    )

    val_static_ds = H5StaticDataset(
        DATA_PATH,
        collect_static_index(
            DATA_PATH,
            splits[
                "validation"
            ],
        ),
    )

    train_dynamic_ds = H5DynamicDataset(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits[
                "train"
            ],
        ),
    )

    val_dynamic_ds = H5DynamicDataset(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits[
                "validation"
            ],
        ),
    )

    test_dynamic_ds = H5DynamicDataset(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits[
                "test"
            ],
        ),
    )

    train_static_loader = DataLoader(
        train_static_ds,
        batch_size=min(
            args.batch_size,
            256,
        ),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_static_loader = DataLoader(
        val_static_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    train_dynamic_loader = DataLoader(
        train_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_dynamic_loader = DataLoader(
        val_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_dynamic_loader = DataLoader(
        test_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 96
    )
    print(
        "SPATIAL MOTION AUTOENCODER V5"
    )
    print(
        "=" * 96
    )
    print(
        "Dynamic encoder : CoordConv delta RGB + XY"
    )
    print(
        "Dynamic decoder : candidate delta x learned motion mask"
    )
    print(
        "Auxiliary       : self-supervised motion centroid"
    )
    print(
        "Stage 2         : static encoder FROZEN"
    )
    print(
        "Batch size      :",
        args.batch_size,
    )
    print(
        "Device          :",
        device,
    )
    print(
        "=" * 96
    )

    model = PainStaticDynamicAutoencoderV5(
        static_latent_dim=
            STATIC_LATENT_DIM,

        dynamic_latent_dim=
            DYNAMIC_LATENT_DIM,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Stage 1: static representation
    # --------------------------------------------------------
    print()
    print(
        "STAGE 1: STATIC PRETRAINING"
    )

    for epoch in range(
        1,
        args.static_epochs
        + 1,
    ):
        model.train()

        total = 0.0
        count = 0

        for frame0 in train_static_loader:
            frame0 = frame0.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                z = model.encode_static(
                    frame0
                )

                pred = model.decode_static_aux(
                    z
                )

                loss = F.mse_loss(
                    pred,
                    frame0,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            n = int(
                frame0.shape[
                    0
                ]
            )

            total += float(
                loss.item()
            ) * n

            count += n

        print(
            f"STATIC {epoch:03d}"
            f" | train="
            f"{total/max(count,1):.7f}"
        )

    # Freeze static representation for Stage 2.
    for p in model.static_encoder.parameters():
        p.requires_grad = False

    for p in model.static_aux_decoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
        weight_decay=1e-5,
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------
    print()
    print(
        "STAGE 2: SPATIAL DYNAMIC TRAINING"
    )

    best_val = float(
        "inf"
    )

    history = []

    for epoch in range(
        1,
        args.joint_epochs
        + 1,
    ):
        model.train()

        # Frozen BN layers must remain eval.
        model.static_encoder.eval()
        model.static_aux_decoder.eval()

        total = 0.0
        count = 0

        for batch in train_dynamic_loader:
            frame0 = batch[
                "frame0"
            ].to(
                device,
                non_blocking=True,
            )

            frame_t = batch[
                "frame_t"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                out = model(
                    frame0,
                    frame_t,
                )

                losses = full_losses(
                    out,
                    frame0,
                    frame_t,
                )

            scaler.scale(
                losses[
                    "total"
                ]
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            n = int(
                frame0.shape[
                    0
                ]
            )

            total += float(
                losses[
                    "total"
                ].item()
            ) * n

            count += n

        val = evaluate(
            model,
            val_dynamic_loader,
            device,
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_total":
                    total
                    / max(
                        count,
                        1,
                    ),

                "validation":
                    val,
            }
        )

        print(
            f"JOINT {epoch:03d}"
            f" | train="
            f"{total/max(count,1):.6f}"
            f" | val="
            f"{val['total']:.6f}"
            f" | dyn="
            f"{val['dynamic']:.6f}"
            f" | frame="
            f"{val['frame']:.6f}"
            f" | mask="
            f"{val['mask']:.6f}"
            f" | bg="
            f"{val['background']:.6f}"
            f" | centroid="
            f"{val['centroid']:.6f}"
        )

        if val[
            "total"
        ] < best_val:
            best_val = val[
                "total"
            ]

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "architecture":
                        "spatial_motion_autoencoder_v5",

                    "image_size":
                        IMAGE_SIZE,

                    "static_latent_dim":
                        STATIC_LATENT_DIM,

                    "dynamic_latent_dim":
                        DYNAMIC_LATENT_DIM,

                    "split_seed":
                        SPLIT_SEED,
                },
                MODEL_PATH,
            )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test = evaluate(
        model,
        test_dynamic_loader,
        device,
    )

    save_examples(
        model,
        test_dynamic_ds,
        device,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "architecture":
                    "spatial_motion_autoencoder_v5",

                "best_validation_total":
                    float(
                        best_val
                    ),

                "test_metrics":
                    {
                        key:
                            float(
                                value
                            )
                        for key, value
                        in test.items()
                    },

                "history":
                    history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 96
    )
    print(
        "V5 COMPLETE"
    )
    print(
        "=" * 96
    )
    print(
        "Best validation:",
        best_val,
    )
    print(
        "Test total     :",
        test[
            "total"
        ],
    )
    print(
        "Test dynamic   :",
        test[
            "dynamic"
        ],
    )
    print(
        "Test frame     :",
        test[
            "frame"
        ],
    )
    print(
        "Test background:",
        test[
            "background"
        ],
    )
    print(
        "Model:",
        MODEL_PATH.resolve(),
    )
    print(
        "Recon:",
        RECON_PATH.resolve(),
    )
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
