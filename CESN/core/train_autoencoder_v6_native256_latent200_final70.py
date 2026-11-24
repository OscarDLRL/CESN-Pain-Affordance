from __future__ import annotations

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from CESN.core.autoencoder_v6_native256_latent200 import (
    PainStaticDynamicAutoencoderV6Native256Latent200,
)

from CESN.core.autoencoder_training_utils import (
    DATA_PATH,
    SPLIT_SEED,
    TRAIN_SEED,
    collect_static_index,
    collect_dynamic_index,
    seed_everything,
    full_losses,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT / "CESN" / "outputs"
    / "pain_affordance_autoencoder_v6_noanchor_native256_latent200_final70"
)

MODEL_PATH = (
    OUT_DIR
    / "autoencoder_v6_noanchor_native256_latent200_final70.pt"
)

SUMMARY_PATH = (
    OUT_DIR
    / "training_summary_native256_latent200_final70.json"
)

FINAL_TRAIN_PAIRS = 70
TEST_PAIRS = 11
IMAGE_SIZE = 256
STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 200


def prepare_rgb_native256(rgb):
    x = (
        torch.from_numpy(
            np.asarray(
                rgb,
                dtype=np.uint8,
            )
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    if tuple(x.shape[-2:]) != (256, 256):
        raise RuntimeError(
            f"Expected native RGB 256x256, got {tuple(x.shape[-2:])}"
        )

    return x


class H5StaticDataset256(Dataset):
    def __init__(self, path, index):
        self.path = str(path)
        self.index = list(index)
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.path, "r"
            )
        return self._h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key = self.index[idx]
        return prepare_rgb_native256(
            self._file()[key]["rgb"][0]
        )


class H5DynamicDataset256(Dataset):
    def __init__(self, path, index):
        self.path = str(path)
        self.index = list(index)
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.path, "r"
            )
        return self._h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, t = self.index[idx]
        g = self._file()[key]

        return {
            "frame0":
                prepare_rgb_native256(
                    g["rgb"][0]
                ),
            "frame_t":
                prepare_rgb_native256(
                    g["rgb"][t]
                ),
        }


def build_final_splits(path):
    pair_ids = set()

    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            if str(key).startswith(
                "trajectory_"
            ):
                pair_ids.add(
                    int(
                        h5[key].attrs[
                            "pair_id"
                        ]
                    )
                )

    pair_ids = np.asarray(
        sorted(pair_ids),
        dtype=np.int64,
    )

    if len(pair_ids) != 81:
        raise RuntimeError(
            f"Expected 81 pairs, got {len(pair_ids)}"
        )

    rng = np.random.default_rng(
        SPLIT_SEED
    )
    rng.shuffle(pair_ids)

    train70 = set(
        pair_ids[:70].tolist()
    )
    test11 = set(
        pair_ids[70:].tolist()
    )

    return {
        "train70": train70,
        "test11": test11,
    }


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
        default=60,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Native 256 uses substantially more VRAM. "
            "Lower to 64 or 32 if needed."
        ),
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=8e-4,
    )

    args = ap.parse_args()

    seed_everything(TRAIN_SEED)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_final_splits(
        DATA_PATH
    )

    train_static_ds = H5StaticDataset256(
        DATA_PATH,
        collect_static_index(
            DATA_PATH,
            splits["train70"],
        ),
    )

    train_dynamic_ds = H5DynamicDataset256(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits["train70"],
        ),
    )

    test_dynamic_ds = H5DynamicDataset256(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits["test11"],
        ),
    )

    static_loader = DataLoader(
        train_static_ds,
        batch_size=min(
            args.batch_size,
            128,
        ),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    dynamic_loader = DataLoader(
        train_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA was not detected."
        )

    print()
    print("=" * 104)
    print("V6 NATIVE256 FINAL70 - NO FUTURE ANCHOR")
    print("=" * 104)
    print("Input/output resolution : 256x256 NATIVE")
    print("Downscale before encoder: NONE")
    print("z_static                : 128")
    print("z_dynamic               : 200")
    print("Train pairs             : 70")
    print("Test pairs              : 11")
    print("Static epochs           :", args.static_epochs)
    print("Dynamic epochs          :", args.joint_epochs)
    print("Batch size              :", args.batch_size)
    print("Device                  :", device)
    print("=" * 104)

    model = (
        PainStaticDynamicAutoencoderV6Native256Latent200(
            static_latent_dim=
                STATIC_LATENT_DIM,
            dynamic_latent_dim=
                DYNAMIC_LATENT_DIM,
        )
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    # --------------------------------------------------------
    # Stage 1: static pretraining
    # --------------------------------------------------------
    for epoch in range(
        1,
        args.static_epochs + 1,
    ):
        model.train()
        total = 0.0
        count = 0

        for frame0 in static_loader:
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
                enabled=True,
            ):
                z = model.encode_static(
                    frame0
                )
                pred = (
                    model.decode_static_aux(z)
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
                frame0.shape[0]
            )

            total += (
                float(loss.item())
                * n
            )
            count += n

        print(
            f"STATIC {epoch:03d}"
            f" | train="
            f"{total/max(count,1):.7f}"
        )

    # Freeze static representation exactly as baseline protocol.
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

    history = []

    # --------------------------------------------------------
    # Stage 2: dynamic training
    # --------------------------------------------------------
    for epoch in range(
        1,
        args.joint_epochs + 1,
    ):
        model.train()
        model.static_encoder.eval()
        model.static_aux_decoder.eval()

        total = 0.0
        count = 0

        for batch in dynamic_loader:
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
                enabled=True,
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

            scaler.scale(
                losses["total"]
            ).backward()

            scaler.step(
                optimizer
            )
            scaler.update()

            n = int(
                frame0.shape[0]
            )

            total += (
                float(
                    losses[
                        "total"
                    ].item()
                )
                * n
            )

            count += n

        train_total = (
            total
            / max(
                count,
                1,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_total":
                    float(
                        train_total
                    ),
            }
        )

        print(
            f"JOINT {epoch:03d}"
            f" | train="
            f"{train_total:.6f}"
        )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------
    test = evaluate(
        model,
        test_loader,
        device,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "architecture":
                "autoencoder_v6_noanchor_native256_latent200_final70",
            "image_size": 256,
            "native_resolution": True,
            "static_latent_dim":
                STATIC_LATENT_DIM,
            "dynamic_latent_dim":
                DYNAMIC_LATENT_DIM,
            "split_seed":
                SPLIT_SEED,
            "final_train_pairs":
                sorted(
                    splits[
                        "train70"
                    ]
                ),
            "test_pairs":
                sorted(
                    splits[
                        "test11"
                    ]
                ),
            "static_epochs":
                int(
                    args.static_epochs
                ),
            "joint_epochs":
                int(
                    args.joint_epochs
                ),
        },
        MODEL_PATH,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "architecture":
                    "autoencoder_v6_noanchor_native256_latent200_final70",
                "image_size": 256,
                "native_resolution": True,
                "z_static": 128,
                "z_dynamic": 200,
                "train_pairs": 70,
                "test_pairs": 11,
                "test_metrics": {
                    k: float(v)
                    for k, v
                    in test.items()
                },
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("NATIVE256 FINAL70 COMPLETE")
    print("=" * 104)
    print("Test total     :", test["total"])
    print("Test dynamic   :", test["dynamic"])
    print("Test frame     :", test["frame"])
    print("Test background:", test["background"])
    print("Model          :", MODEL_PATH.resolve())
    print("=" * 104)


if __name__ == "__main__":
    main()
