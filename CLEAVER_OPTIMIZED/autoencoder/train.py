from __future__ import annotations

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from CLEAVER_OPTIMIZED.autoencoder.model import (
    PainStaticDynamicAutoencoderV6Native256Latent128,
)

from CLEAVER_OPTIMIZED.autoencoder.training_utils import (
    DATA_PATH,
    SPLIT_SEED,
    TRAIN_SEED,
    collect_static_index,
    collect_dynamic_index,
    seed_everything,
    full_losses,
    rgb_to_device,
    evaluate,
)

from CLEAVER_OPTIMIZED import config as cfg

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = cfg.AE_DIR

MODEL_PATH = cfg.AE_MODEL

SUMMARY_PATH = (
    OUT_DIR
    / "training_summary_native256_latent128_extended.json"
)

FINAL_TRAIN_PAIRS = 54
TEST_PAIRS = 9
IMAGE_SIZE = 256
STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 128


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
        self.frame0_cache = {}

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return len(self.index)

    @staticmethod
    def prepare_uint8(rgb):
        return torch.from_numpy(
            np.asarray(rgb, dtype=np.uint8)
        ).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        key, t = self.index[idx]
        g = self._file()[key]

        if key not in self.frame0_cache:
            self.frame0_cache[key] = (
                self.prepare_uint8(g["rgb"][0])
            )

        return {
            "key": key,
            "frame0": self.frame0_cache[key],
            "frame_t": self.prepare_uint8(g["rgb"][t]),
        }


def build_final_splits(path):
    from CLEAVER_OPTIMIZED.config import TRAIN, TEST
    return {"train": TRAIN, "test": TEST}



@torch.no_grad()
def build_static_cache(model, dataset_path, pair_ids, device):

    model.static_encoder.eval()
    model.static_aux_decoder.eval()

    keys = collect_static_index(dataset_path, pair_ids)

    latent_cache = {}
    mse_cache = {}

    with h5py.File(dataset_path, "r") as h5:
        for start in range(0, len(keys), 16):

            chunk = keys[start:start + 16]

            rgb = np.stack([
                h5[key]["rgb"][0]
                for key in chunk
            ])

            frame0 = (
                torch.from_numpy(rgb)
                .permute(0, 3, 1, 2)
                .to(device)
                .float()
                .div_(255.0)
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                z = model.encode_static(frame0)
                reconstruction = model.decode_static_aux(z)

            mse = (
                (reconstruction.float() - frame0)
                .square()
                .flatten(1)
                .mean(dim=1)
            )

            for i, key in enumerate(chunk):
                latent_cache[key] = z[i].detach().clone()
                mse_cache[key] = mse[i].detach().clone()

    print(
        f"STATIC CACHE COMPLETE | "
        f"trajectories={len(latent_cache)}",
        flush=True,
    )

    return latent_cache, mse_cache


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
            splits["train"],
        ),
    )

    train_dynamic_ds = H5DynamicDataset256(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits["train"],
        ),
    )

    test_dynamic_ds = H5DynamicDataset256(
        DATA_PATH,
        collect_dynamic_index(
            DATA_PATH,
            splits["test"],
        ),
    )

    static_loader = DataLoader(
        train_static_ds,
        batch_size=min(
            args.batch_size,
            128,
        ),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )

    dynamic_loader = DataLoader(
        train_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1,
    )

    test_loader = DataLoader(
        test_dynamic_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1,
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
    print("V6 NATIVE256 EXTENDED - NO FUTURE ANCHOR")
    print("=" * 104)
    print("Input/output resolution : 256x256 NATIVE")
    print("Downscale before encoder: NONE")
    print("z_static                : 128")
    print("z_dynamic               : 128")
    print("Train pairs             :", len(splits["train"]))
    print("Test pairs              :", len(splits["test"]))
    print("Static epochs           :", args.static_epochs)
    print("Dynamic epochs          :", args.joint_epochs)
    print("Batch size              :", args.batch_size)
    print("Device                  :", device)
    print("=" * 104)

    model = (
        PainStaticDynamicAutoencoderV6Native256Latent128(
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

    latent_cache, mse_cache = build_static_cache(
        model,
        DATA_PATH,
        splits["train"],
        device,
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
            frame0 = rgb_to_device(
                batch["frame0"], device
            )

            frame_t = rgb_to_device(
                batch["frame_t"], device
            )

            cached_z = torch.stack([
                latent_cache[key]
                for key in batch["key"]
            ])

            cached_mse = torch.stack([
                mse_cache[key]
                for key in batch["key"]
            ])

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
                    cached_z_static=cached_z,
                    cached_static_mse=cached_mse,
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
                "autoencoder_v6_noanchor_native256_latent128_extended",
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
                        "train"
                    ]
                ),
            "test_pairs":
                sorted(
                    splits[
                        "test"
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
                    "autoencoder_v6_noanchor_native256_latent128_extended",
                "image_size": 256,
                "native_resolution": True,
                "z_static": 128,
                "z_dynamic": 128,
                "train_pairs": len(splits["train"]),
                "test_pairs": len(splits["test"]),
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
    print("NATIVE256 EXTENDED COMPLETE")
    print("=" * 104)
    print("Test total     :", test["total"])
    print("Test dynamic   :", test["dynamic"])
    print("Test frame     :", test["frame"])
    print("Test background:", test["background"])
    print("Model          :", MODEL_PATH.resolve())
    print("=" * 104)


from CLEAVER_OPTIMIZED.config import RGB, AE_DIR, AE_MODEL
DATA_PATH = RGB
OUT_DIR = cfg.AE_DIR
MODEL_PATH = cfg.AE_MODEL
SUMMARY_PATH = OUT_DIR / "training_summary.json"

if __name__ == "__main__":
    main()
