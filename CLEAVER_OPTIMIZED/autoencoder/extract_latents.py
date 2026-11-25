from __future__ import annotations

from pathlib import Path
import argparse

import h5py
import numpy as np
import torch

from CLEAVER_OPTIMIZED.autoencoder.model import (
    PainStaticDynamicAutoencoderV6Native256Latent128,
)

from CLEAVER_OPTIMIZED import config as cfg

ROOT = Path(__file__).resolve().parents[2]

RGB_DATA = cfg.RGB

MODEL_PATH = cfg.AE_MODEL

OUT_DIR = cfg.LATENT.parent

OUT_H5 = cfg.LATENT


def prepare_rgb_native256(
    rgb,
    device,
):
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

    if tuple(
        x.shape[-2:]
    ) != (
        256,
        256,
    ):
        raise RuntimeError(
            f"Expected 256x256, got "
            f"{tuple(x.shape[-2:])}"
        )

    return x.to(
        device,
        non_blocking=True,
    )


def parse_label(attrs):
    if "pain_label" in attrs:
        return int(
            attrs["pain_label"]
        )

    if "label" in attrs:
        value = attrs["label"]

        if isinstance(
            value,
            bytes,
        ):
            value = value.decode(
                "utf-8"
            )

        if isinstance(
            value,
            str,
        ):
            value_u = value.upper()
            if value_u in {
                "SAFE",
                "0",
            }:
                return 0
            if value_u in {
                "PAIN",
                "HAZARD",
                "1",
            }:
                return 1

        return int(value)

    if "variant" in attrs:
        value = attrs[
            "variant"
        ]
        if isinstance(
            value,
            bytes,
        ):
            value = value.decode(
                "utf-8"
            )

        value_u = str(
            value
        ).upper()

        if value_u == "SAFE":
            return 0

        if value_u in {
            "PAIN",
            "HAZARD",
        }:
            return 1

    raise KeyError(
        "Could not determine SAFE/PAIN label."
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--frame-batch",
        type=int,
        default=128,
        help=(
            "Lower to 64 if native256 extraction exceeds VRAM."
        ),
    )

    args = ap.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA was not detected."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model = (
        PainStaticDynamicAutoencoderV6Native256Latent128(
            static_latent_dim=
                int(
                    checkpoint.get(
                        "static_latent_dim",
                        128,
                    )
                ),
            dynamic_latent_dim=
                int(
                    checkpoint.get(
                        "dynamic_latent_dim",
                        128,
                    )
                ),
        )
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print("=" * 104)
    print("EXTRACT NATIVE256 EXTENDED LATENTS")
    print("=" * 104)
    print("No downscale: RGB enters encoder at native 256x256")
    print("z_static  = 128")
    print("z_dynamic = 128")
    print("Frame batch:", args.frame_batch)
    print("=" * 104)

    with h5py.File(
        RGB_DATA,
        "r",
    ) as src, h5py.File(
        OUT_H5,
        "w",
    ) as dst:

        keys = [
            k
            for k in sorted(
                src.keys()
            )
            if str(k).startswith(
                "trajectory_"
            )
        ]

        for idx, key in enumerate(
            keys,
            start=1,
        ):
            g = src[key]
            rgb = g["rgb"]

            T = int(
                rgb.shape[0]
            )

            frame0 = (
                prepare_rgb_native256(
                    rgb[0],
                    device,
                )
                .unsqueeze(0)
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    z_static = (
                        model.encode_static(
                            frame0
                        )
                    )

            z_static_np = (
                z_static[0]
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            chunks = []

            for start in range(
                0,
                T,
                args.frame_batch,
            ):
                end = min(
                    T,
                    start
                    + args.frame_batch,
                )

                frame_t = (
                    torch.from_numpy(
                        np.asarray(
                            rgb[start:end],
                            dtype=np.uint8,
                        )
                    )
                    .permute(0, 3, 1, 2)
                    .to(device)
                    .float()
                    .div_(255.0)
                )

                frame0_batch = (
                    frame0.expand(
                        frame_t.shape[0],
                        -1,
                        -1,
                        -1,
                    )
                )

                delta = (
                    frame_t
                    - frame0_batch
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=True,
                    ):
                        z_dynamic = (
                            model.encode_dynamic(
                                delta
                            )
                        )

                chunks.append(
                    z_dynamic
                    .float()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

            z_dynamic_np = (
                np.concatenate(
                    chunks,
                    axis=0,
                )
            )

            out_g = (
                dst.create_group(
                    key
                )
            )

            for (
                attr_name,
                attr_value,
            ) in g.attrs.items():
                try:
                    out_g.attrs[
                        attr_name
                    ] = attr_value
                except TypeError:
                    out_g.attrs[
                        attr_name
                    ] = str(
                        attr_value
                    )

            pain_label = parse_label(
                g.attrs
            )

            out_g.attrs[
                "pair_id"
            ] = int(
                g.attrs[
                    "pair_id"
                ]
            )

            out_g.attrs[
                "pain_label"
            ] = int(
                pain_label
            )

            out_g.attrs[
                "frames"
            ] = T

            out_g.attrs[
                "native_image_size"
            ] = 256

            out_g.create_dataset(
                "z_static",
                data=z_static_np,
                compression="gzip",
            )

            out_g.create_dataset(
                "z_dynamic",
                data=z_dynamic_np,
                compression="gzip",
            )

            out_g.create_dataset(
                "time",
                data=np.linspace(
                    0.0,
                    1.0,
                    T,
                    dtype=np.float32,
                ),
            )

            print(
                f"[{idx:03d}/{len(keys):03d}] "
                f"{key:<18s}"
                f" T={T:>3d}"
                f" label={pain_label}"
                f" z_static={z_static_np.shape}"
                f" z_dynamic={z_dynamic_np.shape}"
            )

    print()
    print("=" * 104)
    print("DONE")
    print("Latent dataset:", OUT_H5.resolve())
    print("=" * 104)


from CLEAVER_OPTIMIZED.config import RGB, AE_MODEL, LATENT
RGB_DATA = cfg.RGB
MODEL_PATH = cfg.AE_MODEL
OUT_DIR = cfg.LATENT.parent
OUT_H5 = cfg.LATENT

if __name__ == "__main__":
    main()
