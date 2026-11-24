from __future__ import annotations

from pathlib import Path
import argparse

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from CESN.core.autoencoder_v5 import (
    PainStaticDynamicAutoencoderV5,
)


ROOT = Path(__file__).resolve().parents[2]

RGB_DATA = (
    ROOT
    / "datasets"
    / "pain_affordance_162"
    / "pain_affordance_162.h5"
)

MODEL_PATH = (
    ROOT
    / "CESN"
    / "outputs"
    / "pain_affordance_spatial_motion_autoencoder_v5"
    / "best_spatial_motion_autoencoder_v5.pt"
)

OUT_DIR = (
    ROOT
    / "datasets"
    / "pain_affordance_latent_v5"
)

OUT_H5 = (
    OUT_DIR
    / "pain_affordance_latent_v5.h5"
)

IMAGE_SIZE = 128


def prepare_rgb(
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
        .permute(
            2,
            0,
            1,
        )
        .float()
        / 255.0
    )

    x = F.interpolate(
        x.unsqueeze(0),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return x.to(
        device,
        non_blocking=True,
    )


def parse_label(
    attrs,
):
    if "pain_label" in attrs:
        return int(
            attrs[
                "pain_label"
            ]
        )

    if "label" in attrs:
        value = attrs[
            "label"
        ]

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
        default=512,
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

    static_dim = int(
        checkpoint.get(
            "static_latent_dim",
            128,
        )
    )

    dynamic_dim = int(
        checkpoint.get(
            "dynamic_latent_dim",
            128,
        )
    )

    model = PainStaticDynamicAutoencoderV5(
        static_latent_dim=
            static_dim,

        dynamic_latent_dim=
            dynamic_dim,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print()
    print(
        "=" * 96
    )
    print(
        "EXTRACT V5 LATENTS"
    )
    print(
        "=" * 96
    )
    print(
        "RGB dataset :",
        RGB_DATA,
    )
    print(
        "Model       :",
        MODEL_PATH,
    )
    print(
        "Output      :",
        OUT_H5,
    )
    print(
        "Device      :",
        device,
    )
    print(
        "Frame batch :",
        args.frame_batch,
    )
    print(
        "=" * 96
    )

    with h5py.File(
        RGB_DATA,
        "r",
    ) as src, h5py.File(
        OUT_H5,
        "w",
    ) as dst:

        keys = [
            key
            for key in sorted(
                src.keys()
            )
            if str(
                key
            ).startswith(
                "trajectory_"
            )
        ]

        for idx, key in enumerate(
            keys,
            start=1,
        ):
            g = src[
                key
            ]

            rgb = g[
                "rgb"
            ]

            T = int(
                rgb.shape[
                    0
                ]
            )

            frame0 = prepare_rgb(
                rgb[
                    0
                ],
                device,
            ).unsqueeze(
                0
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
                z_static[
                    0
                ]
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            z_dynamic_chunks = []

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

                frames = []

                for t in range(
                    start,
                    end,
                ):
                    frames.append(
                        prepare_rgb(
                            rgb[
                                t
                            ],
                            device,
                        )
                    )

                frame_t = torch.stack(
                    frames,
                    dim=0,
                )

                frame0_batch = (
                    frame0.expand(
                        frame_t.shape[
                            0
                        ],
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

                z_dynamic_chunks.append(
                    z_dynamic
                    .float()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

            z_dynamic_np = np.concatenate(
                z_dynamic_chunks,
                axis=0,
            )

            out_g = dst.create_group(
                key
            )

            # Preserve useful original attrs.
            for attr_name, attr_value in g.attrs.items():
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

            # Kept for compatibility with older utilities.
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
    print(
        "=" * 96
    )
    print(
        "DONE"
    )
    print(
        "=" * 96
    )
    print(
        "Latent dataset:",
        OUT_H5.resolve(),
    )
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
