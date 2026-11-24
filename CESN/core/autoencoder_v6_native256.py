from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvEncoder256(nn.Module):
    """
    Native 256x256 image -> latent.
    Five stride-2 stages:
      256 -> 128 -> 64 -> 32 -> 16 -> 8
    """

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        self.fc = nn.Linear(
            256 * 8 * 8,
            int(latent_dim),
        )

    def forward(self, x):
        h = self.features(x)
        return self.fc(h.flatten(1))


class CoordDynamicEncoder256(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder256(
            in_channels=5,
            latent_dim=latent_dim,
        )

    def forward(self, delta):
        B, _, H, W = delta.shape

        x = torch.linspace(
            -1.0, 1.0, W,
            device=delta.device,
            dtype=delta.dtype,
        ).view(1, 1, 1, W).expand(B, 1, H, W)

        y = torch.linspace(
            -1.0, 1.0, H,
            device=delta.device,
            dtype=delta.dtype,
        ).view(1, 1, H, 1).expand(B, 1, H, W)

        return self.encoder(
            torch.cat([delta, x, y], dim=1)
        )


class StaticAuxDecoder256(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(
            int(latent_dim),
            256 * 8 * 8,
        )

        self.up1 = UpBlock(256, 128)  # 8->16
        self.up2 = UpBlock(128, 64)   # 16->32
        self.up3 = UpBlock(64, 32)    # 32->64
        self.up4 = UpBlock(32, 16)    # 64->128
        self.up5 = UpBlock(16, 16)    # 128->256

        self.rgb_head = nn.Conv2d(
            16, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        return torch.sigmoid(self.rgb_head(h))


class MotionGatedDynamicDecoder256(nn.Module):
    """
    Native 256x256 residual decoder.

    The main decoder is truly 256-native.
    The small direct-detail branch remains 128x128 internally and is
    upsampled before fusion so parameter count does not explode.
    """

    def __init__(
        self,
        static_latent_dim=128,
        dynamic_latent_dim=128,
    ):
        super().__init__()

        fused_dim = (
            int(static_latent_dim)
            + int(dynamic_latent_dim)
        )

        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.SiLU(inplace=True),
            nn.Linear(384, 256),
            nn.SiLU(inplace=True),
        )

        self.fc = nn.Linear(
            256,
            256 * 8 * 8,
        )

        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.up5 = UpBlock(16, 16)

        self.delta_head = nn.Conv2d(
            16, 3, kernel_size=3, padding=1
        )

        self.mask_head = nn.Conv2d(
            16, 1, kernel_size=3, padding=1
        )

        # Keep parameter count comparable to baseline.
        self.detail_head = nn.Linear(
            fused_dim,
            3 * 128 * 128,
        )

        self.detail_scale = nn.Parameter(
            torch.tensor(0.05)
        )

    def forward(self, z_dynamic, z_static):
        fused = torch.cat(
            [z_dynamic, z_static],
            dim=1,
        )

        h = self.fuse(fused)
        h = self.fc(h).view(
            -1, 256, 8, 8
        )

        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)

        coarse_delta = self.delta_head(h)

        detail = self.detail_head(
            fused
        ).view(
            -1, 3, 128, 128
        )

        detail = F.interpolate(
            detail,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        candidate_delta = torch.tanh(
            coarse_delta
            + self.detail_scale * detail
        )

        motion_mask_logits = self.mask_head(h)
        motion_mask = torch.sigmoid(
            motion_mask_logits
        )

        delta_recon = (
            candidate_delta
            * motion_mask
        )

        return {
            "candidate_delta": candidate_delta,
            "motion_mask_logits": motion_mask_logits,
            "motion_mask": motion_mask,
            "delta_recon": delta_recon,
        }


class PainStaticDynamicAutoencoderV6Native256(nn.Module):
    """
    V6 native-256, no future anchor.

    Input/output image resolution: 256x256.
    Latents remain:
      z_static  = 128
      z_dynamic = 128

    No task, grasp XY, qpos, TCP or future frames enter the encoder.
    """

    def __init__(
        self,
        static_latent_dim=128,
        dynamic_latent_dim=128,
    ):
        super().__init__()

        self.static_latent_dim = int(
            static_latent_dim
        )
        self.dynamic_latent_dim = int(
            dynamic_latent_dim
        )

        self.static_encoder = ConvEncoder256(
            in_channels=3,
            latent_dim=self.static_latent_dim,
        )

        self.dynamic_encoder = CoordDynamicEncoder256(
            latent_dim=self.dynamic_latent_dim,
        )

        self.static_aux_decoder = StaticAuxDecoder256(
            latent_dim=self.static_latent_dim,
        )

        self.dynamic_decoder = MotionGatedDynamicDecoder256(
            static_latent_dim=self.static_latent_dim,
            dynamic_latent_dim=self.dynamic_latent_dim,
        )

        self.motion_centroid_head = nn.Sequential(
            nn.Linear(self.dynamic_latent_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def encode_static(self, frame0):
        return self.static_encoder(frame0)

    def encode_dynamic(self, delta):
        return self.dynamic_encoder(delta)

    def decode_static_aux(self, z_static):
        return self.static_aux_decoder(z_static)

    def decode_dynamic_full(
        self,
        z_dynamic,
        z_static,
    ):
        return self.dynamic_decoder(
            z_dynamic,
            z_static,
        )

    def decode_dynamic(
        self,
        z_dynamic,
        z_static,
    ):
        return self.decode_dynamic_full(
            z_dynamic,
            z_static,
        )["delta_recon"]

    def forward(self, frame0, frame_t):
        delta = frame_t - frame0

        z_static = self.encode_static(frame0)
        z_dynamic = self.encode_dynamic(delta)

        static_aux_recon = (
            self.decode_static_aux(z_static)
        )

        dynamic_outputs = (
            self.decode_dynamic_full(
                z_dynamic,
                z_static,
            )
        )

        delta_recon = dynamic_outputs[
            "delta_recon"
        ]

        frame_recon = torch.clamp(
            frame0 + delta_recon,
            0.0,
            1.0,
        )

        motion_centroid = (
            self.motion_centroid_head(
                z_dynamic
            )
        )

        return {
            "z_static": z_static,
            "z_dynamic": z_dynamic,
            "delta": delta,
            "static_aux_recon":
                static_aux_recon,
            "candidate_delta":
                dynamic_outputs[
                    "candidate_delta"
                ],
            "motion_mask_logits":
                dynamic_outputs[
                    "motion_mask_logits"
                ],
            "motion_mask":
                dynamic_outputs[
                    "motion_mask"
                ],
            "delta_recon": delta_recon,
            "frame_recon": frame_recon,
            "motion_centroid":
                motion_centroid,
        }
