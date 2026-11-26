"""V6 dual-encoder autoencoder (best AE: former `V6Compact128`).

static encoder : reference RGB          -> z_static  (scene, cleaver, background)
dynamic encoder: (frame_t - reference)  -> z_dynamic (what changed)
reconstruction : reference + mask * delta(z_dynamic, z_static)

Module names are identical to V6Compact128, so its checkpoints load directly
(with detail_size=64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvEncoder256(nn.Module):
    """256 -> 128 -> 64 -> 32 -> 16 -> 8, then linear to the latent."""

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        layers, c = [], in_channels
        for out in (32, 64, 128, 256, 256):
            layers += [nn.Conv2d(c, out, 3, 2, 1), nn.BatchNorm2d(out), nn.SiLU(inplace=True)]
            c = out
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(256 * 8 * 8, int(latent_dim))

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class CoordDynamicEncoder256(nn.Module):
    """Delta image + (x, y) coordinate channels."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder256(in_channels=5, latent_dim=latent_dim)

    def forward(self, delta):
        B, _, H, W = delta.shape
        x = torch.linspace(-1, 1, W, device=delta.device, dtype=delta.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        y = torch.linspace(-1, 1, H, device=delta.device, dtype=delta.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        return self.encoder(torch.cat([delta, x, y], 1))


def _up_stack():
    return [UpBlock(256, 128), UpBlock(128, 64), UpBlock(64, 32), UpBlock(32, 16), UpBlock(16, 16)]


class StaticAuxDecoder256(nn.Module):
    """Auxiliary only: forces z_static to describe the reference image."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(int(latent_dim), 256 * 8 * 8)
        self.up1, self.up2, self.up3, self.up4, self.up5 = _up_stack()
        self.rgb_head = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        h = self.up5(self.up4(self.up3(self.up2(self.up1(h)))))
        return torch.sigmoid(self.rgb_head(h))


class MotionGatedDynamicDecoder256(nn.Module):
    def __init__(self, static_latent_dim=128, dynamic_latent_dim=128, detail_size=64):
        super().__init__()
        fused = int(static_latent_dim) + int(dynamic_latent_dim)
        self.fuse = nn.Sequential(nn.Linear(fused, 384), nn.SiLU(inplace=True), nn.Linear(384, 256), nn.SiLU(inplace=True))
        self.fc = nn.Linear(256, 256 * 8 * 8)
        self.up1, self.up2, self.up3, self.up4, self.up5 = _up_stack()
        self.delta_head = nn.Conv2d(16, 3, 3, padding=1)
        self.mask_head = nn.Conv2d(16, 1, 3, padding=1)
        # Global linear image templates (latent -> pixels). Suspected source of
        # arm-correlated artefacts on static edges; detail_size=0 removes it.
        self.detail_size = int(detail_size)
        if self.detail_size:
            self.detail_head = nn.Linear(fused, 3 * self.detail_size ** 2)
            self.detail_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, z_dynamic, z_static):
        fused = torch.cat([z_dynamic, z_static], 1)
        h = self.fc(self.fuse(fused)).view(-1, 256, 8, 8)
        h = self.up5(self.up4(self.up3(self.up2(self.up1(h)))))
        delta = self.delta_head(h)
        if self.detail_size:
            s = self.detail_size
            detail = self.detail_head(fused).view(-1, 3, s, s)
            delta = delta + self.detail_scale * F.interpolate(detail, size=h.shape[-2:], mode="bilinear", align_corners=False)
        candidate = torch.tanh(delta)
        logits = self.mask_head(h)
        mask = torch.sigmoid(logits)
        return {"candidate_delta": candidate, "motion_mask_logits": logits, "motion_mask": mask, "delta_recon": candidate * mask}


class V6DualEncoder(nn.Module):
    """Only RGB enters the encoders: no task, grasp XY, qpos, TCP or future frames."""

    def __init__(self, static_latent_dim=128, dynamic_latent_dim=128, detail_size=64, state_dim=0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.static_latent_dim = int(static_latent_dim)
        self.dynamic_latent_dim = int(dynamic_latent_dim)
        self.static_encoder = ConvEncoder256(3, self.static_latent_dim)
        self.dynamic_encoder = CoordDynamicEncoder256(self.dynamic_latent_dim)
        self.static_aux_decoder = StaticAuxDecoder256(self.static_latent_dim)
        self.dynamic_decoder = MotionGatedDynamicDecoder256(self.static_latent_dim, self.dynamic_latent_dim, detail_size)
        self.motion_centroid_head = nn.Sequential(nn.Linear(self.dynamic_latent_dim, 64), nn.SiLU(inplace=True), nn.Linear(64, 2), nn.Tanh())
        # Training-only supervision: robot state (TCP xyz, gripper width) from z_dynamic.
        # Never used at inference; forces z_dynamic to encode gripper geometry.
        if self.state_dim:
            self.state_head = nn.Sequential(nn.Linear(self.dynamic_latent_dim, 128), nn.SiLU(inplace=True), nn.Linear(128, self.state_dim))

    def encode_static(self, reference):
        return self.static_encoder(reference)

    def encode_dynamic(self, delta):
        return self.dynamic_encoder(delta)

    def decode_static_aux(self, z_static):
        return self.static_aux_decoder(z_static)

    def decode_dynamic_full(self, z_dynamic, z_static):
        return self.dynamic_decoder(z_dynamic, z_static)

    def decode(self, reference, z_dynamic, z_static):
        out = self.decode_dynamic_full(z_dynamic, z_static)
        return torch.clamp(reference + out["delta_recon"], 0.0, 1.0), out

    def forward(self, reference, frame_t, cached_z_static=None, cached_static_mse=None, z_dynamic=None):
        """cached_* are used in stage 2, when the static branch is frozen. z_dynamic can be
        passed when it was encoded together with other frames (same BatchNorm batch)."""
        delta = frame_t - reference
        z_static = self.encode_static(reference) if cached_z_static is None else cached_z_static
        if z_dynamic is None:
            z_dynamic = self.encode_dynamic(delta)
        static_aux = self.decode_static_aux(z_static) if cached_static_mse is None else None
        frame_recon, dyn = self.decode(reference, z_dynamic, z_static)
        return {
            "z_static": z_static,
            "z_dynamic": z_dynamic,
            "delta": delta,
            "static_aux_recon": static_aux,
            "cached_static_mse": cached_static_mse,
            **dyn,
            "frame_recon": frame_recon,
            "motion_centroid": self.motion_centroid_head(z_dynamic),
            "state": self.state_head(z_dynamic) if self.state_dim else None,
        }


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = V6DualEncoder(ckpt.get("static_latent_dim", 128), ckpt.get("dynamic_latent_dim", 128),
                          ckpt.get("detail_size", 64), ckpt.get("state_dim", 0))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
