"""Causal low-pass filter on z_dynamic before the CESN (no AE retraining).

z_f[t] = (1 - a) * z_f[t-1] + a * z[t]     (uses only past frames: deployable online)
Writes outputs/autoencoder/<ae-run>_ema<a>/latents.h5 for each a, ready for
cesn.train and analysis.latent_probe.

    python -m CLEAVER_OPTIMIZED.experiments.causal_filter --ae-run v6_ref1 --alphas 0.5 0.3
"""
import argparse

import h5py
import numpy as np

from CLEAVER_OPTIMIZED import config as cfg


def ema(z, a):
    out = np.empty_like(z)
    out[0] = z[0]
    for t in range(1, len(z)):
        out[t] = (1 - a) * out[t - 1] + a * z[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-run", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.3])
    a = ap.parse_args()
    src_path = cfg.OUT / "autoencoder" / a.ae_run / "latents.h5"
    for alpha in a.alphas:
        assert 0 < alpha <= 1
        dst_dir = cfg.OUT / "autoencoder" / f"{a.ae_run}_ema{int(round(alpha * 100)):02d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(src_path, "r") as src, h5py.File(dst_dir / "latents.h5", "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["causal_ema_alpha"] = alpha
            for key in src:
                src.copy(src[key], dst, name=key)
                if key.startswith("trajectory_"):
                    z = dst[key]["z_dynamic"][:]
                    del dst[key]["z_dynamic"]
                    dst[key].create_dataset("z_dynamic", data=ema(z, alpha), compression="gzip")
        print("WROTE", dst_dir, flush=True)


if __name__ == "__main__":
    main()
