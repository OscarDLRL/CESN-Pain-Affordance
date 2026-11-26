"""Where is the cleaver? Geometric localisation from the reference image only.

1. One knife template (cut once from a single training reference image).
2. Normalised cross-correlation on frame REFERENCE_FRAME -> knife pixel (translation
   equivariant: works at locations never seen in training).
3. Homography pixel -> table XY (fixed camera, planar table = camera calibration).
   Fitted leave-one-location-out on train+validation; then on all 7 locations and
   applied to test and off-grid.
Goal XY = knife XY + constant offset (verified: goal - knife = (0, -65) mm exactly).

    python -m CLEAVER_OPTIMIZED.experiments.knife_localization
"""
import cv2
import h5py
import numpy as np  # noqa: F401

from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn.model import csv_write

from CLEAVER_OPTIMIZED.perception import apply_h, collect, fit_h, template_from  # noqa: E402

REF = cfg.REFERENCE_FRAME


def main():
    out = cfg.OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg.RGB, "r") as f:
        k0 = next(k for k in sorted(f) if k.startswith("trajectory_") and int(f[k].attrs["pair_id"]) == cfg.TRAIN[0])
        tmpl = template_from(f[k0]["rgb"][0], f[k0]["rgb"][REF])
    cv2.imwrite(str(out / "knife_template.png"), cv2.cvtColor(tmpl, cv2.COLOR_RGB2BGR))
    rows = collect(cfg.RGB, tmpl)
    pool_ids = set(cfg.TRAIN) | set(cfg.VALIDATION)
    pool = [r for r in rows if r["pair"] in pool_ids]
    test = [r for r in rows if r["pair"] in set(cfg.TEST)]
    print(f"template {tmpl.shape[:2]} | match score min {min(r['score'] for r in rows):.3f}", flush=True)

    result = []
    print("Leave-one-location-out (train+validation):")
    for obj in sorted({r["obj"] for r in pool}):
        H = fit_h([r for r in pool if r["obj"] != obj])
        held = [r for r in pool if r["obj"] == obj]
        err = np.linalg.norm(apply_h(H, [r["px"] for r in held]) - np.array([r["goal"] for r in held]), axis=1) * 1000
        result.append(dict(eval="LOLO", held_out_object=obj, n=len(held), mean_mm=float(err.mean()), max_mm=float(err.max())))
        print(f"  object {obj}: mean {err.mean():5.2f} mm  max {err.max():5.2f} mm", flush=True)
    H = fit_h(pool)
    for name, rr in [("test", test), ("offgrid", collect(cfg.OFFGRID, tmpl) if cfg.OFFGRID.exists() else [])]:
        if not rr:
            continue
        err = np.linalg.norm(apply_h(H, [r["px"] for r in rr]) - np.array([r["goal"] for r in rr]), axis=1) * 1000
        result.append(dict(eval=name, held_out_object=-1, n=len(rr), mean_mm=float(err.mean()), max_mm=float(err.max())))
        print(f"{name:8s} (fit on 7 locations): mean {err.mean():5.2f} mm  max {err.max():5.2f} mm  n={len(rr)}", flush=True)
    csv_write(out / "knife_localization.csv", result)


if __name__ == "__main__":
    main()
