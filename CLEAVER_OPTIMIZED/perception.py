"""Where is the grasp goal? Knife localisation on the reference image (no learning
across locations, so it generalises to unseen positions).

knife pixel : normalised cross-correlation with one knife template
goal XY     : homography pixel -> table (fixed camera + planar table = calibration),
              fitted on train+validation reference images (goal = knife + constant offset)
"""
import json

import cv2
import h5py
import numpy as np

from CLEAVER_OPTIMIZED import config as cfg

PAD = 4
FOLDER = cfg.OUT / "perception"


def template_from(rgb0, rgb1):
    """Knife box from the frame-0/1 settling difference of ONE trajectory (template only)."""
    diff = (np.abs(rgb1.astype(np.float32) - rgb0.astype(np.float32)).mean(-1) > 9).astype(np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(diff)
    cands = [i for i in range(1, n) if stats[i, cv2.CC_STAT_TOP] > 40]  # not the robot at the top
    i = max(cands, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x0, y0, w, h = stats[i, :4]
    return rgb1[y0 - PAD:y0 + h + PAD, x0 - PAD:x0 + w + PAD].copy()


def locate(img, tmpl):
    res = cv2.matchTemplate(np.ascontiguousarray(img), tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    return np.array([x + tmpl.shape[1] / 2, y + tmpl.shape[0] / 2], np.float32), float(score)


def collect(path, tmpl):
    rows = []
    with h5py.File(path, "r") as f:
        for k in sorted(k for k in f if k.startswith("trajectory_")):
            g = f[k]
            px, score = locate(g["rgb"][cfg.REFERENCE_FRAME], tmpl)
            rows.append(dict(key=k, pair=int(g.attrs["pair_id"]), obj=int(g.attrs.get("object_id", -1)),
                             px=px, score=score, goal=np.asarray(g["target_world"][:2], np.float64)))
    return rows


def fit_h(rows):
    src = np.array([r["px"] for r in rows], np.float32)
    dst = np.array([r["goal"] for r in rows], np.float32)
    return cv2.findHomography(src, dst, 0)[0]


def apply_h(H, px):
    return cv2.perspectiveTransform(np.asarray(px, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def dataset_template():
    with h5py.File(cfg.RGB, "r") as f:
        k0 = next(k for k in sorted(f) if k.startswith("trajectory_") and int(f[k].attrs["pair_id"]) == cfg.TRAIN[0])
        return template_from(f[k0]["rgb"][0], f[k0]["rgb"][cfg.REFERENCE_FRAME])


class GoalPerception:
    """Fitted once on train+validation; then goal_xy(reference_rgb) for any new scene."""

    def __init__(self, template, H):
        self.template, self.H = template, H

    @classmethod
    def fit(cls, pairs=None, folder=FOLDER, note="train+validation"):
        """Homography from the reference images of `pairs` (default: train+validation)."""
        tmpl = dataset_template()
        pool = set(cfg.TRAIN) | set(cfg.VALIDATION) if pairs is None else set(pairs)
        H = fit_h([r for r in collect(cfg.RGB, tmpl) if r["pair"] in pool])
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "knife_template.png"), cv2.cvtColor(tmpl, cv2.COLOR_RGB2BGR))
        (folder / "homography.json").write_text(json.dumps(dict(H=H.tolist(), fitted_on=note)) + "\n")
        return cls(tmpl, H)

    @classmethod
    def load(cls, folder=FOLDER):
        tmpl = cv2.cvtColor(cv2.imread(str(folder / "knife_template.png")), cv2.COLOR_BGR2RGB)
        H = np.array(json.loads((folder / "homography.json").read_text())["H"])
        return cls(tmpl, H)

    def goal_xy(self, reference_rgb):
        px, score = locate(reference_rgb, self.template)
        return apply_h(self.H, [px])[0].astype(np.float64), score
