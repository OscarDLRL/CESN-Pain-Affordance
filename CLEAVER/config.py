from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "datasets/cleaver_same_goal_162_final/cleaver_same_goal_162.h5"
LATENT = ROOT / "datasets/cleaver_latent200/cleaver_latent200.h5"
OUT = ROOT / "CLEAVER/outputs"
AE_DIR = OUT / "autoencoder"
AE_MODEL = AE_DIR / "autoencoder.pt"
EXPLICIT_DIR = OUT / "explicit"
SPATIAL_DIR = OUT / "spatial_only"
SPLIT = json.loads((ROOT / "CLEAVER/split.json").read_text())
TRAIN = set(SPLIT["train"])
TEST = set(SPLIT["test"])
RESTARTS = [(0.02, -0.045), (0.05, -0.045), (0.08, -0.045), (0.02, -0.015), (0.05, -0.015), (0.08, -0.015), (0.02, 0.015), (0.05, 0.015), (0.08, 0.015)]
