from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "CLEAVER_OPTIMIZED"

RGB = (
    ROOT / "datasets"
    / "cleaver_extended_workspace_162_complete"
    / "cleaver_same_goal_162.h5"
)

OUT = BASE / "outputs" / "final162"

AE_DIR = OUT / "autoencoder_lightweight128_matched"
AE_MODEL = AE_DIR / "autoencoder.pt"

LATENT = AE_DIR / "cleaver_lightweight_latent128.h5"

SPLIT = json.loads(
    (BASE / "split.json").read_text()
)

TRAIN = set(SPLIT["train"])
TEST = set(SPLIT["test"])
VALIDATION = set(SPLIT["validation"])

IMAGE_SIZE = 256

STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 128

STATIC_PCA_DIM = 16

USE_EXPLICIT_TIME = False

RESERVOIR_SIZE = 1500
SPECTRAL_RADIUS = 0.6
LEAK_RATE = 0.015
RIDGE = 10.0

SEED = 42
