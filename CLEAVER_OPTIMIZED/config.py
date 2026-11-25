from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "CLEAVER_OPTIMIZED"

RGB = (
    ROOT / "datasets"
    / "cleaver_extended_workspace_162"
    / "cleaver_same_goal_162.h5"
)

OUT = BASE / "outputs"

AE_DIR = OUT / "autoencoder_latent128"
AE_MODEL = AE_DIR / "autoencoder.pt"

LATENT = OUT / "latents" / "cleaver_latent128.h5"

SPLIT = json.loads(
    (BASE / "split.json").read_text()
)

TRAIN = set(SPLIT["train"])
TEST = set(SPLIT["test"])

IMAGE_SIZE = 256

STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 128

STATIC_PCA_DIM = 16

USE_EXPLICIT_TIME = False

RESERVOIR_SIZE = 1500
SPECTRAL_RADIUS = 0.95
LEAK_RATE = 0.25
RIDGE = 0.5

SEED = 42
