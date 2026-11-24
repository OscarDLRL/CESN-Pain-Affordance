from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LATENT_DATA_PATH = (
    ROOT
    / "datasets"
    / "pain_affordance_latent_v1"
    / "pain_affordance_latent_v1.h5"
)

OUT_DIR = (
    ROOT
    / "CESN"
    / "outputs"
    / "rgb_only_cesn_no_time_v1"
)

MODEL_PATH = (
    OUT_DIR
    / "rgb_only_cesn_no_time_pca16_torch.pt"
)

SUMMARY_PATH = (
    OUT_DIR
    / "training_summary_no_time_pca16.json"
)

PAIN_DIM = 2
STATIC_PCA_DIM = 16

CONTEXT_DIM = (
    PAIN_DIM
    + STATIC_PCA_DIM
)

OUTPUT_DIM = 128

ARCHITECTURE_TAG = (
    "rgb_only_pain_staticpca16_to_zdynamic_no_time_v1"
)

RESERVOIR_SIZE = 1500
SPECTRAL_RADIUS = 0.95
LEAK_RATE = 0.25
INPUT_SCALE = 0.50
BIAS_SCALE = 0.10
RIDGE = 1e-2

RANDOM_SEED = 42

TRAIN_PAIRS = 60
VAL_PAIRS = 10
TEST_PAIRS = 11

SPLIT_SEED = 42
