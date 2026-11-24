from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LATENT_DATA_PATH = (
    ROOT / "datasets"
    / "pain_affordance_latent_v6_noanchor_native256_latent200_final70"
    / "pain_affordance_latent_v6_noanchor_native256_latent200_final70.h5"
)

RGB_DATA_PATH = (
    ROOT / "datasets"
    / "pain_affordance_162"
    / "pain_affordance_162.h5"
)

OUT_DIR = (
    ROOT / "CESN" / "outputs"
    / "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70"
)

MODEL_PATH = (
    OUT_DIR
    / "rgb_grasp_context_cesn_v6_native256_latent200_ridge05_final70.pt"
)

STATIC_PCA_DIM = 16
RESERVOIR_DIM = 1500
RIDGE = 0.5
SPECTRAL_RADIUS = 0.95
LEAK_RATE = 0.25
INPUT_SCALE = 0.50
BIAS_SCALE = 0.10
SEED = 42

GRASP_X_SCALE = 0.145
GRASP_Y_SCALE = 0.080

TRAIN_PAIRS = 70
VAL_PAIRS = 0
TEST_PAIRS = 11

SAFE_TARGET_LOCAL_X = +0.065
PAIN_TARGET_LOCAL_X = -0.065

TIME_INDEX_SCALE = 100.0
