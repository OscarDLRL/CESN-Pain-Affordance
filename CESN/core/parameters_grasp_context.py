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
    / "rgb_grasp_context_cesn_v1"
)

MODEL_PATH = (
    OUT_DIR
    / "rgb_grasp_context_cesn_v1.pt"
)

SUMMARY_PATH = (
    OUT_DIR
    / "training_summary.json"
)

# ============================================================
# Dataset split
# ============================================================
SPLIT_SEED = 42

TRAIN_PAIRS = 60
VAL_PAIRS = 10
TEST_PAIRS = 11

FINAL_TRAIN_PAIRS = (
    TRAIN_PAIRS
    + VAL_PAIRS
)

# ============================================================
# Context definition
# ============================================================
# Context supplied to the forward CESN:
#
#   [task_onehot(2), z_static_pca(16), grasp_x, grasp_y]
#
# During inversion:
#   known    = z_static_pca
#   unknown  = task + grasp_x + grasp_y
#
TASK_DIM = 2
STATIC_PCA_DIM = 16
GRASP_DIM = 2

CONTEXT_DIM = (
    TASK_DIM
    + STATIC_PCA_DIM
    + GRASP_DIM
)

OUTPUT_DIM = 128

# ============================================================
# Physical knife / grasp geometry
# ============================================================
KNIFE_X_VALUES = (
    -0.08,
     0.00,
     0.08,
)

KNIFE_Y_VALUES = (
     0.08,
     0.00,
    -0.08,
)

# Yaw = 0 in the dataset.
#
# SAFE / non-painful:
# handle side = local +X
#
# PAIN / painful:
# blade side = local -X
#
SAFE_GRASP_OFFSET_X = +0.065
PAIN_GRASP_OFFSET_X = -0.065

# Full support:
# knife_x in [-0.08,+0.08]
# +/- 0.065 grasp offset
# -> grasp_x in [-0.145,+0.145]
#
GRASP_X_SCALE = 0.145
GRASP_Y_SCALE = 0.08

GRASP_X_MIN = -0.145
GRASP_X_MAX = +0.145

GRASP_Y_MIN = -0.08
GRASP_Y_MAX = +0.08

# ============================================================
# CESN hyperparameters
# Reuse the selected no-time settings.
# ============================================================
RESERVOIR_SIZE = 1500

SPECTRAL_RADIUS = 0.95
LEAK_RATE = 0.25
INPUT_SCALE = 0.50
BIAS_SCALE = 0.10

RIDGE = 1e-2
DENSITY = 0.05

RANDOM_SEED = 42

ARCHITECTURE_TAG = (
    "rgb_grasp_context_cesn_v1_no_time"
)
