"""Single source of truth for CLEAVER_OPTIMIZED. Nothing here imports from CLEAVER."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "CLEAVER_OPTIMIZED"
OUT = BASE / "outputs"

RGB = ROOT / "datasets" / "cleaver_extended_workspace_162_complete" / "cleaver_same_goal_162.h5"
OFFGRID = ROOT / "datasets" / "cleaver_offgrid_48" / "cleaver_same_goal_162.h5"
OFFGRID_PLAN = OFFGRID.parent / "plan.json"  # 4 off-grid cleaver positions x 6 robot starts

SPLIT_FILE = BASE / "split.json"
SPLIT = json.loads(SPLIT_FILE.read_text())
TRAIN = sorted(map(int, SPLIT["train"]))
VALIDATION = sorted(map(int, SPLIT["validation"]))
TEST = sorted(map(int, SPLIT["test"]))

# ---------------------------------------------------------------- data
NUM_FRAMES = 223
IMAGE_SIZE = 256
# Frame 0 is captured before the physics settles: the cleaver drops ~14.7 mm
# between frame 0 and 1 and stays static afterwards until it is grasped.
# Using frame 0 as reference puts that drop into every delta (moving knife
# edges). Every reference image and every sequence starts here.
REFERENCE_FRAME = 1
SEQ_LEN = NUM_FRAMES - REFERENCE_FRAME  # 222 steps: frames 1..222

# ---------------------------------------------------------------- autoencoder (V6 dual encoder)
STATIC_LATENT_DIM = 128
DYNAMIC_LATENT_DIM = 128
DETAIL_SIZE = 64  # 0 disables the global linear detail branch

# ---------------------------------------------------------------- CESN (configuration B = C161 + ridge 30)
# Conditioning vector = [pain one-hot (2), z_static (128), grasp XY (2)].
# Only SAFE/PAIN is inferred; visual and spatial inputs are known.
# No explicit time. Output feedback = previous z_dynamic (observed when a camera
# frame exists, own prediction otherwise).
# Existing configuration retained from earlier CV; classification-only CV must be reported separately.
RESERVOIR_SIZE = 1500
SPECTRAL_RADIUS = 0.6
LEAK_RATE = 0.015
FEEDBACK_GAIN = 1.0
RIDGE = 30.0
SEED = 42

# ---------------------------------------------------------------- online detector (frozen before test)
AE_RUN = "v6_ref1"
OBS_EVERY = 5        # camera/AE observation every 5 control steps (4 Hz at 20 Hz control)
DETECT_TAU = 0.1     # PAIN score threshold, chosen on CV (100% detection, 0% false alarms)
DETECT_CONSECUTIVE = 2

INVERSION_FRAMES = [1, 2, 3, 5, 10, 12, 23, 45, 67, 112, 157, SEQ_LEN]
