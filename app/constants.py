"""Constants for the crab-dance app.

Robot facts come from the jumper repository (KingKongRobotics/jumper, Liuzu V1.6):
  * assets/jumper/urdf/jumper/urdf/jumper.urdf -- joint ranges
  * tasks/jumper/common/constants.py -- HOME standing pose, STAND_Z
Mapping gains are ours; they are grouped here so a whole behaviour can be
flipped by editing one number.
"""

import math
from pathlib import Path

# ---------------------------------------------------------------- paths ----
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
ASSETS_DIR = PROJECT_DIR / "assets"
JUMPER_DIR = ASSETS_DIR / "jumper"
JUMPER_XML = JUMPER_DIR / "jumper.xml"
CACHE_DIR = Path.home() / ".crab_dance_cache"

JUMPER_RAW_BASE = (
    "https://raw.githubusercontent.com/KingKongRobotics/jumper/main/assets/jumper"
)
# e.g. set JUMPER_MIRROR=https://ghproxy.net/https://raw.githubusercontent.com
MIRROR_ENV = "JUMPER_MIRROR"

SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".m4v", ".flv", ".ts", ".mpg", ".mpeg", ".3gp"}

# ------------------------------------------------- multi-person staging ----
MAX_PERSONS = 6        # Sports2D detection count (each adds inference time)
MAX_CRABS = 6          # crabs rendered in multi mode (one CrabModel each)
STAGE_SPREAD = 0.55    # m, total lateral span crabs line up over

# ------------------------------------------------------- robot structure ----
CLAWS = ("LF", "RF")          # front arms = the two claws, 5 joints each
LEGS = ("LM", "RM", "LR", "RR")  # 4 walking legs, 3 joints each
ALL_JOINTS = tuple(
    f"{p}_J{i}_joint" for p in CLAWS for i in range(5)
) + tuple(
    f"{p}_J{i}_joint" for p in LEGS for i in range(3)
)

# Revolute joint limits in radians, straight from jumper.urdf -- except the
# claw J2 outward ends, widened by +90 deg on user request so overhead claw
# sweeps are not cut short (preview-only; the real servos stop at the URDF
# value).
_J2_EXTRA = math.pi / 2.0
JOINT_RANGE: dict[str, tuple[float, float]] = {
    "LF_J0_joint": (-2.75, 0.7),
    "LF_J1_joint": (-4.75, 1.4),
    "LF_J2_joint": (-2.2, 1.65 + _J2_EXTRA),
    "LF_J3_joint": (-2.1, 1.65),
    "LF_J4_joint": (-1.6, 0.1),
    "RF_J0_joint": (-0.7, 2.75),
    "RF_J1_joint": (-4.75, 1.4),
    "RF_J2_joint": (-1.65 - _J2_EXTRA, 2.1),
    "RF_J3_joint": (-1.65, 2.1),
    "RF_J4_joint": (0.1, 1.6),
    "LM_J0_joint": (0.75, 1.0),
    "LM_J1_joint": (-2.4, 1.5),
    "LM_J2_joint": (-2.4, 1.8),
    "RM_J0_joint": (-1.0, 0.75),
    "RM_J1_joint": (-1.5, 2.4),
    "RM_J2_joint": (-1.8, 2.4),
    "LR_J0_joint": (-0.6, 2.75),
    "LR_J1_joint": (-2.4, 1.6),
    "LR_J2_joint": (-2.4, 1.8),
    "RR_J0_joint": (-2.75, 0.6),
    "RR_J1_joint": (-1.6, 2.4),
    "RR_J2_joint": (-1.8, 2.4),
}

# Preview/SANDBOX-only widening of the LEFT-MIDDLE leg's hip yaw: the
# front-leg stance below rotates LM_J0 negative (forward) past the URDF
#'s +0.75 rad stop. The preview is a puppet and the training sandbox uses
# the wider MJCF limits, so the visual stance is allowed here.
JOINT_RANGE["LM_J0_joint"] = (-0.60, 1.0)

# Global front-leg stance (user: extend the two front walking legs further
# forward -- LM -, RM + moves both feet toward the claws by ~6 cm at 25 deg).
FRONT_LEG_J0_STANCE_DEG = 30.0

# Calibrated standing pose (tasks/jumper/common/constants.py, line 133).
HOME: dict[str, float] = {
    "LF_J0_joint": -0.5507, "LF_J1_joint": -1.0658,
    "LF_J2_joint": -0.4408, "LF_J3_joint": -1.374, "LF_J4_joint": 0.0,
    "RF_J0_joint": 0.5524, "RF_J1_joint": -1.066,
    "RF_J2_joint": 0.4414, "RF_J3_joint": 1.4173, "RF_J4_joint": 0.0,
    "LM_J0_joint": 0.0012, "LM_J1_joint": 0.5356, "LM_J2_joint": -1.8271,
    "RM_J0_joint": -0.0012, "RM_J1_joint": -0.5376, "RM_J2_joint": 1.8313,
    "LR_J0_joint": 0.6313, "LR_J1_joint": 0.5534, "LR_J2_joint": -1.8648,
    "RR_J0_joint": -0.6326, "RR_J1_joint": -0.5548, "RR_J2_joint": 1.8678,
}

# Documented standing height (m); the app re-measures it at load time.
STAND_Z_DOC = 0.10647

# Free-joint body height range (m), relative to measured standing height.
Z_MIN_FACTOR = 0.55   # deep crouch
Z_MAX_FACTOR = 1.08   # standing stretch: dance a touch taller than HOME

# User-facing standing-height slider (multiplies the commanded base_z in the
# view). The walking legs reach ~1.62x standing height, so 1.55 stays planted.
HEIGHT_FACTOR_MIN = 0.70
HEIGHT_FACTOR_MAX = 1.55

# ------------------------------------------------------- retarget gains ----
# Torso euler -> crab body rotation. Human angles in degrees, gains are
# deg-of-crab per deg-of-human, then clamped to CMD_*_MAX (deg).
K_ROLL = 0.6
K_PITCH = 0.6
K_YAW = 0.70           # a touch UNDER 1:1 (user: amplitude smaller than the
                       # detection's -- the monocular estimate overshoots)
CMD_ROLL_MAX = 25.0
CMD_PITCH_MAX = 20.0
CMD_YAW_MAX = 360.0    # full signed yaw (front/back aware) -- slow turns track
BODY_SMOOTH_WIN = 5    # frames of moving average on body commands
# Body-yaw slew limit (user): the crab's turn rate capped under the
# dancer's so jittery estimates cannot whip the body around.
BODY_YAW_MAX_SPEED_DEG = 240.0

# Facing (front/back) from keypoint LATERALITY: anatomically-labelled left
# vs right shoulders/hips swap image sides when the dancer turns around
# (facing camera, the dancer's LEFT side appears image-RIGHT). Width
# foreshortening alone cannot tell front-center from back-center; the
# ordering can. Hysteresis threshold on the normalised lateral offset.
FACING_FLIP_LAT = 0.18  # |left-right| / shoulder-width to flip the bit

# Joint speed limit for the final claw commands (user: 31.4 rad/s).
JOINT_SPEED_MAX_RAD = 31.4

# Body command signs, set by eye for a dancer FACING the camera with the
# default preview camera (azimuth 135 deg). If a video shows the backing
# dancer, or one channel looks mirrored, flip the offending sign here.
BODY_SIGN_ROLL = -1.0
BODY_SIGN_PITCH = -1.0
BODY_SIGN_YAW = +1.0

# Leg-length influence on standing height: leg_ratio ~1.8 (short legs) to
# ~2.3 (long legs) maps to a height scale of LEG_SCALE_LO..HI.
LEG_RATIO_LO, LEG_RATIO_HI = 1.8, 2.3
LEG_SCALE_LO, LEG_SCALE_HI = 0.94, 1.05
# Lower-body tracking validity. Sports2D gives 0 confidence for keypoints it
# could not find; when the legs are out of frame (upper-body-only shots) the
# standing height must NOT collapse to the minimum. Frames whose legs are
# confident but geometrically inconsistent are also treated as invalid.
LEG_CONF_MIN = 0.35
LEG_LEN_RATIO_MIN, LEG_LEN_RATIO_MAX = 0.5, 1.8
LEG_RATIO_FALLBACK = 2.05     # no usable leg frames: assume a normal build
LEG_FALLBACK_CROUCH = 0.9     # long invalid stretches -> normal standing

# Crouch: hip-height/leg-length, 1=standing. Smoothed, then mapped to body z.
CROUCH_LO, CROUCH_HI = 0.45, 0.95
CROUCH_SMOOTH_WIN = 7

# Training stance (user): during physics training the claws kept scraping
# the deck, so the reference is built with the body pitched nose-up and the
# two FRONT ground legs (LM/RM) yawed slightly forward (support shifts
# under the pitched body). Raw pose_dance pitch is negative = nose-up here
# (measured: the claw lowest point rises by ~35 mm at -10 deg).
TRAIN_STANCE_PITCH_DEG = -10.0
TRAIN_FRONT_J0_DEG = 12.0    # LM -this, RM +this -> front feet forward
# Claw J0 FORWARD offset (user: the arms were sitting at the backward
# J0 stops; at the opening, +30/-30 still buried LF in the LM leg --
# a 50 deg forward swing measured 17 -> 0 arm-leg contacts).
TRAIN_CLAW_J0_FWD_DEG = 50.0   # LF -, RF +
# After the forward swing the training stance clamps the claw J0 to the
# measured collision-free window (the raw dance can reach the base).
TRAIN_CLAW_J0_SAFE_DEG = {"LF": (-110.0, -20.0), "RF": (20.0, 110.0)}

# Claw shoulder-yaw (J0) limits (user): the LEFT claw is capped at +30 deg,
# the RIGHT at -30 deg (inward stops); the outward ends keep the URDF range.
# Applied to the COMMANDS (video, manual, rest).
# Claw forearm (J1) usable lower limit (user): never below -110 deg, in
# the main preview and in the training sandbox.
CLAW_J1_MIN_DEG = -110.0

# User-specified video mapping limits (2026-11-19):
#   LF J0 -70..+15, +15 = leftmost, -70 = straight ahead
#   RF J0 -15..+70, +70 = straight ahead, -15 = rightmost
CLAW_J0_LIMIT_DEG = {
    "LF": (-70.0, 15.0),
    "RF": (-15.0, 70.0),
}

# Claw mapping, per the user's joint semantics (round-9 revision):
#   J0 <- shoulder yaw  (horizontal pointing, wrist lateral offset proxy)
#   J1 <- FOREARM direction (elbow->wrist elevation: hanging 0 .. overhead 180)
#   J2 <- elbow bend x split   }  the two joints TOGETHER express the
#   J3 <- elbow bend x split   }  human elbow angle
#   J4 <- hand open/close (wrist-overhead heuristic)
CLAW_SHOULDER_YAW_GAIN = 1.2     # rad J0 per unit lateral offset
# Static biases read off the REAL model (tests/probe_rest_pose.py):
#   J0 0 -- plain HOME is already mirror-symmetric about the sagittal
#      plane (6 mm residual); any UNIFORM bias rotates both claws the same
#      way and breaks the mirror (the old -45 deg did exactly that; the
#      user measured +45 deg of sweep was needed to undo it).
#   J1 -30 -- uniform delta that levels the claw arm plane (upper_arm ->
#      forearm -> palm) with the ground; measured optimum -29 deg (5.6 deg
#      residual), user eyeballed -35.
CLAW_J0_OFFSET_DEG = 0.0         # static J0 bias, uniform, rad = -ccw*this
# Anti-crossing spread (user): both claws' J0 move apart -- LF +, RF - --
# so the two claws stop clipping through each other at the rest/pose range.
CLAW_J0_SPREAD_DEG = 10.0
# No-cross window for the VIDEO mapping: LF J0 >= -75 deg and RF J0
# <= +75 deg. A per-frame keep-out pass in retarget removes the remaining
# mesh contacts, so J0 keeps a 105-degree travel instead of the old 40.
CLAW_J0_CROSS_LIMIT_DEG = 75.0
CLAW_J1_OFFSET_DEG = 30.0        # static J1 bias, uniform, applied NEGATED
CLAW_FOREARM_GAIN = 1.7          # rad J1 per unit forearm elevation
CLAW_ELBOW_J2_GAIN = 0.9         # rad J2 per unit elbow bend (share 1)
CLAW_ELBOW_J3_GAIN = 1.1         # rad J3 per unit elbow bend (share 2)
CLAW_J3_OFFSET_DEG = 60.0        # user-tuned default bias on J3
CLAW_SMOOTH_WIN = 3

# GMR-style claw IK (github.com/YanjieZe/GMR), direction-first variant
# (user directive): the human arm enters as ARM-SEGMENT DIRECTIONS + elbow
# configuration, NOT as a hand pose. J2 and J3 turn about PARALLEL axes, so
# the claw elbow has ~3 effective DoF -- exactly enough for the achievable
# core of that representation:
#   HARD  upper-arm ("elbow") direction   -- 2 constraints (J0, J1)
#   HARD  elbow bend angle (u1.u2)        -- 1 constraint (J2+J3 plane)
#   SOFT  forearm off-plane pointing      -- morphology cannot always do it
#   SOFT  palm position (stage B, leashed) -- the hand may give up
# The soft tiers are damped so their residual is absorbed locally and never
# pushed back into the elbow.
CLAW_IK_ITERS_A = 5             # stage A (directions) iterations per frame
CLAW_IK_ITERS_B = 3             # stage B (soft hand) iterations per frame
CLAW_IK_W_UARM = 60.0           # upper-arm / elbow-direction weight (hard)
CLAW_IK_W_BEND = 60.0           # elbow bend-angle weight (hard)
CLAW_IK_W_FORE_TAN = 30.0       # forearm pointing, full vector (soft tier)
CLAW_IK_W_HAND = 25.0           # palm position, soft
CLAW_IK_W_POSTURE = 1.0         # stay-near-HOME null-space regulariser
CLAW_IK_HAND_LEASH = 0.9        # rad stage B may move q away from stage A
# Shoulder SWING allocation (user): the lateral swing drives J2 and J3
# TOGETHER, each at half response (CLAW_IK_SWING_SPLIT), along each
# joint's probed outward direction -- J0/J1 stop carrying the whole swing
# (the frozen-J2 experiment saturated them). CLAW_IK_J2_LIMIT_DEG is the
# fold box around HOME again.
CLAW_IK_J2_LIMIT_DEG = 50.0
CLAW_IK_SWING_SPLIT = 0.5     # each of J2/J3 carries this share of swing
CLAW_IK_W_SWING = 10.0        # prior weight (hard objectives still rule)
CLAW_IK_MAX_STEP = 0.35         # rad per iteration (trust region)
CLAW_IK_VSTEP = 0.30            # rad per frame velocity cap
CLAW_IK_RETRY_DIR_DEG = 25.0    # hard-objective error that triggers re-seed
# Uniform rotation of all claw targets about the vertical, calibrated to
# the user's anchor pose (ghost pitch 90 = upper arm dead forward -> the
# claw upper arm at world azimuth ~0).
CLAW_IK_FRAME_ROT_DEG = 45.0
CLAW_IK_RHO_MIN, CLAW_IK_RHO_MAX = 0.15, 1.0   # soft-hand extent clamp
CLAW_IK_FLOOR_Z = 0.008         # palm TARGET kept above the deck (m)
GHOST_ARM_LEN = 0.065           # human ghost upper/fore-arm length (m)

# Side-swing mapping limit (user): the claws may swing from vertical (0) up
# to this angle OUTWARD, never across the body -- the same arc as the manual
# ghost's roll slider. Video channels are converted to the outward-positive
# convention before clamping; the manual slider convention is negative-out.
CLAW_SWING_LIMIT_DEG = 120.0

# Floor keep-out (real constraint, not just the target clamp): after the
# direction/hand stages the solver folds the claw up until the FINGER TIP
# body and palm site clear the deck. The tip mesh extends up to ~27 mm
# below its body origin depending on orientation, hence the clearance. The
# constraint is one-sided, so poses above the floor are untouched (pose
# accuracy where it matters, no digging into the ground near it).
CLAW_FLOOR_TIP_Z = 0.025        # finger-tip body origin clearance (m)
CLAW_FLOOR_PALM_Z = 0.010       # palm site clearance (m)
CLAW_IK_W_FLOOR = 60.0          # floor objective weight
CLAW_IK_FLOOR_ITERS = 6         # floor stage iterations
CLAW_IK_FLOOR_LEASH = 0.6       # rad the floor stage may move from stage B

# ------------------------------------------------ anchor calibration ----
# USER CORRESPONDENCE -- NOT a mapping. When the ghost arm is at
# (shoulder pitch, shoulder roll, elbow bend) = (90, 90, 180), the left
# claw should be at the JOINT-SWEEP TOOL pose J0..J4 = (-60, 0, 30, 20, 2)
# degrees RELATIVE TO THE REST POSE (the sweep tool's slider convention),
# the right claw exactly mirrored. The claw mapping itself stays the
# direction-tracking IK; this correspondence only calibrates a CONSTANT
# joint-space offset (measured once by solving the anchor pose) so the
# relationship holds exactly. J4 is out of the mapping (user: rest value),
# so the anchor's 2 deg is informational only.
CLAW_ANCHOR_GHOST = (90.0, 90.0, 180.0)
CLAW_ANCHOR_DELTA_DEG = (-60.0, 0.0, 30.0, 20.0)   # J0..J3, LF-space

# Spin segments: with the full signed yaw (front/back aware), a scripted
# sweep is only needed where the dancer spins FASTER than the estimator can
# count -- windows whose smoothed yaw rate exceeds SPIN_RATE_DEG become a
# pill (scripted, rate-capped, exactly the measured number of whole turns).
# A slow turn or a held back-facing pose is NOT a spin; the plain yaw curve
# tracks it.
SPIN_RATE_DEG = 150.0       # deg/s sustained to read as a spin
SPIN_MIN_FRAMES = 8
SPIN_MERGE_FRAMES = 30      # rate bursts closer than this are one cluster
SPIN_MAX_SPEED_DEG = 450.0  # deg/s scripted-turn rate cap

# Jump segments (same treatment as turns): jump01 above the threshold for
# at least this many frames becomes a clickable pill -- on = the crab hops
# there, off = the hop is suppressed. The hop itself is scripted: crouch
# (发力) -> airborne -> settle.
JUMP_SEG_ON = 0.5
JUMP_SEG_MIN_FRAMES = 2
JUMP_CROUCH_DEPTH = 0.35   # crouch dip, fraction of the hop height

# Jump / leg-raise recognition. Rises are measured against the whole-clip
# standing baseline (median over near-straight-legged frames), normalised by
# torso length (jump) / leg length (raise).
JUMP_RISE_ON, JUMP_RISE_FULL = 0.15, 0.45   # torso units, hips AND both feet
JUMP_SMOOTH_WIN = 3
JUMP_HOP_GAIN = 0.75          # hop height in units of z_stand at full jump
RAISE_ON, RAISE_FULL = 0.20, 0.55           # leg units, one ankle high
RAISE_SMOOTH_WIN = 5
LEG_RAISE_GAIN = 0.40         # foot lift target in units of z_stand (the
                               # folded leg maxes out ~0.46x, so stay under)
LEG_RAISE_REAR = 0.40         # same-side rear leg lifts this fraction

# The preview camera's azimuth (135 deg) is a LOOK-direction convention in
# MuJoCo 4 (verified by eye: +135 showed the rear). With claws facing
# azimuth ~0 at yaw=0, this offset makes "yaw 0" mean "facing the camera",
# matching the ghost figure's reference direction.
CRAB_FRONT_YAW_OFFSET_DEG = -45.0

# ------------------------------------------------ crab view background ----
# Presets for the crab preview backdrop. Everything except "default" is
# composited in the UI over the MuJoCo render (segmentation mask separates
# robot+floor from the background); "floor" tints the dance floor to match.
BG_DEFAULT = "default"
CRAB_BACKGROUNDS: list[dict] = [
    {"id": "default", "name": "默认"},
    {"id": "deepsea", "name": "深海",
     "top": (6, 30, 48), "bottom": (16, 84, 102),
     "floor": (0.05, 0.13, 0.16, 1.0)},
    {"id": "sunset", "name": "落日",
     "top": (66, 32, 88), "bottom": (244, 146, 84),
     "floor": (0.23, 0.14, 0.10, 1.0)},
    {"id": "beach", "name": "沙滩",
     "top": (92, 158, 216), "bottom": (222, 238, 246),
     "floor": (0.76, 0.66, 0.48, 1.0)},
    {"id": "neon", "name": "霓虹",
     "top": (13, 10, 32), "bottom": (88, 26, 96),
     "floor": (0.06, 0.05, 0.10, 1.0)},
]

# ---------------------------------------------------- video / playback ----
PLAYBACK_FPS = 30          # UI timer; video frames are looked up by time
CRAB_RENDER_FPS = 30       # throttle for the MuJoCo offscreen render
CURVE_UPDATE_EVERY = 2     # matplotlib cursor redraw every N ticks
