"""Human -> crab retargeting math.

Works purely from 2D keypoints (image pixels, y down). Every channel is an
honest approximation of a quantity a single uncalibrated camera can see --
that is stated per channel below, because torso yaw in particular is a weak
estimate (shoulder-width foreshortening) and is deliberately mapped to the
crab with a small gain.

Pipeline: keypoints -> HumanChannels (curves shown in the UI) ->
CrabCommandTable (per-sample-time joint/body commands consumed by playback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from app.core.claw_calib import apply_calibration
from app.constants import (
    BODY_SIGN_PITCH, BODY_SIGN_ROLL, BODY_SIGN_YAW, BODY_SMOOTH_WIN,
    BODY_YAW_MAX_SPEED_DEG,
    CLAWS, CLAW_ELBOW_J2_GAIN, CLAW_ELBOW_J3_GAIN, CLAW_FOREARM_GAIN,
    CLAW_IK_FLOOR_Z, CLAW_IK_FRAME_ROT_DEG, CLAW_IK_HAND_LEASH,
    CLAW_FLOOR_PALM_Z, CLAW_FLOOR_TIP_Z, CLAW_IK_FLOOR_ITERS,
    CLAW_IK_FLOOR_LEASH, CLAW_IK_W_FLOOR, CLAW_SWING_LIMIT_DEG,
    CLAW_IK_ITERS_A, CLAW_IK_ITERS_B, CLAW_IK_J2_LIMIT_DEG,
    CLAW_IK_MAX_STEP, CLAW_IK_RETRY_DIR_DEG,
    CLAW_IK_RHO_MAX, CLAW_IK_RHO_MIN, CLAW_IK_VSTEP, CLAW_IK_W_BEND,
    CLAW_IK_W_FORE_TAN, CLAW_IK_W_HAND, CLAW_IK_W_POSTURE,
    CLAW_IK_W_SWING, CLAW_IK_W_UARM,
    CLAW_IK_SWING_SPLIT,
    CLAW_ANCHOR_DELTA_DEG, CLAW_ANCHOR_GHOST,
    CLAW_J0_CROSS_LIMIT_DEG, CLAW_J0_LIMIT_DEG, CLAW_J0_OFFSET_DEG,
    CLAW_J0_SPREAD_DEG,
    CLAW_J1_MIN_DEG,
    CLAW_J1_OFFSET_DEG, CLAW_J3_OFFSET_DEG,
    CLAW_SHOULDER_YAW_GAIN,
    CLAW_SMOOTH_WIN, CMD_PITCH_MAX, CMD_ROLL_MAX, CMD_YAW_MAX, CROUCH_HI,
    CROUCH_LO, CROUCH_SMOOTH_WIN, FACING_FLIP_LAT, GHOST_ARM_LEN,
    HOME, JUMP_CROUCH_DEPTH, JUMP_RISE_FULL, JUMP_RISE_ON,
    JUMP_SEG_MIN_FRAMES, JUMP_SEG_ON, JUMP_SMOOTH_WIN, JOINT_SPEED_MAX_RAD,
    K_PITCH, K_ROLL,
    K_YAW, LEG_RAISE_GAIN,
    LEG_RAISE_REAR, LEG_RATIO_HI, LEG_RATIO_LO, LEG_SCALE_HI, LEG_SCALE_LO,
    LEG_CONF_MIN, LEG_FALLBACK_CROUCH, LEG_LEN_RATIO_MAX,
    LEG_LEN_RATIO_MIN, LEG_RATIO_FALLBACK,
    RAISE_FULL, RAISE_ON, RAISE_SMOOTH_WIN, SPIN_MAX_SPEED_DEG,
    SPIN_MIN_FRAMES, SPIN_MERGE_FRAMES, SPIN_RATE_DEG, Z_MAX_FACTOR,
    Z_MIN_FACTOR,
)
from app.core.crab_model import CrabModel

# Canonical keypoint names the pipeline layer must deliver.
KP_NAMES = (
    "nose", "neck", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "hip_l", "hip_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r",
)

# Bone pairs for the overlay skeleton (indices into KP_NAMES order).
SKELETON_EDGES = (
    ("neck", "nose"),
    ("shoulder_l", "shoulder_r"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("hip_l", "hip_r"),
    ("shoulder_l", "hip_l"), ("shoulder_r", "hip_r"),
    ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"),
    ("neck", "hip_l"), ("neck", "hip_r"),
)

# Ghost-figure basis at yaw 0 -- MUST match CrabView._ghost_segments (d_h
# renders screen-left = the dancer's image-right; user-calibrated).
_GHOST_ZH = np.array([0.0, 0.0, 1.0])
_GHOST_D_H = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
_GHOST_D_F = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)   # toward camera


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-preserving moving average (mirrored edges)."""
    if win <= 1:
        return x.copy()
    n = len(x)
    half = win // 2
    out = np.empty_like(x)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = x[lo:hi]
        valid = seg[~np.isnan(seg)]
        out[i] = valid.mean() if len(valid) else np.nan
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    """Row-wise normalisation kept for callers outside this module."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-9)


@dataclass
class Keypoints:
    """(N, 2) pixel coords + (N,) confidence per canonical keypoint."""
    xyz: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,2)
    conf: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N,)
    times: np.ndarray | None = None  # (N,) sample times, seconds

    def xy(self, name: str) -> np.ndarray:
        if name in self.xyz:
            return self.xyz[name]
        n = len(self.times) if self.times is not None else 0
        return np.full((n, 2), np.nan)


@dataclass
class HumanChannels:
    """Everything the UI plots. Angles in degrees, positions normalised."""
    times: np.ndarray
    # torso
    torso_pitch: np.ndarray   # + = bending toward camera (nose dips), approx
    torso_roll: np.ndarray    # + = leaning to image right (raw image frame)
    torso_yaw: np.ndarray     # weak estimate from shoulder foreshortening
    lean_raw: np.ndarray      # in-image trunk tilt from vertical (debug)
    # arms, per side (deg / normalised)
    elevation_l: np.ndarray   # 0 arm along trunk, 180 overhead (legacy combo)
    elevation_r: np.ndarray
    shoulder_pitch_l: np.ndarray   # 0 in image plane .. 90 pointing at camera
    shoulder_pitch_r: np.ndarray
    shoulder_roll_l: np.ndarray    # signed lateral raise, + = toward image right
    shoulder_roll_r: np.ndarray
    bend_l: np.ndarray        # elbow flexion 0 straight .. 180 folded
    bend_r: np.ndarray
    forearm_l: np.ndarray     # forearm elevation 0 hanging .. 180 overhead
    forearm_r: np.ndarray
    hand_up_l: np.ndarray     # wrist height above mid-hip, torso lengths
    hand_up_r: np.ndarray
    side_l: np.ndarray        # wrist x - body centre x, torso lengths (+ = img right)
    side_r: np.ndarray
    # body scale / height
    crouch: np.ndarray        # 1 legs extended, ->0.4 deep squat
    leg_ratio: float          # leg length / torso length (scale invariant)
    torso_len_px: float
    quality: np.ndarray       # per-frame fraction of confident keypoints
    # air gestures (0..1)
    jump: np.ndarray          # both feet off the ground, hips above baseline
    raise_l: np.ndarray       # left ankle lifted toward its hip
    raise_r: np.ndarray       # right ankle lifted
    # Front/back from keypoint laterality + the resulting full signed yaw
    # (deg, continuous/unwrapped). None on legacy channel sets.
    torso_facing: np.ndarray = None
    torso_yaw_full: np.ndarray = None


@dataclass
class CrabCommandTable:
    """Per-sample-time crab commands; playback interpolates between rows."""
    times: np.ndarray
    base_z: np.ndarray        # m
    roll: np.ndarray          # deg
    pitch: np.ndarray         # deg
    yaw: np.ndarray           # deg
    leg_sway: np.ndarray      # rad offset for leg J0
    claw: dict[str, np.ndarray]  # joint name -> rad command
    jump: np.ndarray = None       # 0..1 hop strength (added on top of base_z)
    raise_l: np.ndarray = None    # 0..1 left leg raise
    raise_r: np.ndarray = None    # 0..1 right leg raise
    foot_lift: dict[str, np.ndarray] | None = None  # leg -> lift target (m)
    jump_mask: np.ndarray = None  # bool: frames inside a detected jump
    spin_mask: np.ndarray = None   # bool: frames inside a suspected turn
    yaw_spin: np.ndarray = None    # deg: scripted full-circle sweep there
    spin_segs: list | None = None  # detected turns: {i0,i1,dir,deg,mode}


# --------------------------------------------------------------- helpers ----
def _fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-interpolate NaN runs shorter than max_gap, hold the rest."""
    n = len(x)
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() < 2:
        return np.zeros_like(x) if not good.any() else np.nan_to_num(x, nan=float(x[good][0]))
    out = np.interp(idx, idx[good], x[good])
    # Re-blank long gaps so they read as "unknown", then hold previous value.
    bad = np.isnan(x)
    if bad.any():
        # runs of bad
        start = None
        for i in range(n + 1):
            if i < n and np.isnan(x[i]):
                if start is None:
                    start = i
            elif start is not None:
                if i - start > max_gap:
                    out[start:i] = out[start - 1] if start > 0 else out[i]
                start = None
    return out


def compute_human_channels(kps: Keypoints, gap_frames: int = 15) -> HumanChannels:
    t = kps.times if kps.times is not None else np.arange(len(kps.xy("nose")))
    n = len(t)

    def fill2(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.ndim != 2:
            return _fill_gaps(a, gap_frames)
        return np.stack(
            [_fill_gaps(a[:, i], gap_frames) for i in range(a.shape[1])], axis=1
        )

    P = {name: fill2(kps.xy(name)) for name in KP_NAMES}

    neck = P["neck"]
    hip_mid = (P["hip_l"] + P["hip_r"]) / 2
    sho_mid = (P["shoulder_l"] + P["shoulder_r"]) / 2

    trunk = neck - hip_mid                       # hip -> neck, image frame
    trunk_len = np.linalg.norm(trunk, axis=1)
    torso_len_px = float(np.nanmedian(trunk_len)) or 1.0
    trunk_u = trunk / np.maximum(trunk_len, 1e-6)[:, None]

    # ---- torso lean (in-image): + = dancer tilts toward image right.
    # Channels stay in the raw image frame; facing correction happens in
    # build_crab_commands via BODY_SIGN_* so the plotted curves always show
    # what the human actually did.
    lean = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))
    torso_roll = lean

    # ---- pitch proxy: trunk shortening + nose dipping below shoulder line.
    shrink = np.clip(trunk_len / torso_len_px, 0.0, 1.0)
    bend_mag = np.degrees(np.arccos(shrink))                    # 0..90
    nose_dip = (P["nose"][:, 1] - sho_mid[:, 1]) / torso_len_px
    nose_dip = moving_average(nose_dip, 5)
    pitch_sign = np.sign(nose_dip)
    torso_pitch = bend_mag * np.clip(np.abs(nose_dip) * 2.0, 0.3, 1.0) * pitch_sign

    # ---- yaw proxy: torso-width foreshortening (HIP-DOMINANT, user:
    # shoulders follow the arms and mislead; hips ride the pelvis), nose
    # offset (direction). Soft-signed with tanh and noise-floored: arccos
    # explodes near front-on, so small widths are damped, but a real spin
    # tracks fully.
    sho_w = np.linalg.norm(P["shoulder_r"] - P["shoulder_l"], axis=1)
    hip_w = np.linalg.norm(P["hip_r"] - P["hip_l"], axis=1)
    sho_med = float(np.nanmedian(sho_w)) or 1.0
    hip_med = float(np.nanmedian(hip_w)) or 1.0
    width = 0.25 * moving_average(sho_w, 5) / sho_med \
        + 0.75 * moving_average(hip_w, 5) / hip_med
    floor = 0.35  # shoulders never fully vanish front-on
    cos_y = (np.clip(width, floor, 1.0) - floor) / (1.0 - floor)
    yaw_mag = np.degrees(np.arccos(cos_y))
    yaw_mag = yaw_mag * _smoothstep01((yaw_mag - 10.0) / 25.0)
    nose_side = moving_average((P["nose"][:, 0] - sho_mid[:, 0]) / torso_len_px, 5)
    torso_yaw = yaw_mag * np.tanh(nose_side * 4.0)

    # ---- facing (front/back) from keypoint LATERALITY + full signed yaw.
    # The width-based magnitude cannot tell front-center from back-center;
    # the anatomically-labelled LEFT/RIGHT ordering can: facing the camera,
    # the dancer's LEFT shoulder appears image-RIGHT (mirrored), turned
    # away it appears image-LEFT. sign(lat) = sign(cos psi), so lat splits
    # the yaw circle into the facing hemisphere (|psi| < 90, yaw = mag) and
    # the back hemisphere (yaw = 180 - mag). Within the facing hemisphere
    # the nose picks the sign; in the back hemisphere the nose is hidden
    # and the sign rides continuity with the previous frame.
    lat_px = 0.25 * (P["shoulder_l"][:, 0] - P["shoulder_r"][:, 0]) \
        + 0.75 * (P["hip_l"][:, 0] - P["hip_r"][:, 0])
    lat = moving_average(lat_px, BODY_SMOOTH_WIN) / max(sho_med, 1e-6)
    facing = np.zeros(n, dtype=bool)
    state = True
    for i in range(n):
        v = lat[i]
        if np.isfinite(v) and abs(v) > FACING_FLIP_LAT:
            state = v > 0
        facing[i] = state
    psi_abs = np.where(facing, yaw_mag, 180.0 - yaw_mag)
    yaw_full = np.zeros(n)
    prev = 0.0
    for i in range(n):
        pa = float(psi_abs[i])
        if facing[i] and abs(nose_side[i]) > 0.04:
            val = pa if nose_side[i] > 0.0 else -pa
        else:
            val = pa if abs(pa - prev) <= abs(-pa - prev) else -pa
        yaw_full[i] = val
        prev = val
    # continuous (unwrapped) so downstream smoothing and nearest-angle
    # tracking never see a +-180 seam.
    torso_yaw_full = moving_average(
        np.degrees(np.unwrap(np.radians(yaw_full))), BODY_SMOOTH_WIN)

    # ---- arms
    def _unit_rows(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-6)[:, None]

    def arm(side: str) -> dict[str, np.ndarray]:
        s, e, w = P[f"shoulder_{side}"], P[f"elbow_{side}"], P[f"wrist_{side}"]
        upper = e - s
        upper_len = np.linalg.norm(upper, axis=1)
        full = float(np.nanmedian(upper_len)) or 1.0
        upper_u = upper / np.maximum(upper_len, 1e-6)[:, None]
        # Trunk direction is 2D per frame; cosine with upper-arm per row.
        cos_trunk = np.sum(upper_u * trunk_u, axis=1)
        elev = 180.0 - np.degrees(np.arccos(np.clip(cos_trunk, -1.0, 1.0)))
        # Shoulder decomposition in the trunk frame: "down" is -trunk_u,
        # "right" its perpendicular. The projected upper-arm then is
        #   a = sin(roll) * cos(pitch),  d = cos(roll) * cos(pitch)
        # with the depth component sin(pitch) invisible except as
        # foreshortening |upper| / full = cos(pitch).
        perp = np.stack([trunk_u[:, 1], -trunk_u[:, 0]], axis=1)  # trunk right
        a = np.sum(upper_u * perp, axis=1)
        d = -np.sum(upper_u * trunk_u, axis=1)
        roll = np.degrees(np.arctan2(a, d))
        foreshorten = moving_average(
            np.clip(upper_len / full, 0.0, 1.0), 5)
        pitch = np.degrees(np.arccos(foreshorten))
        bend = np.degrees(np.arccos(np.clip(np.sum(
            _unit_rows(s - e) * _unit_rows(w - e), axis=1), -1.0, 1.0)))
        # Forearm (elbow -> wrist) elevation in the trunk frame: 0 hanging
        # along the body, 180 overhead. Drives claw J1 directly (user
        # semantics: J1 = forearm direction).
        fore = _unit_rows(w - e)
        fore_elev = 180.0 - np.degrees(np.arccos(np.clip(
            np.sum(fore * trunk_u, axis=1), -1.0, 1.0)))
        hand_up = (hip_mid[:, 1] - w[:, 1]) / torso_len_px
        side_off = (w[:, 0] - sho_mid[:, 0]) / torso_len_px
        return {"elev": elev, "pitch": pitch, "roll": roll, "bend": bend,
                "fore": fore_elev, "up": hand_up, "side": side_off}

    arm_l = arm("l")
    arm_r = arm("r")

    # ---- legs / scale (with LOWER-BODY VALIDITY)
    # Only frames where the hips/knees/ankles are confidently detected AND
    # the limb geometry is consistent may drive the standing height or the
    # jump/raise channels. Upper-body-only shots otherwise collapse the
    # crouch estimate to its clip floor (the "lowest standing height" bug).
    def leg_len(side: str) -> np.ndarray:
        hip, knee, ank = P[f"hip_{side}"], P[f"knee_{side}"], P[f"ankle_{side}"]
        return (np.linalg.norm(knee - hip, axis=1)
                + np.linalg.norm(ank - knee, axis=1))

    def conf_of(name: str) -> np.ndarray:
        return np.nan_to_num(np.asarray(
            kps.conf.get(name, np.ones(n)), dtype=float), nan=0.0)

    def raw_ok_of(name: str) -> np.ndarray:
        """Per-FRAME presence (the pipeline conf is per-keypoint only)."""
        a = np.asarray(kps.xy(name), dtype=float)
        return np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])

    leg_names = ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r")
    leg_conf = np.min(np.stack([conf_of(x) for x in leg_names]), axis=0)
    raw_ok = np.ones(n, dtype=bool)
    for name in leg_names:
        raw_ok &= raw_ok_of(name)
    ll = (leg_len("l") + leg_len("r")) / 2.0
    conf_ok = leg_conf > LEG_CONF_MIN
    ref_ll = ll[conf_ok] if conf_ok.any() else ll
    med_ll = float(np.nanmedian(ref_ll)) or 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lr_frame = ll / max(med_ll, 1e-6)
    geom_ok = (np.isfinite(lr_frame)
               & (lr_frame > LEG_LEN_RATIO_MIN)
               & (lr_frame < LEG_LEN_RATIO_MAX))
    # Hip->ankle span sanity: a close-up where the legs are out of frame
    # hallucinates the ankles right next to the hips (span ~0) -> invalid.
    raw_hip_ank = 0.5 * (
        np.linalg.norm(np.asarray(kps.xy("ankle_l")) - np.asarray(kps.xy("hip_l")), axis=1)
        + np.linalg.norm(np.asarray(kps.xy("ankle_r")) - np.asarray(kps.xy("hip_r")), axis=1))
    ref_ha = raw_hip_ank[raw_ok & conf_ok]
    med_ha = float(np.nanmedian(ref_ha)) if len(ref_ha) else med_ll
    with np.errstate(invalid="ignore", divide="ignore"):
        ha_ok = (raw_hip_ank / max(med_ha, 1e-6)) > 0.35
    # Close-up framing: when the hips sit in the bottom fifth of the whole
    # image the legs are cut off below the frame and the pose model
    # hallucinates ankles near the hips -> the standing height collapses.
    all_y = np.concatenate([np.asarray(kps.xy(k), dtype=float)[:, 1]
                            for k in KP_NAMES])
    y_ref = (float(np.nanpercentile(all_y, 98))
             if np.isfinite(all_y).any() else 1.0)
    hip_y_raw = 0.5 * (np.asarray(kps.xy("hip_l"), dtype=float)[:, 1]
                       + np.asarray(kps.xy("hip_r"), dtype=float)[:, 1])
    hips_low = hip_y_raw > 0.82 * y_ref
    # Medium close-up: the space below the hips is shorter than a plausible
    # leg (~0.7 torso), i.e. the legs are cut off by the frame bottom.
    y_bottom = float(np.nanmax(all_y))
    neck_y_raw = np.asarray(kps.xy("neck"), dtype=float)[:, 1]
    torso_px_raw = np.abs(hip_y_raw - neck_y_raw)
    legs_cut = (y_bottom - hip_y_raw) < 0.95 * torso_px_raw
    leg_valid = (raw_ok & conf_ok & geom_ok
                 & np.nan_to_num(ha_ok, nan=False).astype(bool)
                 & ~np.nan_to_num(hips_low, nan=False).astype(bool)
                 & ~np.nan_to_num(legs_cut, nan=False).astype(bool))
    leg_px = float(np.nanmedian(ll[leg_valid])) if leg_valid.any() else med_ll
    leg_px = max(leg_px, 1.0)
    leg_ratio = float(leg_px / torso_len_px) if torso_len_px > 0 else LEG_RATIO_FALLBACK
    if not np.isfinite(leg_ratio) or not (1.2 <= leg_ratio <= 3.2):
        leg_ratio = LEG_RATIO_FALLBACK

    hip_y = (P["hip_l"][:, 1] + P["hip_r"][:, 1]) / 2
    ank_yl = P["ankle_l"][:, 1]
    ank_yr = P["ankle_r"][:, 1]
    ank_y = (ank_yl + ank_yr) / 2
    # Image y points down: standing legs put the ankle well below the hip.
    crouch_raw = np.clip((ank_y - hip_y) / leg_px, 0.25, 1.0)
    # A value pinned at the clip floor means the ankles sit at/above the
    # hips -- never a usable crouch estimate. Leave it invalid so the
    # normal-standing fallback applies.
    crouch_raw = np.where(crouch_raw <= 0.26, np.nan, crouch_raw)
    crouch_raw = np.where(leg_valid, crouch_raw, np.nan)
    if np.isfinite(crouch_raw).sum() < 2:
        crouch = np.full(n, LEG_FALLBACK_CROUCH)
    else:
        crouch = moving_average(crouch_raw, CROUCH_SMOOTH_WIN)
        crouch = _fill_gaps(crouch, gap_frames)
        # Stretches without usable legs default to NORMAL standing height
        # instead of holding/interpolating a (possibly crouched) value from
        # a poorly tracked neighbour. Only tiny 1-2 frame blips interpolate.
        idx = np.arange(n)
        valid_idx = idx[np.isfinite(crouch_raw)]
        nearest = np.min(np.abs(idx[:, None] - valid_idx[None, :]), axis=1)
        crouch = np.where(nearest > 2, LEG_FALLBACK_CROUCH, crouch)
        crouch = moving_average(crouch, CROUCH_SMOOTH_WIN)

    # ---- jumps & leg raises: rises against the standing baseline (median
    # over near-straight-legged frames), so a dance spent mostly crouching
    # does not read as one long jump. Invalid-leg frames never jump/raise.
    standing = leg_valid & (crouch >= 0.8)
    if standing.mean() >= 0.15:
        base_sl = standing
    elif leg_valid.any():
        base_sl = leg_valid
    else:
        base_sl = slice(None)
    hip_base = float(np.nanmedian(hip_y[base_sl]))
    ank_base_l = float(np.nanmedian(ank_yl[base_sl]))
    ank_base_r = float(np.nanmedian(ank_yr[base_sl]))

    # Jump: hips AND both ankles clearly above their baselines (image y
    # down, so "higher" = smaller y = positive rise). Leg raise: one ankle
    # high while the hips stay put; suppressed while airborne.
    hip_rise = (hip_base - hip_y) / torso_len_px
    rise_l = np.where(leg_valid, (ank_base_l - ank_yl) / leg_px, 0.0)
    rise_r = np.where(leg_valid, (ank_base_r - ank_yr) / leg_px, 0.0)
    air = np.where(leg_valid,
                   np.minimum(hip_rise, np.minimum(rise_l, rise_r)), 0.0)
    jump = _smoothstep01((air - JUMP_RISE_ON) / (JUMP_RISE_FULL - JUMP_RISE_ON))
    jump = moving_average(_fill_gaps(jump, gap_frames), JUMP_SMOOTH_WIN)
    grounded = 1.0 - jump
    raise_l = _smoothstep01((rise_l - RAISE_ON) / (RAISE_FULL - RAISE_ON)) * grounded
    raise_r = _smoothstep01((rise_r - RAISE_ON) / (RAISE_FULL - RAISE_ON)) * grounded
    raise_l = moving_average(_fill_gaps(raise_l, gap_frames), RAISE_SMOOTH_WIN)
    raise_r = moving_average(_fill_gaps(raise_r, gap_frames), RAISE_SMOOTH_WIN)

    # quality: fraction of confident keypoints
    conf = np.zeros(n)
    total = 0
    for name in KP_NAMES:
        c = np.asarray(kps.conf.get(name, np.ones(n)), dtype=float)
        conf += np.nan_to_num(c, nan=0.0)
        total += 1
    quality = conf / max(total, 1)

    return HumanChannels(
        times=t,
        torso_pitch=_fill_gaps(torso_pitch, gap_frames),
        torso_roll=_fill_gaps(torso_roll, gap_frames),
        torso_yaw=_fill_gaps(torso_yaw, gap_frames),
        torso_facing=facing,
        torso_yaw_full=torso_yaw_full,
        lean_raw=lean,
        elevation_l=_fill_gaps(arm_l["elev"], gap_frames),
        elevation_r=_fill_gaps(arm_r["elev"], gap_frames),
        shoulder_pitch_l=_fill_gaps(arm_l["pitch"], gap_frames),
        shoulder_pitch_r=_fill_gaps(arm_r["pitch"], gap_frames),
        shoulder_roll_l=_fill_gaps(arm_l["roll"], gap_frames),
        shoulder_roll_r=_fill_gaps(arm_r["roll"], gap_frames),
        bend_l=_fill_gaps(arm_l["bend"], gap_frames),
        bend_r=_fill_gaps(arm_r["bend"], gap_frames),
        forearm_l=_fill_gaps(arm_l["fore"], gap_frames),
        forearm_r=_fill_gaps(arm_r["fore"], gap_frames),
        hand_up_l=_fill_gaps(arm_l["up"], gap_frames),
        hand_up_r=_fill_gaps(arm_r["up"], gap_frames),
        side_l=_fill_gaps(arm_l["side"], gap_frames),
        side_r=_fill_gaps(arm_r["side"], gap_frames),
        crouch=crouch,
        leg_ratio=leg_ratio,
        torso_len_px=torso_len_px,
        quality=quality,
        jump=jump,
        raise_l=raise_l,
        raise_r=raise_r,
    )


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def wrap180(deg: float) -> float:
    """Wrap degrees to [-180, 180) -- nearest-angle convention."""
    return (deg + 180.0) % 360.0 - 180.0


# Per-turn-pill states (tri-state cycle on the timeline strip).
SPIN_MODE_OFF = 0     # follow the plain yaw curve
SPIN_MODE_FULL = 1    # scripted 360 deg sweep
SPIN_MODE_HALF = 2    # scripted 180 deg sweep (dancer ends back to camera)


def claw_joints_from_channels(leg: str, forearm, bend, side, up,
                              signs: dict) -> dict:
    """Human arm channels -> raw claw joint commands (rad, unclipped).

    THE shared mapping: playback tables (arrays) and the manual ghost-arm
    probe mode (scalars) both go through it, so dragging a ghost arm moves
    the crab claw exactly the way the video does.
      forearm  deg   0 hanging .. 180 overhead  -> J1
      bend     deg   0 straight .. 180 folded   -> J2+J3 (fold-negated)
      side     torso units, + = image right     -> J0 (+ static bias)
      up       torso units above mid-hip        -> J4 open
    """
    fore01 = np.clip(np.asarray(forearm, dtype=float) / 180.0, 0.0, 1.0)
    bend01 = np.clip(np.asarray(bend, dtype=float) / 150.0, 0.0, 1.0)
    side01 = np.clip(np.asarray(side, dtype=float) * 1.2, -1.0, 1.0)

    # User-defined joint semantics (round-9):
    #   J0 shoulder yaw, J1 forearm direction, J2+J3 the elbow angle
    #   shared between them, J4 hand open. J0/J3 carry the static biases
    #   (HOME reads as claws yawed 45 deg CCW and J3 fully folded).
    j0 = (HOME[f"{leg}_J0_joint"]
          - signs[leg]["ccw"] * np.radians(CLAW_J0_OFFSET_DEG)
          + signs[leg]["lat"] * CLAW_SHOULDER_YAW_GAIN * side01)
    # J1: uniform static bias (NOT sign-flipped per side -- measured with a
    # same-delta sweep on both claws, see CLAW_J1_OFFSET_DEG).
    j1 = (HOME[f"{leg}_J1_joint"]
          - np.radians(CLAW_J1_OFFSET_DEG)
          + signs[leg]["lift"] * CLAW_FOREARM_GAIN * (fore01 - 0.12))
    j2 = (HOME[f"{leg}_J2_joint"]
          - signs[leg]["fold"] * CLAW_ELBOW_J2_GAIN * (bend01 - 0.2))
    j3 = (HOME[f"{leg}_J3_joint"]
          - signs[leg]["fold"] * (CLAW_ELBOW_J3_GAIN * (bend01 - 0.2)
                                  + np.radians(CLAW_J3_OFFSET_DEG)))
    open01 = np.clip((np.asarray(up, dtype=float) - 0.9) / 0.5, 0.0, 1.0)
    j4 = HOME[f"{leg}_J4_joint"] + signs[leg]["open"] * (0.3 + 0.9 * open01)
    return {f"{leg}_J0_joint": j0, f"{leg}_J1_joint": j1,
            f"{leg}_J2_joint": j2, f"{leg}_J3_joint": j3,
            f"{leg}_J4_joint": j4}


def clip_claw_joints(cmds: dict, joint_range: dict) -> dict:
    """Clamp raw claw commands into each joint's usable range."""
    return {name: np.clip(value, *joint_range[name])
            for name, value in cmds.items()}


def claw_ik_targets(pitch: float, roll: float, bend: float,
                    anchor: np.ndarray,
                    reach: tuple[float, float],
                    side: str = "l",
                    extra_rot_rad: float = 0.0) -> tuple[np.ndarray, ...]:
    """Ghost-arm channels -> (u1 dir, u2 dir, soft palm, outward swing).

    The human arm enters as SEGMENT DIRECTIONS (scale-invariant -- the
    ghost figure is ~3x the crab, directions carry over untouched) plus a
    palm position target for the soft stage: direction u_dir scaled by the
    relative extent rho * R_palm about the crab's own shoulder, floored
    above the deck ("arms hanging" would otherwise sit 118 mm under the
    floor: shoulder 108 mm, claw 226 mm). The fourth value is the OUTWARD
    swing (rad, + = away from the body): -side_sign * roll, the shoulder
    swing the J2/J3 split-prior consumes.

    Everything is then rotated by CLAW_IK_FRAME_ROT_DEG about the vertical:
    the ghost's swing plane sits outside the claw's elbow cone, and this
    maps the ghost's toward-camera axis onto the crab's forward so the
    choreography lands inside the cone (see the constant's note).
    """
    elbow, wrist = ghost_arm_points(pitch, roll, bend, anchor, GHOST_ARM_LEN,
                                    _GHOST_D_H, _GHOST_D_F, _GHOST_ZH)
    u1 = elbow - anchor
    u1 = u1 / max(float(np.linalg.norm(u1)), 1e-9)
    u2 = wrist - elbow
    u2 = u2 / max(float(np.linalg.norm(u2)), 1e-9)
    v = wrist - anchor
    n = max(float(np.linalg.norm(v)), 1e-9)
    rho = min(max(n / (2.0 * GHOST_ARM_LEN), CLAW_IK_RHO_MIN), CLAW_IK_RHO_MAX)
    palm_off = (v / n) * rho * reach[0]
    swing = (-1.0 if side == "l" else 1.0) * math.radians(roll)

    rot = math.radians(CLAW_IK_FRAME_ROT_DEG) + extra_rot_rad
    if rot != 0.0:
        c, s = math.cos(rot), math.sin(rot)
        rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        u1 = rot_z @ u1
        u2 = rot_z @ u2
        palm_off = rot_z @ palm_off
    palm = anchor + palm_off
    palm = palm.copy()
    palm[2] = max(palm[2], CLAW_IK_FLOOR_Z)
    return u1, u2, palm, swing


def _claw_solver_ctx(crab: CrabModel) -> dict:
    """Per-claw IK bookkeeping shared by the series and pose solvers.

    Cached on the CrabModel: the outward-sign probing perturbs qpos and
    resets data, so building it mid-frame would clobber the live pose (the
    manual mode crab sank into the floor and its ghost vanished until this
    was cached).
    """
    cached = getattr(crab, "_claw_ik_ctx", None)
    if cached is not None:
        return cached
    m = crab.model
    names = {leg: [f"{leg}_J{i}_joint" for i in range(4)] for leg in CLAWS}
    geom = crab.claw_geometry
    # J2 fold box around HOME (CLAW_IK_J2_LIMIT_DEG; 0 freezes J2).
    j2_lim = (math.radians(CLAW_IK_J2_LIMIT_DEG)
              if CLAW_IK_J2_LIMIT_DEG > 0.0 else None)

    def j2_box(n: str) -> tuple[float, float]:
        lo, hi = crab.joint_range[n]
        if j2_lim is None:
            return HOME[n], HOME[n]
        return max(lo, HOME[n] - j2_lim), min(hi, HOME[n] + j2_lim)

    # Outward signs for the J2/J3 swing prior: which way does +rotation
    # move the palm HORIZONTALLY relative to the body's outward axis
    # (+y for LF, -y for RF)? Probed once, never trusted from CAD. The
    # probing perturbs data, so the incoming pose is snapshotted and put
    # back afterwards.
    d = crab.data
    qpos_saved = d.qpos.copy()
    mocap_pos_saved = d.mocap_pos.copy()
    mocap_quat_saved = d.mocap_quat.copy()
    swing_sign: dict[str, tuple[float, float]] = {}
    for leg in CLAWS:
        mujoco.mj_resetData(m, d)
        crab.set_home_joints()
        crab.set_body(crab.z_stand, 0.0, 0.0, 0.0)
        mujoco.mj_forward(m, d)
        out_axis = np.array([0.0, 1.0 if leg == "LF" else -1.0, 0.0])
        palm0 = np.asarray(d.site(leg).xpos).copy()
        signs = []
        for j in (2, 3):
            adr = crab.joint_adr[f"{leg}_J{j}_joint"]
            save = d.qpos[adr]
            d.qpos[adr] = save + 0.25
            mujoco.mj_forward(m, d)
            dp = np.asarray(d.site(leg).xpos) - palm0
            signs.append(float(np.sign(dp @ out_axis) or 1.0))
            d.qpos[adr] = save
        swing_sign[leg] = (signs[0], signs[1])
    ctx = {
        "names": names,
        "adr": {leg: [crab.joint_adr[n] for n in names[leg]] for leg in CLAWS},
        "dof": {leg: [int(m.joint(n).dofadr[0]) for n in names[leg]]
                for leg in CLAWS},
        "site": {leg: m.site(leg).id for leg in CLAWS},
        "fore": {leg: m.body(f"{leg}_forearm_link").id for leg in CLAWS},
        "tip": {leg: m.body(f"{leg}_finger_tip_link").id for leg in CLAWS},
        "swing_sign": swing_sign,
        "lo": {leg: np.array([
            j2_box(n)[0] if j == 2 else crab.joint_range[n][0]
            for j, n in enumerate(names[leg])]) for leg in CLAWS},
        "hi": {leg: np.array([
            j2_box(n)[1] if j == 2 else crab.joint_range[n][1]
            for j, n in enumerate(names[leg])]) for leg in CLAWS},
        "q_ref": {leg: np.array([HOME[n] for n in names[leg]]) for leg in CLAWS},
        "L": geom["seg_len"],
        "jacp": np.zeros((3, m.nv)),
        "jacr": np.zeros((3, m.nv)),
        "eye4": np.eye(4),
    }
    # User anchor correspondence -> constant joint-space calibration. This
    # resets mjData (and probes for the rest-pose signs), so the caller's
    # pose is snapshotted at the top and restored below.
    ctx["anchor_offset"] = _anchor_offset(crab, ctx)

    d.qpos[:] = qpos_saved
    for i in range(len(mocap_pos_saved)):
        d.mocap_pos[i] = mocap_pos_saved[i]
        d.mocap_quat[i] = mocap_quat_saved[i]
    mujoco.mj_forward(m, d)

    crab._claw_ik_ctx = ctx
    return ctx


def _claw_dls(crab: CrabModel, ctx: dict, leg: str, targets: tuple,
              seed: np.ndarray, lo_w: np.ndarray, hi_w: np.ndarray,
              iters_a: int, iters_b: int) -> tuple[np.ndarray, float, float]:
    """Staged damped-least-squares IK for one claw (directions first).

    Stage A -- arm geometry (HIGH weight): match the upper-arm direction
    ("elbow direction") and the forearm direction as unit vectors. With
    J0..J3 that nearly determines the pose, locking the elbow
    configuration before anything else gets a say. Jacobian of a unit
    segment u = (pB - pA)/L wrt the joints is (J_B - J_A)/L; the shoulder
    end is upstream of every claw joint, so J_A = 0 there.

    Stage B -- hand, soft (LOW weight + leash): the palm position target
    joins at CLAW_IK_W_HAND while q may only travel CLAW_IK_HAND_LEASH rad
    away from the stage-A solution. When the claw cannot reproduce the
    hand pose exactly, the residual is absorbed AT THE HAND -- the arm
    keeps the human's segment directions; the error never propagates back
    into the elbow.

    Trust-region clipped per iteration, projected into [lo_w, hi_w] (joint
    box, possibly narrowed by the series velocity cap). Jacobian IK only
    needs kinematics + COM frames (mink does the same). Returns
    (q, direction error deg, palm miss m).
    """
    m, d = crab.model, crab.data
    jacp, jacr, eye4 = ctx["jacp"], ctx["jacr"], ctx["eye4"]
    adr, dof = ctx["adr"][leg], ctx["dof"][leg]
    q_ref = ctx["q_ref"][leg]
    u1_t, u2_t, palm_t, swing = targets
    L1, L2 = ctx["L"][leg]
    site_id, fore_id = ctx["site"][leg], ctx["fore"][leg]
    s2, s3 = ctx["swing_sign"][leg]

    q = np.clip(np.array(seed, dtype=float), lo_w, hi_w)

    def forward() -> None:
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def direction_err() -> tuple[float, float]:
        """Errors of the HARD objectives: elbow AZIMUTH + bend angle."""
        p_fore = np.asarray(d.body(f"{leg}_forearm_link").xpos)
        p_palm = np.asarray(d.site(leg).xpos)
        u1 = p_fore - np.asarray(d.body(f"{leg}_shoulder_link").xpos)
        u1 = u1 / max(float(np.linalg.norm(u1)), 1e-9)
        u2 = p_palm - p_fore
        u2 = u2 / max(float(np.linalg.norm(u2)), 1e-9)
        clip = lambda v: min(1.0, max(-1.0, float(v)))
        az = lambda v: v[:2] / max(float(np.linalg.norm(v[:2])), 1e-9)
        err1 = math.degrees(math.acos(clip(az(u1) @ az(u1_t))))
        bend_t = math.degrees(math.acos(clip(u1_t @ u2_t)))
        bend_l = math.degrees(math.acos(clip(u1 @ u2)))
        return err1, abs(bend_t - bend_l)

    def step(rows: list, res: list, lo: np.ndarray, hi: np.ndarray) -> None:
        J = np.vstack(rows)
        e = np.concatenate(res)
        dq = np.linalg.lstsq(J, e, rcond=None)[0]
        dq = np.clip(dq, -CLAW_IK_MAX_STEP, CLAW_IK_MAX_STEP)
        q[:] = np.clip(q + dq, lo, hi)

    def build_rows(include_hand: bool) -> tuple[list, list]:
        """Stacked objectives against the CURRENT qpos. NB: jacp is a
        shared buffer -- each jacobian's dof columns must be copied out
        (or vstacked) before the next mj_jac* call overwrites it.

        u1 (elbow direction) enters through its AZIMUTH rows only: the
        claw shoulder geometry pins u1's elevation to ~-30 deg no matter
        the joints (azimuth spins freely), so a full-vector u1 residual
        would fight the solver forever with a structural 29 deg error.
        HARD: u1 azimuth + the elbow BEND angle (the radial component of
        the u2 error along u1; cos(bend) lives on u1.u2, jacobian row
        u1' dU2 to first order). SOFT: full forearm pointing u2 (elevation
        AND off-plane -- the claw elbow plane cannot always do it) and in
        stage B the palm position."""
        P_h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        mujoco.mj_jacBody(m, d, jacp, jacr, fore_id)
        J_fore = jacp[:, dof].copy()
        p_fore = np.asarray(d.body(f"{leg}_forearm_link").xpos)
        u1 = p_fore - np.asarray(d.body(f"{leg}_shoulder_link").xpos)
        u1 = u1 / max(float(np.linalg.norm(u1)), 1e-9)
        rows = [CLAW_IK_W_UARM * (P_h @ J_fore) / L1]
        res = [CLAW_IK_W_UARM * (P_h @ (u1_t - u1))]
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        p_palm = np.asarray(d.site(leg).xpos)
        u2 = p_palm - p_fore
        u2 = u2 / max(float(np.linalg.norm(u2)), 1e-9)
        J_u2 = (jacp[:, dof] - J_fore) / L2
        e2 = u2_t - u2
        # bend angle (hard)
        rows.append(np.array([CLAW_IK_W_BEND * (u1 @ J_u2)]))
        res.append(np.array([CLAW_IK_W_BEND * float(e2 @ u1)]))
        # full forearm pointing (soft, but the next-highest tier)
        rows.append(CLAW_IK_W_FORE_TAN * J_u2)
        res.append(CLAW_IK_W_FORE_TAN * e2)
        if include_hand:
            rows.append(CLAW_IK_W_HAND * jacp[:, dof])
            res.append(CLAW_IK_W_HAND * (palm_t - p_palm))
        # Shoulder-swing split prior (user): the outward swing drives J2
        # and J3 TOGETHER, each at CLAW_IK_SWING_SPLIT along its probed
        # outward sign -- so the fold joints carry the swing instead of
        # J0/J1 saturating (the frozen-J2 experiment).
        e2 = np.zeros(4)
        e2[2] = 1.0
        e3 = np.zeros(4)
        e3[3] = 1.0
        rows.append(CLAW_IK_W_SWING * e2)
        res.append(np.array([CLAW_IK_W_SWING * (
            q_ref[2] + s2 * CLAW_IK_SWING_SPLIT * swing - q[2])]))
        rows.append(CLAW_IK_W_SWING * e3)
        res.append(np.array([CLAW_IK_W_SWING * (
            q_ref[3] + s3 * CLAW_IK_SWING_SPLIT * swing - q[3])]))
        rows.append(CLAW_IK_W_POSTURE * 0.3 * eye4)
        res.append(CLAW_IK_W_POSTURE * 0.3 * (q_ref - q))
        return rows, res

    # ---- stage A: segment directions + posture, no hand objective ----
    for _ in range(iters_a):
        forward()
        rows, res = build_rows(include_hand=False)
        step(rows, res, lo_w, hi_w)

    # ---- stage B: the soft hand joins, on a leash around stage A ----
    q_a = q.copy()
    lo_b = np.maximum(lo_w, q_a - CLAW_IK_HAND_LEASH)
    hi_b = np.minimum(hi_w, q_a + CLAW_IK_HAND_LEASH)
    for _ in range(iters_b):
        forward()
        rows, res = build_rows(include_hand=True)
        step(rows, res, lo_b, hi_b)

    # ---- stage C: floor keep-out (one-sided) ----
    # Folds the claw up only while the finger tip / palm are actually below
    # the deck; above the floor this stage does nothing, so the mapped pose
    # stays exact. The tip mesh hangs up to ~22 mm below its body origin,
    # hence CLAW_FLOOR_TIP_Z. The posture row anchors to the stage-B pose
    # so the correction is the smallest that clears the ground; the stage
    # may travel the whole joint window (clearing the floor from a hanging
    # pose needs a large elbow change).
    q_b = q.copy()
    tip_id = ctx["tip"][leg]

    def floor_violation() -> float:
        return max(0.0,
                   CLAW_FLOOR_TIP_Z - float(np.asarray(d.xpos[tip_id])[2]),
                   CLAW_FLOOR_PALM_Z - float(np.asarray(d.site(leg).xpos)[2]))

    for _ in range(CLAW_IK_FLOOR_ITERS):
        forward()
        rows, res = [], []
        violated = False
        mujoco.mj_jacBody(m, d, jacp, jacr, tip_id)
        J_tip = jacp[:, dof].copy()
        v = CLAW_FLOOR_TIP_Z - float(np.asarray(d.xpos[tip_id])[2])
        if v > 0.0:
            rows.append(CLAW_IK_W_FLOOR * J_tip[2:3])
            res.append(np.array([CLAW_IK_W_FLOOR * v]))
            violated = True
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        J_palm = jacp[:, dof].copy()
        v = CLAW_FLOOR_PALM_Z - float(np.asarray(d.site(leg).xpos)[2])
        if v > 0.0:
            rows.append(CLAW_IK_W_FLOOR * J_palm[2:3])
            res.append(np.array([CLAW_IK_W_FLOOR * v]))
            violated = True
        if not violated:
            break
        rows.append(CLAW_IK_W_POSTURE * 0.5 * eye4)
        res.append(CLAW_IK_W_POSTURE * 0.5 * (q_b - q))
        step(rows, res, lo_w, hi_w)

    forward()
    e1, e2 = direction_err()
    miss = float(np.linalg.norm(palm_t - np.asarray(d.site(leg).xpos)))
    return q, max(e1, e2), miss, floor_violation()


def _claw_pair_contacts(crab: CrabModel) -> int:
    """Number of LF-vs-RF mesh contacts in the current crab pose."""
    m, d = crab.model, crab.data
    n = 0
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = int(np.asarray(m.geom_bodyid[c.geom1]).ravel()[0])
        b2 = int(np.asarray(m.geom_bodyid[c.geom2]).ravel()[0])
        n1 = m.body(b1).name
        n2 = m.body(b2).name
        if (n1[:2] in ("LF", "RF") and n2[:2] in ("LF", "RF")
                and n1[:2] != n2[:2]):
            n += 1
    return n


def _dilate(x: np.ndarray, window: int) -> np.ndarray:
    k = max(0, window // 2)
    return np.array([x[max(0, i - k):min(len(x), i + k + 1)].max()
                     for i in range(len(x))])


def _resolve_claw_collisions(claw: dict, crab: CrabModel,
                             signs: dict) -> None:
    """Push the two claws apart only where their meshes actually collide.

    The hard no-cross window alone leaves a tiny J0 travel. Here the video
    table keeps its full mapping wherever it is clear; contact frames get a
    smoothed outward J0 offset, so the user sees a large, natural J0 range
    and the left/right claws never pass through each other.
    """
    from app.constants import CLAW_J0_CROSS_LIMIT_DEG  # local: avoid cycles
    names = list(claw)
    n = len(np.asarray(claw[names[0]]))
    if n == 0:
        return
    j0 = {leg: np.asarray(claw[f"{leg}_J0_joint"], dtype=float)
          for leg in ("LF", "RF")}
    for _round in range(2):
        need = np.zeros(n, dtype=float)
        for i in range(n):
            crab.set_home_joints()
            crab.set_body(crab.z_stand)
            for name in names:
                crab.data.qpos[crab.joint_adr[name]] = float(claw[name][i])
            mujoco.mj_forward(crab.model, crab.data)
            if _claw_pair_contacts(crab) == 0:
                continue
            for step in range(5, 125, 5):
                for leg in ("LF", "RF"):
                    lo, hi = CLAW_J0_LIMIT_DEG[leg]
                    q = j0[leg][i] + signs[leg]["lat"] * math.radians(step)
                    crab.data.qpos[crab.joint_adr[f"{leg}_J0_joint"]] =                         np.clip(q, math.radians(lo), math.radians(hi))
                mujoco.mj_forward(crab.model, crab.data)
                if _claw_pair_contacts(crab) == 0:
                    need[i] = step
                    break
        if not need.any():
            break
        for leg in ("LF", "RF"):
            off = _dilate(need, 9)
            off = moving_average(off, 9)
            off = np.maximum(off, need)
            q = j0[leg] + signs[leg]["lat"] * np.radians(off)
            lo, hi = CLAW_J0_LIMIT_DEG[leg]
            claw[f"{leg}_J0_joint"] = np.clip(
                q, np.radians(lo), np.radians(hi))
            j0[leg] = np.asarray(claw[f"{leg}_J0_joint"], dtype=float)
    crab.set_home_joints()
    crab.set_body(crab.z_stand)
    mujoco.mj_forward(crab.model, crab.data)


def _sol_key(sol: tuple) -> tuple:
    """Candidate ordering: a floor-safe pose ALWAYS beats a digging one,
    then the smallest floor violation, then direction error / palm miss.
    Without this the seed selection -- which only looked at direction
    error -- happily picked a pose that put the claw through the deck."""
    viol = sol[3]
    unsafe = viol > 1e-6
    return (unsafe, viol if unsafe else 0.0, sol[1], sol[2])


def _floor_fix(crab: CrabModel, ctx: dict, q_lf: np.ndarray) -> np.ndarray:
    """Post-calibration floor keep-out on the LF kinematics.

    q_lf is the LF-space joint vector AFTER the user-anchor offset (the
    offset can lower the claw into the deck even when the raw IK pose was
    clear). One-sided DLS folds the claw up only while the finger tip /
    palm are actually below the deck, anchored to the incoming pose so the
    correction is minimal. Right-claw callers apply their own mirror to
    the result (reflection about a vertical plane preserves height).
    """
    m, d = crab.model, crab.data
    jacp, jacr, eye4 = ctx["jacp"], ctx["jacr"], ctx["eye4"]
    adr, dof = ctx["adr"]["LF"], ctx["dof"]["LF"]
    site_id, tip_id = ctx["site"]["LF"], ctx["tip"]["LF"]
    lo, hi = ctx["lo"]["LF"], ctx["hi"]["LF"]
    q0 = np.array(q_lf, dtype=float)

    def forward(q) -> None:
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def violation() -> float:
        return max(0.0,
                   CLAW_FLOOR_TIP_Z - float(np.asarray(d.xpos[tip_id])[2]),
                   CLAW_FLOOR_PALM_Z - float(np.asarray(d.site("LF").xpos)[2]))

    def attempt(seed: np.ndarray) -> tuple[np.ndarray, float]:
        q = np.clip(np.array(seed, dtype=float), lo, hi)
        for _ in range(CLAW_IK_FLOOR_ITERS):
            forward(q)
            if violation() <= 1e-6:
                return q, 0.0
            rows, res = [], []
            v = CLAW_FLOOR_TIP_Z - float(np.asarray(d.xpos[tip_id])[2])
            mujoco.mj_jacBody(m, d, jacp, jacr, tip_id)
            J_tip = jacp[:, dof].copy()
            if v > 0.0:
                rows.append(CLAW_IK_W_FLOOR * J_tip[2:3])
                res.append(np.array([CLAW_IK_W_FLOOR * v]))
            v = CLAW_FLOOR_PALM_Z - float(np.asarray(d.site("LF").xpos)[2])
            mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
            J_palm = jacp[:, dof].copy()
            if v > 0.0:
                rows.append(CLAW_IK_W_FLOOR * J_palm[2:3])
                res.append(np.array([CLAW_IK_W_FLOOR * v]))
            rows.append(CLAW_IK_W_POSTURE * 0.5 * eye4)
            res.append(CLAW_IK_W_POSTURE * 0.5 * (q0 - q))
            J = np.vstack(rows)
            e = np.concatenate(res)
            dq = np.linalg.lstsq(J, e, rcond=None)[0]
            dq = np.clip(dq, -CLAW_IK_MAX_STEP, CLAW_IK_MAX_STEP)
            q[:] = np.clip(q + dq, lo, hi)
        forward(q)
        return q, violation()

    best: tuple[np.ndarray, float] | None = None
    for seed in (q0, ctx["q_ref"]["LF"]):
        q, viol = attempt(seed)
        if viol <= 1e-6:
            return q
        if best is None or viol < best[1]:
            best = q, viol
    return best[0]


def _anchor_offset(crab: CrabModel, ctx: dict) -> np.ndarray:
    """Constant joint-space calibration from the user's anchor correspondence.

    Solves the direction IK for the ghost pose CLAW_ANCHOR_GHOST in the
    canonical frame (body at yaw 0, standing height) and returns
    anchor_pose - ik_pose, where anchor_pose is the JOINT-SWEEP pose
    rest + CLAW_ANCHOR_DELTA_DEG (the sweep tool's deltas are relative to
    the rest pose). Added to every IK result, it makes the user's
    relationship exact -- "ghost at that angle -> crab at this angle" --
    without turning the mapping into a direct map: the offset is a
    constant, so the ghost still drives the claw direction everywhere.
    J4 is not part of the mapping (rest value), so its anchor value is
    not used. Resets mjData -- callers must save/restore their pose.
    """
    m, d = crab.model, crab.data
    geom = crab.claw_geometry
    # Calibrate at the manual ghost's display height (CrabView uses
    # 1.02 x z_stand) so the sliders reproduce the anchor exactly; the
    # video path solves at its own base_z and is unaffected by the few-mm
    # difference.
    z_cal = crab.z_stand * 1.02
    mujoco.mj_resetData(m, d)
    crab.set_home_joints()
    d.qpos[crab.free_adr + 2] = z_cal
    d.qpos[crab.free_adr + 3:crab.free_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    anchor = geom["anchor"]["LF"] + np.array([0.0, 0.0, z_cal])
    # Calibrate at the roll the manual sliders can actually command: the
    # user's anchor roll (90, old convention) clamps into the new
    # negative-outward slider range, i.e. 0 at pitch 90 (where the arm
    # direction is roll-independent anyway; only the elbow split differs).
    pitch_a, roll_a, bend_a = CLAW_ANCHOR_GHOST
    roll_a = float(np.clip(roll_a, -CLAW_SWING_LIMIT_DEG, 0.0))
    targets = claw_ik_targets(pitch_a, roll_a, bend_a, anchor,
                              geom["reach"]["LF"], side="l")
    best: tuple | None = None
    for j0 in np.radians(np.arange(-180.0, 91.0, 30.0)):
        seed = np.array([j0,
                         ctx["q_ref"]["LF"][1],
                         ctx["q_ref"]["LF"][2],
                         ctx["q_ref"]["LF"][3]])
        # Match the manual-mode iteration budget (10+6): calibrating on a
        # less-converged solve leaves a few degrees of drift between the
        # anchor and what the sliders actually produce.
        sol = _claw_dls(crab, ctx, "LF", targets, seed,
                        ctx["lo"]["LF"], ctx["hi"]["LF"], 10, 6)
        if best is None or _sol_key(sol) < _sol_key(best):
            best = sol
    signs = crab.calibrate_claw_signs()
    rest = np.array([
        HOME["LF_J0_joint"] + math.radians(CLAW_J0_SPREAD_DEG),
        HOME["LF_J1_joint"] - math.radians(CLAW_J1_OFFSET_DEG),
        HOME["LF_J2_joint"],
        HOME["LF_J3_joint"] - signs["LF"]["fold"] * math.radians(
            CLAW_J3_OFFSET_DEG),
    ])
    return rest + np.radians(CLAW_ANCHOR_DELTA_DEG) - best[0]


def solve_claw_ik_pose(crab: CrabModel, angles: dict,
                       iters_a: int = 10, iters_b: int = 6
                       ) -> dict[str, dict[str, float]]:
    """Single-pose IK from ghost-arm slider angles (manual ghost mode).

    angles: {"l": (pitch, roll, bend), "r": (...)} in degrees, the same
    convention the ghost overlay renders. The body must ALREADY be posed
    (pose_dance with any claw values) -- the target anchor is the live
    shoulder position, so the claw mirrors the ghost's segment directions
    in world space, whatever base height/yaw is on screen. Two starts (the
    live pose and HOME) guard against local minima. Returns
    {leg: {J0..J3 joint name: rad}} -- J4 stays with the caller.
    """
    ctx = _claw_solver_ctx(crab)
    geom = crab.claw_geometry
    d = crab.data
    out: dict[str, dict[str, float]] = {}
    mirror = np.array([-1.0, 1.0, -1.0, -1.0])
    off = ctx["anchor_offset"]
    # Claw joints are BODY-relative: the manual body is displayed at the
    # front-yaw offset (CRAB_FRONT_YAW_OFFSET), while the video solve parks
    # it at yaw 0. Passing +body_yaw cancels FRAME_ROT for the displayed
    # pose, so the same slider angles produce the SAME joints as the video
    # table (and as the joint-sweep tool's rest-relative pose).
    quat = d.qpos[crab.free_adr + 3:crab.free_adr + 7]
    rot9 = np.zeros(9)
    mujoco.mju_quat2Mat(rot9, quat)
    rot9 = rot9.reshape(3, 3)
    body_yaw = math.atan2(float(rot9[1, 0]), float(rot9[0, 0]))
    # Snapshot the live pose ONCE: both arms start from the same seeds, so
    # equal slider angles produce exactly mirrored joints. (Seeding the
    # right solve from the just-solved left pose let a weakly-constrained
    # null space -- the hanging pose's azimuth -- drift tens of degrees.)
    cur = np.array([d.qpos[a] for a in ctx["adr"]["LF"]])
    for side, leg in (("l", "LF"), ("r", "RF")):
        pitch, roll, bend = angles[side]
        # Same side-swing limit as the sliders (negative = outward).
        roll = float(np.clip(roll, -CLAW_SWING_LIMIT_DEG, 0.0))
        # The manual sliders use the USER'S convention: NEGATIVE roll is
        # outward on BOTH arms, so the right arm is solved on the LEFT
        # kinematics with the slider value AS-IS and its joints are
        # mirrored back (the video path negates roll_r instead -- its
        # channels use the image-space convention where a symmetric raise
        # is +r / -r).
        anchor_l = np.asarray(
            d.body("LF_shoulder_link").xpos, dtype=float).copy()
        targets = claw_ik_targets(pitch, roll, bend, anchor_l,
                                  geom["reach"]["LF"], side="l",
                                  extra_rot_rad=body_yaw)
        best: tuple | None = None
        for seed in (cur, ctx["q_ref"]["LF"]):
            sol = _claw_dls(crab, ctx, "LF", targets, seed,
                            ctx["lo"]["LF"], ctx["hi"]["LF"], iters_a, iters_b)
            if best is None or _sol_key(sol) < _sol_key(best):
                best = sol
        # Anchor calibration, THEN the floor keep-out (the offset can put
        # the claw under the deck even when the raw IK pose was clear).
        q_sol = _floor_fix(crab, ctx, best[0] + off)
        q = q_sol if leg == "LF" else q_sol * mirror
        out[leg] = {ctx["names"][leg][j]: float(q[j]) for j in range(4)}
    return out


def solve_claw_ik_series(ch: HumanChannels, crab: CrabModel,
                         base_z: np.ndarray,
                         analytic: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-frame staged IK for the claw positioning joints J0..J3.

    Each frame runs the two-stage solve of _claw_dls: stage A matches the
    human ARM-SEGMENT DIRECTIONS (upper arm / elbow + forearm -- the elbow
    configuration follows from the pair) at the highest weight; stage B
    adds the palm position as a damped soft objective leashed to the
    stage-A pose, so an unreachable hand gives up WITHOUT pushing its
    error back into the elbow. Frames warm-start from the previous
    solution with a velocity cap, so the series is smooth by construction.

    Solved in the yaw-0 frame: claw joints are body-relative and the ghost
    basis co-rotates with the commanded yaw (CrabView._ghost_segments), so
    the solution holds at any body yaw -- scripted turns included, without
    re-solving. Body roll/pitch (<=25 deg) are ignored, as in the analytic
    mapping. Returns J0..J3 arrays per claw; J4 stays with `analytic`.
    """
    m, d = crab.model, crab.data
    geom = crab.claw_geometry
    ctx = _claw_solver_ctx(crab)
    names, lo, hi = ctx["names"], ctx["lo"], ctx["hi"]
    q_ref = ctx["q_ref"]

    n = len(ch.times)
    out = {name: np.empty(n) for leg in CLAWS for name in names[leg]}
    q_prev: dict[str, np.ndarray | None] = {leg: None for leg in CLAWS}

    # Base parked at the origin, identity attitude; only its height follows
    # base_z so the palm floor clamp tracks the crouch.
    mujoco.mj_resetData(m, d)
    crab.set_home_joints()
    d.qpos[crab.free_adr + 3:crab.free_adr + 7] = (1.0, 0.0, 0.0, 0.0)

    side_ch = {
        "LF": (ch.shoulder_pitch_l, ch.shoulder_roll_l, ch.bend_l),
        # The RIGHT claw is solved on the LEFT kinematics against the
        # GHOST-SPACE mirror of its channels (roll negated): the ghost
        # figure's own symmetry plane is span(zh, d_f), under which the
        # right arm at roll r maps exactly onto the left arm at roll -r.
        # (Mirroring the WORLD vectors is wrong here -- it maps "arm
        # pointing at the camera" to a different world direction.) The
        # joint result then mirrors back (URDF boxes are exact mirrors).
        "RF": (ch.shoulder_pitch_r, -ch.shoulder_roll_r, ch.bend_r),
    }
    mirror = np.array([-1.0, 1.0, -1.0, -1.0])   # J0,J1,J2,J3
    for i in range(n):
        z = float(base_z[i])
        d.qpos[crab.free_adr + 2] = z
        anchor = geom["anchor"]["LF"] + np.array([0.0, 0.0, z])
        for leg in CLAWS:
            pitch_a, roll_a, bend_a = side_ch[leg]
            # Side-swing limit (user): the LF-space roll is outward-positive,
            # so clamp to the same 0..120 arc as the manual ghost sliders --
            # no swinging across the body.
            roll_i = float(np.clip(np.nan_to_num(roll_a[i], nan=0.0),
                                   0.0, CLAW_SWING_LIMIT_DEG))
            # Video bend is 0 straight / 180 folded; ghost_arm_points takes
            # the manual-slider convention 180 straight / 0 folded.
            targets = claw_ik_targets(
                float(np.nan_to_num(pitch_a[i], nan=0.0)),
                roll_i,
                float(180.0 - np.nan_to_num(bend_a[i], nan=0.0)),
                anchor, geom["reach"]["LF"], side="l")

            an_i = np.array([analytic[f"LF_J{j}_joint"][i] for j in range(4)])
            if q_prev[leg] is None:
                # First frame: fresh, unconstrained; try HOME and the
                # analytic pose and keep the better one.
                lo_w, hi_w = lo["LF"], hi["LF"]
                seeds = (q_ref["LF"], an_i)
            else:
                lo_w = np.maximum(lo["LF"], q_prev[leg] - CLAW_IK_VSTEP)
                hi_w = np.minimum(hi["LF"], q_prev[leg] + CLAW_IK_VSTEP)
                seeds = (q_prev[leg],)
            best: tuple | None = None
            for s in seeds:
                sol = _claw_dls(crab, ctx, "LF", targets, s, lo_w, hi_w,
                                CLAW_IK_ITERS_A, CLAW_IK_ITERS_B)
                if best is None or _sol_key(sol) < _sol_key(best):
                    best = sol
            if best[1] > CLAW_IK_RETRY_DIR_DEG or best[3] > 1e-6:
                # Stuck (a folded warm start sits in a local minimum the
                # DLS cannot leave, and the velocity cap pins it there).
                # Re-seed from HOME / the analytic pose inside a 3x-widened
                # window -- it catches up over a few frames instead of
                # teleporting. Still large afterwards = genuinely
                # unreachable (across the body midline, dead overhead).
                if q_prev[leg] is None:
                    lo3, hi3 = lo["LF"], hi["LF"]
                else:
                    lo3 = np.maximum(lo["LF"], q_prev[leg] - 3 * CLAW_IK_VSTEP)
                    hi3 = np.minimum(hi["LF"], q_prev[leg] + 3 * CLAW_IK_VSTEP)
                for s in (q_ref["LF"], an_i):
                    sol = _claw_dls(crab, ctx, "LF", targets, s, lo3, hi3,
                                    CLAW_IK_ITERS_A, CLAW_IK_ITERS_B)
                    if _sol_key(sol) < _sol_key(best):
                        best = sol
            q_sol = _floor_fix(crab, ctx, best[0] + ctx["anchor_offset"])
            q = q_sol if leg == "LF" else q_sol * mirror
            q_prev[leg] = best[0].copy()
            for j, name in enumerate(names[leg]):
                out[name][i] = q[j]

    mujoco.mj_resetData(m, d)
    return out


def ghost_arm_points(pitch_deg: float, roll_deg: float, bend_deg: float,
                     shoulder: np.ndarray, L: float,
                     d_h: np.ndarray, d_f: np.ndarray, zh: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Forward ghost-arm reconstruction: shoulder pitch/roll + elbow flexion
    -> (elbow, wrist) world points (same math the ghost overlay renders)."""
    ref = d_f + 0.35 * zh
    ref = ref / np.linalg.norm(ref)
    # Pitch is a sagittal swing: -90 straight back, 0 hanging, +90 toward
    # the camera, 180 overhead. Video channels only ever land in 0..90
    # (arccos of foreshortening); manual posing uses the whole span.
    p = np.radians(np.clip(pitch_deg, -90.0, 180.0))
    r = np.radians(np.clip(roll_deg, -150.0, 150.0))
    u1 = (np.sin(p) * d_f
          + np.cos(p) * (-np.sin(r) * d_h - np.cos(r) * zh))
    u1 = u1 / np.linalg.norm(u1)
    elbow = shoulder + L * u1
    ang = np.pi - np.radians(bend_deg)
    axis = np.cross(u1, ref)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else d_h
    u2 = u1 * np.cos(ang) + np.cross(axis, u1) * np.sin(ang)
    u2 = u2 / np.linalg.norm(u2)
    return elbow, elbow + L * u2


def arm_channels_from_points(shoulder: np.ndarray, elbow: np.ndarray,
                             wrist: np.ndarray, centre: np.ndarray,
                             d_h: np.ndarray, zh: np.ndarray,
                             torso: float) -> dict:
    """Inverse of ghost_arm_points for the channels the claw mapping eats.

    Only forearm / bend / side / up are recovered (the claws do not consume
    shoulder pitch/roll). d_h is the ghost's screen-left axis, which for the
    mirrored figure corresponds to image-RIGHT in the source video -- hence
    the negated d_h term in `side`. `up` is measured from ~0.5 torso below
    the shoulder bar, matching the human channel's mid-hip reference.
    """
    u1 = elbow - shoulder
    u1 = u1 / max(np.linalg.norm(u1), 1e-9)
    fore = wrist - elbow
    fore = fore / max(np.linalg.norm(fore), 1e-9)
    to_sho = shoulder - elbow
    to_sho = to_sho / max(np.linalg.norm(to_sho), 1e-9)
    forearm = 180.0 - np.degrees(np.arccos(np.clip(fore @ zh, -1.0, 1.0)))
    bend = np.degrees(np.arccos(np.clip(to_sho @ fore, -1.0, 1.0)))
    rel = wrist - centre
    side = float(-(rel @ d_h) / torso)
    up = float((rel @ zh + 0.5 * torso) / torso)
    return {"forearm": float(forearm), "bend": float(bend),
            "side": side, "up": up}


def build_crab_commands(ch: HumanChannels, crab: CrabModel,
                        claw_ik: bool = False) -> CrabCommandTable:
    """Map human channels to crab body + claw commands (per sample time).

    claw_ik=True replaces the analytic J0..J3 channel mapping with the
    GMR-style per-frame IK solve (see solve_claw_ik_series); J4 keeps the
    open/close channel either way. The analytic table is always computed
    too -- it seeds the IK and supplies J4.
    """
    signs = crab.calibrate_claw_signs()
    z_stand = crab.z_stand

    roll = np.clip(BODY_SIGN_ROLL * K_ROLL * ch.torso_roll, -CMD_ROLL_MAX, CMD_ROLL_MAX)
    pitch = np.clip(BODY_SIGN_PITCH * K_PITCH * ch.torso_pitch, -CMD_PITCH_MAX, CMD_PITCH_MAX)
    # Yaw source: the FULL signed estimate (front/back aware via keypoint
    # laterality) when the channel exists; a slow turn or a held back-facing
    # pose then tracks directly -- no scripted sweep needed. The legacy
    # magnitude-only estimate is the fallback.
    yaw_src = getattr(ch, "torso_yaw_full", None)
    yaw_src = (np.nan_to_num(yaw_src, nan=0.0) if yaw_src is not None
               else ch.torso_yaw)
    yaw = np.clip(BODY_SIGN_YAW * K_YAW * yaw_src, -CMD_YAW_MAX, CMD_YAW_MAX)
    roll = moving_average(roll, BODY_SMOOTH_WIN)
    pitch = moving_average(pitch, BODY_SMOOTH_WIN)
    yaw = moving_average(yaw, BODY_SMOOTH_WIN)
    # Body-yaw slew limit (user): the crab turns no faster than this, so a
    # jittery estimate cannot whip the body around.
    if len(ch.times) > 1:
        yaw_step = BODY_YAW_MAX_SPEED_DEG * float(
            np.median(np.diff(ch.times)))
        for i in range(1, len(yaw)):
            yaw[i] = min(max(yaw[i], yaw[i - 1] - yaw_step),
                         yaw[i - 1] + yaw_step)

    # Leg length sets the standing scale; crouch modulates around it.
    lr = np.clip(ch.leg_ratio, LEG_RATIO_LO, LEG_RATIO_HI)
    leg_scale = LEG_SCALE_LO + (LEG_SCALE_HI - LEG_SCALE_LO) * (
        (lr - LEG_RATIO_LO) / (LEG_RATIO_HI - LEG_RATIO_LO))
    s = _smoothstep01((ch.crouch - CROUCH_LO) / (CROUCH_HI - CROUCH_LO))
    z_factor = Z_MIN_FACTOR + (Z_MAX_FACTOR - Z_MIN_FACTOR) * s
    base_z = np.clip(z_stand * z_factor * leg_scale, 0.5 * z_stand, 1.13 * z_stand)

    # A little life: sway legs against the body roll/yaw.
    leg_sway = moving_average(0.08 * np.radians(roll) + 0.05 * np.radians(yaw), BODY_SMOOTH_WIN)

    # Spins: with a truthful signed yaw, a scripted sweep is only needed
    # where the dancer rotates FASTER than the estimate can count -- a
    # sustained |dyaw/dt| over SPIN_RATE_DEG becomes a pill scripted at the
    # measured number of whole turns, rate-capped, ADDED onto the live
    # curve. A slow turn or a held back-facing pose is not a spin. During a
    # fast spin the estimate itself jitters (nose hidden, laterality
    # crossings), so the rate fires in bursts: bursts closer than
    # SPIN_MERGE_FRAMES merge into one window judged by its NET rotation --
    # oscillation nets under 180 deg and is dropped.
    if len(ch.times) > 1:
        dt = float(np.median(np.diff(ch.times)))
    else:
        dt = 1.0 / 30.0
    raw = np.nan_to_num(BODY_SIGN_YAW * yaw_src, nan=0.0)
    rate = moving_average(np.abs(np.gradient(raw) / dt), 9)
    over = rate > SPIN_RATE_DEG
    bursts: list[tuple[int, int]] = []
    start = None
    for i in range(len(rate) + 1):
        on = i < len(rate) and over[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= SPIN_MIN_FRAMES:
                bursts.append((start, i))
            start = None
    segs: list[tuple[int, int, float, float]] = []   # (s0, s1, dir, deg)
    if bursts:
        merged: list[list[int]] = [list(bursts[0])]
        for s0, s1 in bursts[1:]:
            if s0 - merged[-1][1] <= SPIN_MERGE_FRAMES:
                merged[-1][1] = s1
            else:
                merged.append([s0, s1])
        for s0, s1 in merged:
            delta = float(raw[s1 - 1] - raw[s0])
            # A cluster netting under half a turn is usually oscillation
            # (the plain full-yaw curve already renders it) -- but it may
            # also be a whole turn the estimator LOST while spinning. Every
            # cluster stays as a CANDIDATE pill at one 360; they default to
            # OFF and the user flips on the ones the crab visibly missed.
            turns = max(1, int(round(abs(delta) / 360.0)))
            segs.append((s0, s1, float(np.sign(delta) or 1.0), 360.0 * turns))

    spin_segs = [
        {"i0": s0, "i1": s1, "dir": direction, "deg": degrees,
         "mode": SPIN_MODE_FULL}
        for s0, s1, direction, degrees in segs
    ]
    spin_mask, yaw_spin = _scripted_yaw(yaw, dt, spin_segs)

    claw: dict[str, np.ndarray] = {}
    for human_side, leg in (("l", "LF"), ("r", "RF")):
        raw = claw_joints_from_channels(
            leg,
            getattr(ch, f"forearm_{human_side}"),
            getattr(ch, f"bend_{human_side}"),
            getattr(ch, f"side_{human_side}"),
            getattr(ch, f"hand_up_{human_side}"),
            signs)
        for jname, arr in raw.items():
            smoothed = moving_average(_fill_gaps(arr, 15), CLAW_SMOOTH_WIN)
            lo, hi = crab.joint_range[jname]
            claw[jname] = np.clip(smoothed, lo, hi)

    if claw_ik:
        # Direction-tracking IK (staged u1/u2 solve), calibrated by the
        # user's anchor correspondence through ctx["anchor_offset"]. The
        # analytic table above seeded the solve and supplies J4.
        ik = solve_claw_ik_series(ch, crab, base_z, claw)
        for jname, arr in ik.items():
            smoothed = moving_average(_fill_gaps(arr, 15), CLAW_SMOOTH_WIN)
            lo, hi = crab.joint_range[jname]
            claw[jname] = np.clip(smoothed, lo, hi)

    # Anti-crossing spread (user): LF J0 +, RF J0 - so the claws stop
    # clipping through each other. Same constant the manual/idle poses use.
    for leg, sgn in (("LF", 1.0), ("RF", -1.0)):
        name = f"{leg}_J0_joint"
        claw[name] = np.clip(
            claw[name] + sgn * math.radians(CLAW_J0_SPREAD_DEG),
            *crab.joint_range[name])
    # Usable shoulder-yaw limits (user): keep both arms clear of the leg
    # links -- the opening frames used to park LF J0 at +40 (inside LM).
    for leg in CLAWS:
        lo, hi = CLAW_J0_LIMIT_DEG[leg]
        name = f"{leg}_J0_joint"
        claw[name] = np.clip(claw[name], np.radians(lo), np.radians(hi))
    # Claw forearm J1 never below -110 deg (user).
    for leg in CLAWS:
        name = f"{leg}_J1_joint"
        claw[name] = np.clip(claw[name], np.radians(CLAW_J1_MIN_DEG),
                             crab.joint_range[name][1])

    # The gripper (J4) is OUT of the channel mapping: rest by default. A
    # recorded keyframe may drive it explicitly.
    for leg in CLAWS:
        claw[f"{leg}_J4_joint"] = np.full(len(ch.times), HOME[f"{leg}_J4_joint"])

    # Hard no-cross window: the two claws must stay on their own side of
    # the midline. LF inward is negative, RF inward positive. Applied BEFORE
    # calibration so a user keyframe can move J0 OUTWARD from the mapping.
    cross = math.radians(CLAW_J0_CROSS_LIMIT_DEG)
    claw["LF_J0_joint"] = np.maximum(claw["LF_J0_joint"], -cross)
    claw["RF_J0_joint"] = np.minimum(claw["RF_J0_joint"], cross)

    # User-recorded keyframes: piecewise-linear corrections on top of the
    # current mapping. Re-clamp the no-cross window afterwards so a keyframe
    # cannot put the two claws through each other.
    apply_calibration(claw, ch, crab)
    claw["LF_J0_joint"] = np.maximum(claw["LF_J0_joint"], -cross)
    claw["RF_J0_joint"] = np.minimum(claw["RF_J0_joint"], cross)

    # Joint speed limit (user: 31.4 rad/s) as a final safety net on every
    # claw command -- the IK's own per-frame cap is far tighter, this also
    # bounds the analytic channel and any re-seed jump.
    if len(ch.times) > 1:
        step = JOINT_SPEED_MAX_RAD * float(np.median(np.diff(ch.times)))
        for jname, arr in claw.items():
            limited = arr.copy()
            for i in range(1, len(limited)):
                lo, hi = limited[i - 1] - step, limited[i - 1] + step
                limited[i] = min(max(limited[i], lo), hi)
            claw[jname] = limited

    # J0 was twitching in the first seconds where the pose estimate is
    # noisy. Low-pass it over ~0.2 s before the collision pass; this keeps
    # the smooth dance motion and removes frame-to-frame J0 spikes.
    if len(ch.times) > 1:
        dt = float(np.median(np.diff(ch.times)))
        win = max(3, int(round(0.20 / max(dt, 1e-3))) | 1)
        for leg in ("LF", "RF"):
            name = f"{leg}_J0_joint"
            claw[name] = moving_average(claw[name], win)

    # Last line of defence: only frames whose claw meshes touch are nudged
    # outward. This runs after the speed limit so the resolved pose is what
    # the preview and the exported Jumper clip actually use.
    _resolve_claw_collisions(claw, crab, signs)

    # Leg raises lift the same-side walking legs (rear partner follows at a
    # fraction); jump stays a scalar the view adds on top of base_z.
    raise_l = _fill_gaps(ch.raise_l, 15)
    raise_r = _fill_gaps(ch.raise_r, 15)
    foot_lift = {
        "LM": LEG_RAISE_GAIN * z_stand * raise_l,
        "LR": LEG_RAISE_GAIN * LEG_RAISE_REAR * z_stand * raise_l,
        "RM": LEG_RAISE_GAIN * z_stand * raise_r,
        "RR": LEG_RAISE_GAIN * LEG_RAISE_REAR * z_stand * raise_r,
    }

    # Jump segments, same treatment as turns: a run of jump01 over the
    # threshold becomes a toggleable pill; when disabled the hop is muted.
    # Inside a segment the hop is SCRIPTED (user request): a short crouch
    # (发力), then the airborne rise, then settling -- not just a raw
    # proportional lift.
    jump01 = _fill_gaps(ch.jump, 15)
    jump_mask = np.zeros(len(jump01), dtype=bool)
    hop = np.zeros(len(jump01))
    over_j = jump01 > JUMP_SEG_ON
    start = None
    for i in range(len(jump01) + 1):
        on = i < len(jump01) and over_j[i]
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= JUMP_SEG_MIN_FRAMES:
                jump_mask[start:i] = True
                m = i - start
                crouch_end = start + max(int(round(m * 0.3)), 2)
                air_end = min(start + max(int(round(m * 0.7)), crouch_end + 2), i)
                c = crouch_end - start
                a = air_end - crouch_end
                d = i - air_end
                for k in range(start, crouch_end):
                    p = _smoothstep01((k - start) / max(c - 1, 1))
                    hop[k] = -JUMP_CROUCH_DEPTH * p
                for k in range(crouch_end, air_end):
                    p = _smoothstep01((k - crouch_end) / max(a - 1, 1))
                    hop[k] = -JUMP_CROUCH_DEPTH * (1.0 - p) + 1.0 * p
                for k in range(air_end, i):
                    p = _smoothstep01((k - air_end) / max(d - 1, 1))
                    hop[k] = 1.0 * (1.0 - p)
            start = None

    return CrabCommandTable(
        times=ch.times, base_z=base_z, roll=roll, pitch=pitch, yaw=yaw,
        leg_sway=leg_sway, claw=claw,
        jump=hop, raise_l=raise_l, raise_r=raise_r,
        foot_lift=foot_lift,
        jump_mask=jump_mask,
        spin_mask=spin_mask, yaw_spin=yaw_spin,
        spin_segs=spin_segs,
    )


def _scripted_yaw(yaw: np.ndarray, dt: float,
                  segs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the scripted-turn sweeps onto (spin_mask, yaw_spin).

    Every sweep is ADDED onto the live yaw curve and rate-capped; a short
    detection window finishes past the segment end. Accumulated turns are
    HELD to clip end (see the tail note inside). segs carry their mode and
    their measured amplitude: anything not SPIN_MODE_OFF is active, at the
    segment's own deg (whole multiples of 360).
    """
    yaw_spin = yaw.astype(float).copy()
    spin_mask = np.zeros(len(yaw), dtype=bool)
    step_max = max(SPIN_MAX_SPEED_DEG * dt / 360.0, 1e-3)
    offset = 0.0       # accumulated whole turns (deg, signed) -- HELD
    i_hold = 0         # first frame not yet written
    n = len(yaw)
    for seg in sorted(segs, key=lambda s: s["i0"]):
        mode = seg.get("mode", SPIN_MODE_FULL)
        if mode == SPIN_MODE_OFF:
            continue
        s0, s1 = seg["i0"], seg["i1"]
        direction = seg["dir"]
        degrees = float(seg.get("deg", 360.0))
        # Between the previous sweep and this one, HOLD the accumulated
        # turn (see the tail note below).
        if offset != 0.0 and i_hold < s0:
            sl = slice(i_hold, s0)
            yaw_spin[sl] = yaw[sl] + offset
            spin_mask[sl] = True
        prog, i = 0.0, s0
        while i < n:
            left = s1 - i
            # Prefer finishing exactly at the segment end (linear sweep),
            # but never faster than the rate cap.
            step = step_max if left <= 0 else min((1.0 - prog) / left, step_max)
            prog = min(1.0, prog + step)
            yaw_spin[i] = yaw[i] + offset + direction * degrees * prog
            spin_mask[i] = True
            i += 1
            if prog >= 1.0:
                break
        offset += direction * degrees
        i_hold = i
    # HOLD the turn: the 2D estimator cannot represent "facing away"
    # (it relaxes back toward 0), so dropping the accumulated offset would
    # visibly spin the crab back. The offset persists until another
    # scripted turn changes it; the mask covers the held tail so the pills
    # stay consistent with what is displayed.
    if offset != 0.0 and i_hold < n:
        tail = slice(i_hold, n)
        yaw_spin[tail] = yaw[tail] + offset
        spin_mask[tail] = True
    return spin_mask, yaw_spin


def respin_table(table: "CrabCommandTable", modes: list[int]) -> "CrabCommandTable":
    """Re-evaluate the scripted turns from per-segment pill states.

    modes: one state per detected segment (order = table.spin_segs);
    SPIN_MODE_OFF mutes the segment, anything else sweeps it at its own
    measured amplitude. Mutates spin_mask/yaw_spin in place and returns
    the table, so toggling a pill re-times every later sweep without
    re-running the analysis.
    """
    if not table.spin_segs:
        return table
    segs = [dict(seg, mode=int(mode))
            for seg, mode in zip(table.spin_segs, modes)]
    times = table.times
    dt = (float(np.median(np.diff(times))) if len(times) > 1
          else 1.0 / 30.0)
    table.spin_mask, table.yaw_spin = _scripted_yaw(table.yaw, dt, segs)
    return table


def sample_commands(table: CrabCommandTable, t: float, spin=False,
                    jump=True) -> dict:
    """Linear interpolation of a command table at an arbitrary time.

    spin selects the yaw source inside suspected-turn segments: True uses
    all of them, an ndarray is a per-frame enable mask, False/None follows
    the plain yaw curve. jump gates the hop the same way: True/None keeps
    every detected jump, an ndarray is a per-frame enable mask, False mutes.
    """
    t_arr = np.clip(t, table.times[0], table.times[-1])
    def interp(arr: np.ndarray) -> float:
        return float(np.interp(t_arr, table.times, arr))
    yaw_arr = table.yaw
    if spin is not False and spin is not None and table.spin_mask is not None:
        use = table.spin_mask if spin is True else np.asarray(spin, dtype=bool)
        if use.any():
            yaw_arr = np.where(use, table.yaw_spin, table.yaw)
    jump_val = interp(table.jump) if table.jump is not None else 0.0
    if jump is False or jump is None:
        jump_val = 0.0
    elif jump is not True and table.jump_mask is not None:
        use_j = np.asarray(jump, dtype=bool)
        if use_j.any():
            jump_val = float(np.interp(
                t_arr, table.times,
                np.where(use_j, table.jump, 0.0)))
        else:
            jump_val = 0.0
    return {
        "base_z": interp(table.base_z),
        "roll": interp(table.roll),
        "pitch": interp(table.pitch),
        "yaw": interp(yaw_arr),
        "leg_sway": interp(table.leg_sway),
        "claw": {name: interp(arr) for name, arr in table.claw.items()},
        "jump": jump_val,
        "raise": max(
            interp(table.raise_l) if table.raise_l is not None else 0.0,
            interp(table.raise_r) if table.raise_r is not None else 0.0,
        ),
        "foot_lift": ({name: interp(arr) for name, arr in table.foot_lift.items()}
                      if table.foot_lift else {}),
    }
