"""Keyframe calibration for the claw channel mapping.

The user poses the ghost arms and the crab claws freely in the probe tool,
records matched poses, and the app stores the *correction* between the current
mapping and the recorded claw pose. At playback each joint's correction is
interpolated piecewise-linearly against one ghost control angle:

    J0, J1  <- shoulder pitch
    J2, J3  <- shoulder side swing, outward-positive
    J4      <- elbow bend

Two recorded points therefore define a real mapping ratio, e.g. a 20 degree
claw change over a 25 degree ghost change is 80%. One point is applied as a
constant offset; outside the recorded range the nearest endpoint holds.

The file lives in the app cache, so the core mapping can load it without Qt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.constants import CACHE_DIR, CLAW_J0_LIMIT_DEG, CLAW_J1_MIN_DEG, CLAWS

CALIB_PATH = CACHE_DIR / "claw_calib.json"

#: Keyframe calibration is temporarily disabled (user request): the probe
#: tool keeps only the slider-driven mapping. Existing calibration files are
#: ignored until this is switched back on.
CALIBRATION_ENABLED = False


def load_frames() -> list[dict]:
    try:
        data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list):
        return []
    out = []
    for f in frames:
        if (isinstance(f, dict) and isinstance(f.get("ghost"), dict)
                and isinstance(f.get("crab"), dict)
                and isinstance(f.get("corr"), dict)):
            out.append(f)
    return out


def save_frames(frames: list[dict]) -> None:
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "frames": frames}, indent=2),
                   encoding="utf-8")
    tmp.replace(CALIB_PATH)


def clear() -> None:
    try:
        CALIB_PATH.unlink()
    except OSError:
        pass


def joint_control(ghost: dict, joint: str) -> float:
    """The ghost angle one claw joint is calibrated against, in degrees."""
    side = "l" if joint.startswith("LF") else "r"
    pitch, roll, bend = (float(v) for v in ghost[side])
    index = int(joint.split("_J")[1].split("_")[0])
    if index in (0, 1):
        return pitch
    if index in (2, 3):
        return -roll          # manual sliders: negative = outward
    return bend


def control_degrees(ghost: dict, joint: str) -> float:
    return joint_control(ghost, joint)


def make_record(ghost: dict, crab_joints: dict, base_joints: dict) -> dict:
    """Build a stored keyframe; ghost angles are degrees, joints radians."""
    crab = {name: float(crab_joints[name]) for name in crab_joints}
    corr = {name: float(crab_joints[name] - base_joints[name])
            for name in crab_joints}
    return {
        "ghost": {side: [float(v) for v in ghost[side]]
                  for side in ("l", "r")},
        "crab": crab,
        "corr": corr,
        "control": {name: control_degrees(ghost, name) for name in crab},
    }


#: Keyframe distance scales, degrees. Corrections fade out beyond RADIUS,
#: so a calibration only nudges the mapping near the poses the user recorded.
_SIGMA_DEG = 40.0
_RADIUS_DEG = 100.0


def _key_ghost(frame: dict) -> np.ndarray:
    g = frame["ghost"]
    lp, lr, lb = (float(v) for v in g["l"])
    rp, rr, rb = (float(v) for v in g["r"])
    # Manual slider convention: roll is negative outward; bend is 180 straight.
    return np.array([lp, -lr, lb, rp, -rr, rb], dtype=float)


def _frame_ghost(ch, n: int) -> np.ndarray:
    out = np.empty((n, 6), dtype=float)
    for side, col in (("l", 0), ("r", 3)):
        out[:, col] = np.asarray(getattr(ch, f"shoulder_pitch_{side}"),
                                 dtype=float)
        roll = np.asarray(getattr(ch, f"shoulder_roll_{side}"), dtype=float)
        out[:, col + 1] = roll if side == "l" else -roll
        # Video bend is 0 straight / 180 folded; keyframes are 180 straight.
        out[:, col + 2] = 180.0 - np.asarray(getattr(ch, f"bend_{side}"),
                                             dtype=float)
    return out


def _channel(ch, joint: str) -> np.ndarray:
    side = "l" if joint.startswith("LF") else "r"
    index = int(joint.split("_J")[1].split("_")[0])
    if index in (0, 1):
        return np.asarray(getattr(ch, f"shoulder_pitch_{side}"), dtype=float)
    if index in (2, 3):
        roll = np.asarray(getattr(ch, f"shoulder_roll_{side}"), dtype=float)
        # Video convention: outward is + for LF and - for RF.
        return roll if side == "l" else -roll
    # Video bend is 0 straight / 180 folded; keyframes use 180 straight.
    return 180.0 - np.asarray(getattr(ch, f"bend_{side}"), dtype=float)


def apply_calibration(claw: dict, ch, crab) -> None:
    """Add the recorded keyframe bias to the current vector mapping.

    Each keyframe stores the difference between the hand-posed claw angle
    and the mapping's angle at the same ghost pose. For every joint the
    biases are interpolated against its control angle (J0/J1 pitch, J2/J3
    outward swing, J4 bend) and added to the mapping. One keyframe gives a
    constant bias; two or more give piecewise segments. Outside the recorded
    control range the nearest bias is held. The result is clamped to the
    joint limits and the per-frame collision pass still runs afterwards.
    """
    if not CALIBRATION_ENABLED:
        return
    frames = load_frames()
    if not frames:
        return
    for name in list(claw):
        pairs = []
        for f in frames:
            if name in f.get("corr", {}) and name in f.get("control", {}):
                pairs.append((float(f["control"][name]),
                              float(f["corr"][name])))
        if not pairs:
            continue
        pairs.sort(key=lambda p: p[0])
        xs, ys = [], []
        for x, y in pairs:
            if xs and abs(x - xs[-1]) < 1e-9:
                ys[-1] = 0.5 * (ys[-1] + y)
            else:
                xs.append(x)
                ys.append(y)
        x = _channel(ch, name)
        if len(xs) == 1:
            bias = np.full_like(x, ys[0], dtype=float)
        else:
            bias = np.interp(x, np.asarray(xs), np.asarray(ys))
        out = np.clip(np.asarray(claw[name], dtype=float) + bias,
                      *crab.joint_range[name])
        leg = name[:2]
        index = int(name.split("_J")[1].split("_")[0])
        if index == 0:
            lo, hi = CLAW_J0_LIMIT_DEG[leg]
            out = np.clip(out, np.radians(lo), np.radians(hi))
        elif index == 1:
            out = np.clip(out, np.radians(CLAW_J1_MIN_DEG),
                          crab.joint_range[name][1])
        claw[name] = out


def ratio_text(frames: list[dict], a: int, b: int) -> str:
    """Readable per-joint ratio between two keyframes, e.g. 'LF_J0 80%'."""
    if not (0 <= a < len(frames) and 0 <= b < len(frames)):
        return ""
    fa, fb = frames[a], frames[b]
    parts = []
    for leg in CLAWS:
        for j in range(5):
            name = f"{leg}_J{j}_joint"
            dc = (fb.get("crab", {}).get(name, 0.0)
                  - fa.get("crab", {}).get(name, 0.0))
            dg = (fb.get("control", {}).get(name, 0.0)
                  - fa.get("control", {}).get(name, 0.0))
            if abs(dg) >= 1.0:
                parts.append(f"{leg} J{j} {100.0 * np.degrees(dc) / dg:.0f}%")
    return " · ".join(parts)
