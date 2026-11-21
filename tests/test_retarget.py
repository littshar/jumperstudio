"""Offline unit tests for the retargeting math (no GUI, no Sports2D).

Run:  cd crab-dance && PYTHONPATH=. python -m tests.test_retarget
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from app.core.crab_model import CrabModel
from app.core.retarget import (
    KP_NAMES, SPIN_MODE_FULL, SPIN_MODE_OFF, Keypoints,
    arm_channels_from_points, build_crab_commands, claw_joints_from_channels,
    clip_claw_joints, compute_human_channels, ghost_arm_points,
    respin_table, sample_commands, wrap180,
)
from app.constants import HOME, JOINT_RANGE, SPIN_MAX_SPEED_DEG

TORSO = 100.0  # px, toy scale


def make_kp(lean_deg: float, crouch_ratio: float,
            elev_l: float, bend_r: float, bend_l: float = 0.0) -> dict[str, np.ndarray]:
    """One synthetic frame. Image y down, torso 100 px, hip at (300, 400).

    NOTE the fixture's arm convention: forearm dir f = -cos(bend)*u +
    sin(bend)*perp, so bend=0 folds the forearm BACK (f = -u) and bend=180
    is a straight arm (f = u). The bend channel measures the angle between
    elbow->shoulder and elbow->wrist, which matches these labels.
    """
    lean = math.radians(lean_deg)
    hip = np.array([300.0, 400.0])
    neck = hip + TORSO * np.array([math.sin(lean), -math.cos(lean)])
    sho_off = np.array([0.0, -0.18 * TORSO])
    sho_l, sho_r = neck + sho_off + [-22, 0], neck + sho_off + [22, 0]

    def arm_pts(elev_deg: float, bend_deg: float, shoulder: np.ndarray):
        """Shoulder -> elbow (40 px) -> wrist (40 px).

        elev_deg: 0 = hanging along the trunk, 180 = overhead (relative to
        trunk). bend_deg: anatomical elbow flexion, 0 = straight.
        """
        elev = math.radians(elev_deg)
        bend = math.radians(bend_deg)
        down = np.array([0.0, 1.0])   # along trunk, pointing down
        perp = np.array([1.0, 0.0])
        u = math.cos(elev) * down + math.sin(elev) * perp
        elbow = shoulder + 40 * u
        f = -math.cos(bend) * u + math.sin(bend) * perp
        wrist = elbow + 40 * f
        return elbow, wrist

    elbow_l, wrist_l = arm_pts(elev_l, bend_l, sho_l)
    elbow_r, wrist_r = arm_pts(0.0, bend_r, sho_r)

    # Legs: two fixed 100 px segments. Hip-to-ankle vertical drop =
    # crouch_ratio * 200; knee sits on the perpendicular bisector.
    seg, total = 100.0, 200.0
    drop = crouch_ratio * total
    def leg_pts(sx: float):
        hip_s = hip + np.array([sx, 0.0])
        ank = hip_s + np.array([sx * 0.3, drop])
        mid = (hip_s + ank) / 2
        off = math.sqrt(max(seg * seg - (drop / 2) ** 2, 0.0))
        knee = mid + np.array([np.sign(sx) * off, 0.0])
        return hip_s, knee, ank
    hip_l, knee_l, ank_l = leg_pts(-15)
    hip_r, knee_r, ank_r = leg_pts(15)

    pts = {
        "nose": neck + [0, -25], "neck": neck,
        "shoulder_l": sho_l, "shoulder_r": sho_r,
        "elbow_l": elbow_l, "wrist_l": wrist_l,
        "elbow_r": elbow_r, "wrist_r": wrist_r,
        "hip_l": hip_l, "hip_r": hip_r,
        "knee_l": knee_l, "knee_r": knee_r,
        "ankle_l": ank_l, "ankle_r": ank_r,
    }
    return pts


def kps_from_frames(frames: list[dict]) -> Keypoints:
    n = len(frames)
    xyz = {name: np.stack([f[name] for f in frames]) for name in KP_NAMES}
    conf = {name: np.ones(n) for name in KP_NAMES}
    times = np.arange(n, dtype=float) / 30.0
    return Keypoints(xyz=xyz, conf=conf, times=times)


def test_air_gestures(crab: CrabModel) -> list[str]:
    """Jump / leg-raise channels from synthetic keypoints."""
    failures = []
    stand = [make_kp(0.0, 1.0, 0.0, 0.0) for _ in range(10)]
    # Jump: everything shifts up 70 px (0.7 torso / 0.35 leg units).
    jumpf = [{k: v - np.array([0.0, 70.0]) for k, v in f.items()} for f in stand]
    ch = compute_human_channels(kps_from_frames(stand + jumpf + stand))
    j = float(np.nanmax(ch.jump))
    if j < 0.5:
        failures.append(f"jump not detected: peak {j:.2f}")
    if max(np.nanmax(ch.raise_l), np.nanmax(ch.raise_r)) > 0.25:
        failures.append("leg raise fired during a jump (should be suppressed)")

    # Right-leg raise: right ankle up 130 px (0.65 leg units), hips stay.
    raisef = []
    for f in stand:
        g = dict(f)
        g["ankle_r"] = f["ankle_r"] - np.array([0.0, 130.0])
        g["knee_r"] = f["knee_r"] - np.array([0.0, 65.0])
        raisef.append(g)
    ch = compute_human_channels(kps_from_frames(stand + raisef + stand))
    r = float(np.nanmax(ch.raise_r))
    if r < 0.5:
        failures.append(f"right leg raise not detected: peak {r:.2f}")
    if float(np.nanmax(ch.raise_l)) > 0.25:
        failures.append("left raise fired for a right-leg raise")
    if float(np.nanmax(ch.jump)) > 0.25:
        failures.append("jump fired for a single-leg raise")

    # Command table carries the gestures through.
    table = build_crab_commands(ch, crab)
    row = sample_commands(table, ch.times[10 + len(raisef) // 2])
    if row["raise"] < 0.3:
        failures.append(f"table raise at peak: {row['raise']:.2f}")
    if row["foot_lift"]["RM"] < 0.2 * crab.z_stand:
        failures.append(f"RM lift too small: {row['foot_lift']['RM']*1000:.1f} mm")
    rr, rm = row["foot_lift"]["RR"], row["foot_lift"]["RM"]
    if rm <= 0 or not (0.3 <= rr / rm <= 0.5):
        failures.append(f"RR/RM lift ratio {rr / max(rm, 1e-9):.2f}, want ~0.4")
    return failures


def _fake_channels(yaw: np.ndarray) -> "HumanChannels":
    """Minimal channels for exercising build_crab_commands' spin logic."""
    from app.core.retarget import HumanChannels
    n = len(yaw)
    zeros = np.zeros(n)
    ones = np.ones(n) * 0.9
    return HumanChannels(
        times=np.arange(n, dtype=float) / 30.0,
        torso_pitch=zeros, torso_roll=zeros, torso_yaw=yaw, lean_raw=zeros,
        elevation_l=zeros, elevation_r=zeros,
        shoulder_pitch_l=zeros, shoulder_pitch_r=zeros,
        shoulder_roll_l=zeros, shoulder_roll_r=zeros,
        bend_l=zeros, bend_r=zeros,
        forearm_l=zeros, forearm_r=zeros,
        hand_up_l=zeros, hand_up_r=zeros,
        side_l=zeros, side_r=zeros,
        crouch=ones, leg_ratio=2.0, torso_len_px=100.0, quality=ones,
        jump=zeros, raise_l=zeros, raise_r=zeros,
    )


def test_hidden_legs(crab: CrabModel) -> list[str]:
    """Legs out of frame (upper-body-only shot) -> normal standing height."""
    failures = []
    n = 60
    half = n // 2
    leg_names = ("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r")
    frames = []
    for i in range(n):
        f = make_kp(0.0, 1.0, 0.0, 0.0)
        if i < half:
            for name in leg_names:
                f[name] = np.array([np.nan, np.nan])
        frames.append(f)
    kps = kps_from_frames(frames)
    for name in leg_names:
        c = kps.conf[name].copy()
        c[:half] = 0.0
        kps.conf[name] = c
    ch = compute_human_channels(kps)
    if np.any(ch.crouch[:half] < 0.8):
        failures.append(
            f"hidden-leg crouch {np.round(ch.crouch[:half].min(), 2)} < 0.8")
    if abs(ch.leg_ratio - 2.0) > 0.7:
        failures.append(
            f"leg_ratio {ch.leg_ratio:.2f} not ~2.0 with hidden legs")
    table = build_crab_commands(ch, crab)
    z_hidden = float(np.min(table.base_z[:half]))
    if z_hidden < 0.95 * crab.z_stand:
        failures.append(
            f"hidden-leg base_z {z_hidden*1000:.0f}mm collapsed "
            f"(stand {crab.z_stand*1000:.0f}mm)")
    if float(np.max(ch.jump[:half])) > 0.1:
        failures.append("hidden legs produced a false jump")
    return failures


def test_multi_person_trc() -> list[str]:
    """load_persons: one Keypoints per personNN.trc, ordered by number."""
    import tempfile
    from app.core.s2d_pipeline import load_persons, _ALIASES

    def write_trc(path, dx: float) -> None:
        names = ["nose", "lshoulder", "rshoulder", "lelbow", "relbow",
                 "lwrist", "rwrist", "lhip", "rhip", "lknee", "rknee",
                 "lankle", "rankle"]
        rows = []
        for f in range(5):
            t = f / 30.0
            cols = [f, t]
            for i, _ in enumerate(names):
                cols += [100.0 + dx + i * 10.0, 200.0 + i * 5.0, 0.0]
            rows.append("\t".join(str(c) for c in cols))
        # Layout mirrors Sports2D: name row repeats nothing (one cell per
        # marker), the sub-header row leads with the two empty Frame#/Time
        # cells, then X1 Y1 Z1 triples.
        name_row = "\t".join(["Frame#", "Time", *names])
        sub_row = "\t\t" + "\t".join(
            c for i in range(1, len(names) + 1)
            for c in (f"X{i}", f"Y{i}", f"Z{i}"))
        path.write_text(
            "PathFileType\t4\t(X/Y/Z)\ttest\n"
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
            "30\t30\t5\t13\tmm\n"
            + name_row + "\n" + sub_row + "\n" + "\n".join(rows) + "\n",
            encoding="utf-8")

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sub = root / "vid_Sports2D"
        sub.mkdir()
        # person01 written first: ordering must still put person00 first.
        # Spacing is > 2 torso widths so the repair pass sees three
        # distinct dancers, not duplicate tracks of one.
        write_trc(sub / "vid_px_person01.trc", dx=300.0)
        write_trc(sub / "vid_px_person00.trc", dx=0.0)
        write_trc(sub / "vid_px_person02.trc", dx=600.0)
        persons = load_persons(root)
        if len(persons) != 3:
            return [f"load_persons got {len(persons)} persons, want 3"]
        xs = [float(p.xy("nose")[0, 0]) for p in persons]
        if xs != [100.0, 400.0, 700.0]:
            failures.append(f"person order by number broken: {xs}")
    return failures


def test_spin_sweep() -> list[str]:
    """Scripted turns: exactly one circle, rate-capped, curve-continuous.

    Spins are detected on the yaw RATE (a fast ramp), not on a sustained
    magnitude -- a held back-facing pose is not a spin.
    """
    failures = []
    crab = CrabModel()
    # 0 -> 360 deg over 12 frames (900 deg/s at 30 fps) -> one scripted
    # turn; the plateaus around it must NOT fire (rate ~0). The tail is
    # long enough for the rate-capped sweep to finish inside the clip.
    ramp = np.linspace(0.0, 360.0, 12)
    yaw = np.concatenate([np.zeros(10), ramp, np.full(30, 360.0)])
    ch = _fake_channels(yaw)
    table = build_crab_commands(ch, crab)

    # A SLOW turn must NOT become a pill (rate stays under the threshold;
    # the true yaw follows it directly).
    slow = np.concatenate([np.zeros(10), np.linspace(0.0, 70.0, 30),
                           np.full(20, 70.0),
                           np.linspace(70.0, 0.0, 30), np.zeros(10)])
    t_slow = build_crab_commands(_fake_channels(slow), crab)
    if getattr(t_slow, "spin_segs", None):
        failures.append("sustained slow yaw wrongly detected as a spin")

    # Jump pills: a run of jump01 over the threshold becomes a mask; the
    # gate mutes the scripted hop (crouch -> airborne -> settle) outside
    # enabled frames.
    ch3 = _fake_channels(np.zeros(40))
    jump = np.zeros(40)
    jump[15:19] = 1.0
    ch3.jump = jump.copy()
    t3 = build_crab_commands(ch3, crab)
    if not getattr(t3, "jump_mask", np.zeros(1)).any():
        failures.append("jump segment not detected")
    else:
        seg = t3.jump[15:19]
        if not (np.nanmin(seg[:2]) < -0.1 and np.nanmax(seg) > 0.8):
            failures.append(f"hop profile lacks crouch/air: {seg.round(2)}")
    off = np.zeros(40, dtype=bool)
    quiet = sample_commands(t3, 18.0 / 30.0, jump=off)["jump"]
    on = sample_commands(t3, 18.0 / 30.0, jump=t3.jump_mask)["jump"]
    if quiet != 0.0 or on < 0.5:
        failures.append(f"jump gating: muted={quiet:.2f} on={on:.2f}")

    if not table.spin_mask.any():
        return failures + ["no spin segment detected for the fast ramp"]

    i0 = int(np.argmax(table.spin_mask))
    i1 = len(table.spin_mask) - int(np.argmax(table.spin_mask[::-1]))
    # The scripted ramp is additive on the commanded curve, so at the last
    # masked frame it must equal exactly one full turn.
    ramp = table.yaw_spin - table.yaw
    turn = float(ramp[i1 - 1])
    if abs(abs(turn) - 360.0) > 1.0:
        failures.append(f"scripted turn = {turn:.1f} deg (want +/-360)")
    if table.spin_mask[i0 - 1]:
        failures.append("spin mask bleeds before the segment")

    # Rate cap: the SCRIPTED ramp (yaw_spin minus the live yaw curve) never
    # exceeds the speed limit between masked frames. Total motion may be
    # faster where the plain yaw itself ramps (a fast human turn onset).
    # The one-frame return to the plain curve after the sweep is a raw 360
    # step by design -- the view's nearest-angle tracking renders it as a
    # no-move.)
    dt = float(np.median(np.diff(ch.times)))
    ramp = table.yaw_spin - table.yaw
    diffs = np.abs(np.diff(ramp))
    both = table.spin_mask[:-1] & table.spin_mask[1:]
    max_step = float(np.max(diffs[both])) if both.any() else 0.0
    if max_step > SPIN_MAX_SPEED_DEG * dt * 1.10 + 1.0:
        failures.append(f"scripted sweep speed {max_step / dt:.0f} deg/s > cap")

    # The sweep starts on the plain curve (no entry jump).
    if abs(ramp[i0]) > SPIN_MAX_SPEED_DEG * dt * 1.2:
        failures.append("sweep entry not continuous with plain yaw")

    # Nearest-angle tracking renders the post-sweep return as a no-move.
    for got, want in ((wrap180(420.0), 60.0), (wrap180(-190.0), 170.0),
                      (wrap180(180.0), -180.0)):
        if abs(got - want) > 1e-9:
            failures.append(f"wrap180({want}) = {got}")
    return failures


def test_facing() -> list[str]:
    """Front/back from keypoint laterality; full signed yaw."""
    failures = []
    frames = [make_kp(0.0, 0.9, 90.0, 120.0, bend_l=120.0) for _ in range(12)]

    # The fixture puts shoulder_l on image-LEFT: for an anatomically
    # labelled left keypoint that is the BACK convention (facing camera,
    # the dancer's LEFT side appears image-RIGHT).
    ch = compute_human_channels(kps_from_frames(frames))
    if bool(ch.torso_facing[6]) is not False:
        failures.append(f"fixture (left on image-left) reads facing={ch.torso_facing[6]}")
    if abs(abs(float(ch.torso_yaw_full[6])) - 180.0) > 30.0:
        failures.append(f"back-facing yaw {ch.torso_yaw_full[6]:.0f}, want ~180")

    # Swapping the L/R labels mirrors the figure -> facing the camera.
    swapped = []
    for f in frames:
        g = dict(f)
        g["shoulder_l"], g["shoulder_r"] = f["shoulder_r"], f["shoulder_l"]
        g["hip_l"], g["hip_r"] = f["hip_r"], f["hip_l"]
        swapped.append(g)
    ch2 = compute_human_channels(kps_from_frames(swapped))
    if bool(ch2.torso_facing[6]) is not True:
        failures.append(f"swapped labels read facing={ch2.torso_facing[6]}")
    if abs(float(ch2.torso_yaw_full[6])) > 30.0:
        failures.append(f"front-facing yaw {ch2.torso_yaw_full[6]:.0f}, want ~0")
    return failures


def test_respin_modes() -> list[str]:
    """Per-segment pill states (off / scripted) re-time the sweeps."""
    failures = []
    # Two disjoint fast ramps: +360 then -360 (both ~900 deg/s at 30 fps),
    # separated by more than SPIN_MERGE_FRAMES so they stay two clusters,
    # with enough tail for each rate-capped sweep to complete.
    yaw = np.concatenate([
        np.zeros(6), np.linspace(0.0, 360.0, 12),
        np.full(40, 360.0), np.linspace(360.0, 0.0, 12), np.zeros(30)])
    ch = _fake_channels(yaw)
    crab = CrabModel()
    table = build_crab_commands(ch, crab)
    if len(table.spin_segs or []) != 2:
        return [f"detected {len(table.spin_segs or [])} turn segs, want 2"]
    dirs = [int(s["dir"]) for s in table.spin_segs]
    if dirs != [1, -1]:
        failures.append(f"turn directions {dirs}, want [+1, -1]")

    times = table.times
    last = len(times) - 1

    def turn_total() -> float:
        return float(table.yaw_spin[last] - table.yaw[last])

    # Default (both on): +360 then -360 cancel; the held tail offset is 0.
    if abs(turn_total()) > 2.0:
        failures.append(f"default total turn {turn_total():.0f} (want 0)")

    # First pill off: only the second (-360) sweep remains.
    respin_table(table, [SPIN_MODE_OFF, SPIN_MODE_FULL])
    if abs(turn_total() + 360.0) > 2.0:
        failures.append(f"off+full total turn {turn_total():.0f} (want -360)")
    i0, i1 = table.spin_segs[0]["i0"], table.spin_segs[0]["i1"]
    if table.spin_mask[i0:i1].any():
        failures.append("disabled segment still masked")

    # All off: scripted arrays collapse back to the plain curve.
    respin_table(table, [SPIN_MODE_OFF, SPIN_MODE_OFF])
    if table.spin_mask.any() or not np.allclose(table.yaw_spin, table.yaw):
        failures.append("all-off respin did not restore the plain yaw curve")

    # And a full restore reproduces the default evaluation exactly.
    default_mask = table.spin_mask.copy()
    respin_table(table, [SPIN_MODE_FULL, SPIN_MODE_FULL])
    if not np.allclose(table.yaw_spin,
                       build_crab_commands(ch, crab).yaw_spin):
        failures.append("respin(full) diverged from the default build")
    if np.any(default_mask) and not table.spin_mask.any():
        failures.append("mask lost after full respin")
    return failures


def test_manual_ghost_mapping() -> list[str]:
    """Shared claw fn (scalar ghost-drag path == array playback path) and
    the ghost-arm forward/inverse round-trip."""
    failures = []
    crab = CrabModel()
    signs = crab.calibrate_claw_signs()

    rng = np.random.default_rng(1)
    fore = rng.uniform(0.0, 180.0, 40)
    bend = rng.uniform(0.0, 180.0, 40)
    side = rng.uniform(-1.2, 1.2, 40)
    up = rng.uniform(-0.5, 2.0, 40)
    arr = claw_joints_from_channels("LF", fore, bend, side, up, signs)
    for i in (0, 13, 39):
        scalar = claw_joints_from_channels(
            "LF", float(fore[i]), float(bend[i]), float(side[i]),
            float(up[i]), signs)
        for name in arr:
            if abs(float(arr[name][i]) - float(scalar[name])) > 1e-9:
                failures.append(f"scalar/array mismatch {name}[{i}]")
                break
    clipped = clip_claw_joints(
        claw_joints_from_channels("LF", 200.0, 200.0, 5.0, 5.0, signs),
        crab.joint_range)
    for name, value in clipped.items():
        lo, hi = crab.joint_range[name]
        if not (lo <= float(value) <= hi):
            failures.append(f"clip_claw_joints left {name} out of range")
            break

    # Ghost round-trip: bend/forearm recover exactly, side tracks roll.
    zh = np.array([0.0, 0.0, 1.0])
    d_h = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    d_f = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    sh = np.array([0.0, 0.0, 0.3])
    for pitch in (0.0, 45.0, 90.0):
        sides = []
        for roll in (-70.0, -20.0, 20.0, 70.0):
            bend_deg = 110.0
            elbow, wrist = ghost_arm_points(
                pitch, roll, bend_deg, sh, 0.065, d_h, d_f, zh)
            got = arm_channels_from_points(
                sh, elbow, wrist, sh, d_h, zh, 0.10)
            if abs(got["bend"] - bend_deg) > 1e-6:
                failures.append(
                    f"bend round-trip {got['bend']:.3f} != {bend_deg}")
            sides.append(got["side"])
        if sides != sorted(sides):
            failures.append(f"side not monotone in roll at pitch {pitch}")
    return failures


def main() -> int:
    failures = []

    # Frame set: lean 20 deg, deep crouch 0.5, left arm up 150 with a
    # STRAIGHT forearm (bend_l=180 in the fixture convention), right arm
    # hanging with 120 deg bend.
    frames = [make_kp(20.0, 0.5, 150.0, 120.0, bend_l=180.0) for _ in range(12)]
    kps = kps_from_frames(frames)
    ch = compute_human_channels(kps)

    def check(name, got, want, tol):
        if not (abs(got - want) <= tol):
            failures.append(f"{name}: got {got:.2f}, want {want:.2f} +/- {tol:.2f}")

    mid = ch.torso_roll[len(ch.torso_roll) // 2]
    check("torso_roll(deg)", mid, 20.0, 3.0)
    # Elevation is measured relative to the trunk, which leans 20 deg here:
    # fixture 150 deg from vertical => ~170 deg from the trunk.
    check("elevation_l(deg)", float(ch.elevation_l[6]), 150.0 + 20.0, 15.0)
    check("bend_r(deg)", float(ch.bend_r[6]), 120.0, 15.0)
    check("crouch", float(ch.crouch[6]), 0.5, 0.12)
    print(f"leg_ratio={ch.leg_ratio:.2f} (expect ~2.0)")

    # Crab commands, with a real model for sign calibration.
    crab = CrabModel()
    table = build_crab_commands(ch, crab)
    row = sample_commands(table, 0.2)

    z_lo, z_hi = 0.5 * crab.z_stand, 1.05 * crab.z_stand
    if not (z_lo <= row["base_z"] <= z_hi):
        failures.append(f"base_z {row['base_z']:.3f} outside [{z_lo:.3f},{z_hi:.3f}]")
    print(f"base_z={row['base_z']*1000:.1f} mm (stand={crab.z_stand*1000:.1f})")

    # Crouch 0.5 should lower the body well below standing height.
    if row["base_z"] > 0.92 * crab.z_stand:
        failures.append(f"base_z={row['base_z']:.3f} did not drop for crouch 0.5")

    # Left arm up must lift the LF claw: J1 moved by lift-signed amount.
    j1_delta = row["claw"]["LF_J1_joint"] - crab.joint_range["LF_J1_joint"][0]
    lift_sign = crab.calibrate_claw_signs()["LF"]["lift"]
    print(f"LF J1 cmd={row['claw']['LF_J1_joint']:.3f} "
          f"(home {table.claw['LF_J1_joint'][6]:.3f})")

    # Round-9 semantics: J1 follows the FOREARM elevation (elbow->wrist).
    # Fixture: left forearm straight up (~170 deg), right forearm mostly
    # down (~80 deg) -> LF's J1 must move further from HOME along lift.
    fore_l = float(ch.forearm_l[6])
    fore_r = float(ch.forearm_r[6])
    if not (fore_l > 150 and fore_r < 110):
        failures.append(f"forearm elevations: l={fore_l:.0f}, r={fore_r:.0f} "
                        f"(want ~170 / ~80)")
    home_l, home_r = HOME["LF_J1_joint"], HOME["RF_J1_joint"]
    dl = abs(row["claw"]["LF_J1_joint"] - home_l)
    dr = abs(row["claw"]["RF_J1_joint"] - home_r)
    if lift_sign * (row["claw"]["LF_J1_joint"] - home_l) <= 0 or dl <= dr:
        failures.append(f"J1 forearm mapping: LF d={dl:.2f} RF d={dr:.2f}")
    # J2 and J3 BOTH follow the elbow channel (left 180 = extended,
    # right 120): every one of them must respond on both sides.
    for jname, home in (("LF_J2_joint", HOME["LF_J2_joint"]),
                        ("RF_J2_joint", HOME["RF_J2_joint"]),
                        ("LF_J3_joint", HOME["LF_J3_joint"]),
                        ("RF_J3_joint", HOME["RF_J3_joint"])):
        a = table.claw[jname]
        if abs(float(a[6]) - home) < 0.05:
            failures.append(f"{jname} did not respond to the elbow channel")

    # All claw commands inside usable ranges for a whole randomised table.
    rng = np.random.default_rng(0)
    noisy = Keypoints(
        xyz={k: v + rng.normal(0, 4, v.shape) for k, v in kps.xyz.items()},
        conf=kps.conf, times=kps.times,
    )
    ch2 = compute_human_channels(noisy)
    table2 = build_crab_commands(ch2, crab)
    for name, arr in table2.claw.items():
        lo, hi = crab.joint_range[name]
        if np.any(arr < lo - 1e-6) or np.any(arr > hi + 1e-6):
            failures.append(f"{name} exceeds usable range under noise")
    for arr_name in ("base_z",):
        arr = getattr(table2, arr_name)
        if np.any(np.isnan(arr)):
            failures.append(f"{arr_name} contains NaN")

    failures.extend(test_hidden_legs(crab))
    failures.extend(test_air_gestures(crab))
    failures.extend(test_spin_sweep())
    failures.extend(test_facing())
    failures.extend(test_respin_modes())
    failures.extend(test_manual_ghost_mapping())
    failures.extend(test_multi_person_trc())

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("retarget unit tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
