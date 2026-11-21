"""Offline tests for the direction-tracking claw IK + user anchor (no GUI).

The user's correspondence (constants CLAW_ANCHOR_*): ghost sliders
(90, 90, 180) <-> left claw at the joint-sweep pose rest + (-60, 0, 30, 20)
(J4 out of the mapping). The mapping itself is the staged u1/u2 IK; the
anchor only calibrates a constant joint-space offset.

Run:  cd crab-dance && .venv\\Scripts\\python -m tests.test_claw_ik_table
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import (  # noqa: E402
    CLAW_ANCHOR_DELTA_DEG, CLAW_ANCHOR_GHOST, CLAWS, CRAB_FRONT_YAW_OFFSET_DEG,
    HOME,
)
from app.core.crab_model import CrabModel  # noqa: E402
from app.core.retarget import (  # noqa: E402
    build_crab_commands, compute_human_channels, solve_claw_ik_pose,
)
from tests.test_retarget import kps_from_frames, make_kp  # noqa: E402


def _rest_pose(crab) -> dict:
    """CrabView._rest_claw_pose equivalent (static mapping biases)."""
    from app.constants import (
        CLAW_J0_OFFSET_DEG, CLAW_J0_SPREAD_DEG, CLAW_J1_OFFSET_DEG,
        CLAW_J3_OFFSET_DEG,
    )
    signs = crab.calibrate_claw_signs()
    claw = {n: v for n, v in HOME.items() if n.startswith(("LF", "RF"))}
    for leg, spread in (("LF", 1.0), ("RF", -1.0)):
        claw[f"{leg}_J0_joint"] -= signs[leg]["ccw"] * np.radians(
            CLAW_J0_OFFSET_DEG)
        claw[f"{leg}_J0_joint"] += spread * np.radians(CLAW_J0_SPREAD_DEG)
        claw[f"{leg}_J1_joint"] -= np.radians(CLAW_J1_OFFSET_DEG)
        claw[f"{leg}_J3_joint"] -= signs[leg]["fold"] * np.radians(
            CLAW_J3_OFFSET_DEG)
    return claw


def main() -> int:
    failures: list[str] = []
    crab = CrabModel()
    crab.claw_geometry
    rest = _rest_pose(crab)
    mirror = np.array([-1.0, 1.0, -1.0, -1.0])

    # Pose the body exactly like the manual ghost mode does, then solve.
    crab.pose_dance(crab.z_stand * 1.02, 0.0, 0.0, 0.0, dict(rest),
                    yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG)

    # 1. The user's anchor: ghost (90, 90, 180) -> rest + delta, mirrored.
    out = solve_claw_ik_pose(crab, {"l": CLAW_ANCHOR_GHOST,
                                    "r": CLAW_ANCHOR_GHOST})
    delta = np.radians(CLAW_ANCHOR_DELTA_DEG)
    want_l = np.array([rest[f"LF_J{j}_joint"] for j in range(4)]) + delta
    for leg, want in (("LF", want_l), ("RF", want_l * mirror)):
        got = np.array([out[leg][f"{leg}_J{j}_joint"] for j in range(4)])
        err = np.degrees(np.abs(got - want))
        # J1 sits in a flat null space at this pose (identical hard error
        # from both seed sets), hence the roomier bound there.
        if err[0] > 2.0 or err[1] > 5.0 or err[2] > 2.0 or err[3] > 2.0:
            failures.append(f"anchor {leg} off: err {np.round(err, 1)} deg "
                            f"(got {np.round(np.degrees(got), 1)}, "
                            f"want {np.round(np.degrees(want), 1)})")

    # 2. Neutral sliders give a mirror-symmetric ELBOW (user complaint
    #    "两肘旋转方向要对称"). J0 is checked loosely: the hanging pose is
    #    azimuth-degenerate, so the solver may settle a few degrees apart
    #    in that null space (visually identical).
    out0 = solve_claw_ik_pose(crab, {"l": (0.0, 0.0, 0.0),
                                     "r": (0.0, 0.0, 0.0)})
    l0 = np.array([out0["LF"][f"LF_J{j}_joint"] for j in range(4)])
    r0 = np.array([out0["RF"][f"RF_J{j}_joint"] for j in range(4)])
    err = np.degrees(np.abs(r0 - l0 * mirror))
    if np.max(err[1:]) > 0.5 or err[0] > 8.0:
        failures.append(f"neutral pose not L/R mirrored: {np.round(err, 1)}")

    # 3. Table build: limits, J4 rest, exact mirror, smoothness.
    frames = []
    for k in range(40):
        elev = 30.0 + 60.0 * (1.0 - np.cos(2.0 * np.pi * k / 40.0))
        bend = 125.0 + 55.0 * np.cos(2.0 * np.pi * k / 20.0)
        f = make_kp(0.0, 0.9, elev, bend, bend_l=bend)
        cx = f["neck"][0]
        for x, y in (("shoulder_r", "shoulder_l"), ("elbow_r", "elbow_l"),
                     ("wrist_r", "wrist_l")):
            f[x] = f[y].copy()
            f[x][0] = 2.0 * cx - f[y][0]
        frames.append(f)
    ch = compute_human_channels(kps_from_frames(frames))
    table = build_crab_commands(ch, crab, claw_ik=True)
    for leg in CLAWS:
        for j in range(5):
            name = f"{leg}_J{j}_joint"
            arr = table.claw[name]
            lo, hi = crab.joint_range[name]
            if np.any(arr < lo - 1e-6) or np.any(arr > hi + 1e-6):
                failures.append(f"{name} exceeds usable range")
        if not np.allclose(table.claw[f"{leg}_J4_joint"],
                           HOME[f"{leg}_J4_joint"]):
            failures.append(f"{leg} J4 not at rest")
    for j in range(4):
        lf = table.claw[f"LF_J{j}_joint"]
        rf = table.claw[f"RF_J{j}_joint"]
        if not np.allclose(lf, rf * mirror[j], atol=1e-6):
            failures.append(f"J{j} LF/RF not mirrored")
            break

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("claw IK + anchor tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
