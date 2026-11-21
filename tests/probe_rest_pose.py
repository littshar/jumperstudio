"""Headless rest-pose calibration probe.

Measures, with mj_forward on the real model:
  1. mirror symmetry of the two claws about the robot's sagittal (y->-y)
     plane, as a function of a UNIFORM J0 delta (what the joint-sweep
     dialog applies) -- finds the delta where the claws sit symmetric.
  2. the tilt of the claw arm plane (upper_arm -> forearm -> palm) from
     horizontal, as a function of a uniform J1 delta -- finds the delta
     where "the elbow plane" is parallel to the ground.

Run:  .venv/Scripts/python -m tests.probe_rest_pose
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import HOME
from app.core.crab_model import CrabModel

_CHAIN = ("shoulder_link", "upper_arm_link", "forearm_link", "palm_link",
          "finger_link")
_MIRROR = np.array([1.0, -1.0, 1.0])   # LF/RF swap sides along y


def main() -> int:
    crab = CrabModel()
    signs = crab.calibrate_claw_signs()
    print("probed claw signs:", signs)

    def pose(joint: int | None = None, delta: float = 0.0,
             mirrored: bool = False) -> None:
        mujoco.mj_resetData(crab.model, crab.data)
        crab.set_home_joints()
        crab.data.qpos[crab.free_adr + 2] = 0.20
        if joint is not None:
            for leg in ("LF", "RF"):
                d = delta * (-1.0 if (mirrored and leg == "RF") else 1.0)
                adr = crab.joint_adr[f"{leg}_J{joint}_joint"]
                crab.data.qpos[adr] = crab.clamp(
                    f"{leg}_J{joint}_joint", crab.data.qpos[adr] + d)
        mujoco.mj_forward(crab.model, crab.data)

    def points(leg: str) -> list[np.ndarray]:
        return [np.array(crab.data.body(f"{leg}_{n}").xpos) for n in _CHAIN]

    def sym_error() -> float:
        pl, pr = points("LF"), points("RF")
        return float(sum(np.linalg.norm(a - b * _MIRROR)
                         for a, b in zip(pl, pr)))

    def plane_tilt_deg() -> float:
        """Angle of the (upper_arm, forearm, palm) plane normal vs vertical."""
        out = []
        for leg in ("LF", "RF"):
            p = points(leg)
            a, b, c = p[1], p[2], p[3]
            n = np.cross(b - a, c - a)
            n /= max(np.linalg.norm(n), 1e-9)
            out.append(np.degrees(np.arccos(min(abs(n[2]), 1.0))))
        return float(np.mean(out))

    # --- HOME baseline -----------------------------------------------------
    pose()
    print(f"\nHOME: mirror symmetry err {sym_error() * 1000:7.2f} mm, "
          f"arm-plane tilt {plane_tilt_deg():6.1f} deg from horizontal")

    # --- J0 sweep ----------------------------------------------------------
    print("\nuniform J0 delta -> mirror symmetry error (mm):")
    best = (None, 1e9)
    for d in np.arange(-90.0, 90.1, 5.0):
        pose(0, np.radians(d))
        e = sym_error() * 1000
        if e < best[1]:
            best = (d, e)
        if int(d) % 15 == 0:
            print(f"  J0 {d:+5.0f} deg: {e:8.2f}")
    # refine
    for d in np.arange(best[0] - 5.0, best[0] + 5.1, 1.0):
        pose(0, np.radians(d))
        e = sym_error() * 1000
        if e < best[1]:
            best = (d, e)
    print(f"  => most symmetric at uniform J0 delta {best[0]:+.1f} deg "
          f"(err {best[1]:.2f} mm)")

    print("\nmirrored J0 delta (LF +d / RF -d) -> symmetry error (mm):")
    bestm = (None, 1e9)
    for d in np.arange(-60.0, 60.1, 10.0):
        pose(0, np.radians(d), mirrored=True)
        e = sym_error() * 1000
        if e < bestm[1]:
            bestm = (d, e)
    print(f"  => best mirrored J0 delta {bestm[0]:+.0f} deg (err {bestm[1]:.2f} mm)")

    # --- J1 sweep ----------------------------------------------------------
    print("\nuniform J1 delta -> arm-plane tilt from horizontal (deg):")
    best1 = (None, 999.0)
    for d in np.arange(-90.0, 20.1, 5.0):
        pose(1, np.radians(d))
        t = plane_tilt_deg()
        if t < best1[1]:
            best1 = (d, t)
        if int(d) % 15 == 0:
            print(f"  J1 {d:+5.0f} deg: tilt {t:6.1f}")
    for d in np.arange(best1[0] - 5.0, best1[0] + 5.1, 1.0):
        pose(1, np.radians(d))
        t = plane_tilt_deg()
        if t < best1[1]:
            best1 = (d, t)
    print(f"  => arm plane parallel to ground at uniform J1 delta "
          f"{best1[0]:+.1f} deg (tilt {best1[1]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
