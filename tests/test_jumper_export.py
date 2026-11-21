"""Offline checks for the Jumper/SuperMimic clip export.

  * the slew limiter really bounds every step, and leaves a slow ramp alone
  * `build_dance_clip` writes every channel Jumper's dance loader requires
  * no exported joint channel exceeds the servo-friendly speed bound
  * the base height is shifted onto Jumper's calibrated STAND_Z
  * `clip_stats` reads the same clip back consistently

Run:  .venv/Scripts/python -m tests.test_jumper_export
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import HOME, STAND_Z_DOC  # noqa: E402


def main() -> int:
    from app.core.crab_model import CrabModel
    from app.core.jumper_clip import (  # noqa: E402
        MAX_JOINT_SPEED_RAD_S, _JUMPER_LEGS, _limit_speed, build_dance_clip,
    )
    from app.core.jumper_train import clip_stats  # noqa: E402
    from app.core.retarget import CrabCommandTable  # noqa: E402

    failures: list[str] = []

    # ---- limiter: bounds steps, does not move an already-slow channel
    dt = 1.0 / 30.0
    x = np.array([0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0])
    y = _limit_speed(x, dt, MAX_JOINT_SPEED_RAD_S)
    vmax = np.abs(np.diff(y)).max() / dt
    if vmax > MAX_JOINT_SPEED_RAD_S + 1e-6:
        failures.append(f"limiter step {vmax:.2f} rad/s above the bound")
    slow = np.linspace(0.0, 0.4, 30)
    if not np.allclose(_limit_speed(slow, dt), slow):
        failures.append("limiter altered a channel that was already slow")

    # ---- a short synthetic dance with pitch, yaw and a hop
    n = 45
    times = np.arange(n) * dt
    base_z = np.full(n, 0.0978)
    pitch = np.linspace(0.0, 18.0, n)
    yaw = np.linspace(0.0, 90.0, n)
    jump = np.zeros(n)
    jump[n // 2:] = 0.8
    claw = {
        f"{leg}_J{i}_joint": np.full(n, HOME[f"{leg}_J{i}_joint"])
        for leg in ("LF", "RF") for i in range(5)
    }
    table = CrabCommandTable(
        times=times, base_z=base_z, roll=np.zeros(n), pitch=pitch, yaw=yaw,
        leg_sway=np.zeros(n), claw=claw, jump=jump)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.npz"
        crab = CrabModel()
        try:
            build_dance_clip(table, crab, out)
        finally:
            crab.close()
        with np.load(out) as data:
            keys = set(data.files)
            need = ["dt", "meas_body_z", "meas_body_roll", "meas_body_pitch",
                    "meas_body_yaw"]
            for k, (leg, nj) in enumerate(_JUMPER_LEGS):
                need += [f"meas_L{k}_j{j}" for j in range(nj)]
            missing = [k for k in need if k not in keys]
            if missing:
                failures.append(f"clip missing channels: {missing}")
            dt_src = float(data["dt"])
            worst = 0.0
            for k, (leg, nj) in enumerate(_JUMPER_LEGS):
                for j in range(nj):
                    q = np.asarray(data[f"meas_L{k}_j{j}"])
                    worst = max(worst,
                                float(np.abs(np.diff(q)).max() / dt_src))
            z = np.asarray(data["meas_body_z"]).copy()
        if worst > MAX_JOINT_SPEED_RAD_S + 1e-3:
            failures.append(f"exported joint speed {worst:.2f} rad/s above bound")
        # The non-hop frames should sit near Jumper's calibrated stance.
        shift = STAND_Z_DOC - 0.0978
        if abs(float(z[0] - (0.0978 + shift))) > 0.004:
            failures.append(f"base height not shifted onto STAND_Z: z0={z[0]:.4f}")
        if float(z.max()) <= float(z[n // 2 - 1]):
            failures.append("hop did not raise the exported base")
        stats = clip_stats(out)
        if stats["frames"] != n or abs(stats["duration_s"] - n * dt_src) > 1e-6:
            failures.append(f"clip_stats mismatch: {stats['frames']} frames")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"jumper export tests passed (worst joint speed {worst:.2f} rad/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
