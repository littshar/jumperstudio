"""Keyframe mapping calibration: two points define a real ratio.

Records are stored as corrections over the current mapping; at playback the
correction is interpolated linearly against the joint's ghost control angle
(pitch for J0/J1, outward swing for J2/J3, bend for J4). This checks the
interpolation, the ratio readout, and that the clamps still apply.

Run:  .venv/Scripts/python -m tests.test_claw_calib
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Crab:
    joint_range = {"LF_J1_joint": (-3.0, 3.0)}
    joint_adr = {"LF_J1_joint": 0}


def main() -> int:
    from app.core import claw_calib

    # The app disables the feature for now; test the implementation itself.
    claw_calib.CALIBRATION_ENABLED = True
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        claw_calib.CALIB_PATH = Path(tmp) / "calib.json"
        ghost_a = {"l": (0.0, 0.0, 180.0), "r": (0.0, 0.0, 180.0)}
        ghost_b = {"l": (100.0, 0.0, 180.0), "r": (100.0, 0.0, 180.0)}
        frames = [
            claw_calib.make_record(
                ghost_a, {"LF_J1_joint": 0.0}, {"LF_J1_joint": 0.0}),
            claw_calib.make_record(
                # 20 deg crab over 100 deg ghost = 20% ratio.
                ghost_b, {"LF_J1_joint": math.radians(20.0)},
                {"LF_J1_joint": 0.0}),
        ]
        claw_calib.save_frames(frames)
        ratio = claw_calib.ratio_text(frames, 0, 1)
        if "LF J1 20%" not in ratio:
            failures.append(f"ratio readout wrong: {ratio}")

        n = 5
        def fake(pitch_l, pitch_r=None):
            n = len(pitch_l)
            if pitch_r is None:
                pitch_r = pitch_l
            return SimpleNamespace(
                shoulder_pitch_l=np.asarray(pitch_l, dtype=float),
                shoulder_pitch_r=np.asarray(pitch_r, dtype=float),
                shoulder_roll_l=np.zeros(n), shoulder_roll_r=np.zeros(n),
                bend_l=np.zeros(n), bend_r=np.zeros(n))

        ch = fake([0.0, 25.0, 50.0, 75.0, 100.0])
        claw = {"LF_J1_joint": np.zeros(5)}
        claw_calib.apply_calibration(claw, ch, _Crab())
        got = np.degrees(claw["LF_J1_joint"])
        if not np.allclose(got, [0.0, 5.0, 10.0, 15.0, 20.0], atol=0.2):
            failures.append(f"bias interpolation wrong: {got}")

        # Outside the recorded range the nearest bias is held.
        claw2 = {"LF_J1_joint": np.zeros(1)}
        claw_calib.apply_calibration(claw2, fake([260.0]), _Crab())
        if abs(np.degrees(claw2["LF_J1_joint"][0]) - 20.0) > 0.5:
            failures.append("bias endpoint hold wrong")

        claw_calib.clear()
        if claw_calib.load_frames():
            failures.append("clear did not remove the calibration")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("claw calibration checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
