"""Regression: an imported floor image must actually tint the floor.

`MjSpec.add_material(textures=[...])` indexes material texture *slots*; slot 1
is the diffuse map used by MJCF's `texture=` attribute. Putting the name in
slot 0 (emission) left the diffuse white, so every custom floor rendered as a
pure-white plane. Render a solid red floor and sample it.

Run:  .venv/Scripts/python -m tests.test_floor_texture
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from app.constants import HOME
    from app.core.crab_model import CrabModel

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "red_floor.png"
        cv2.imwrite(str(png), np.full((64, 64, 3), (0, 0, 255), np.uint8))

        crab = CrabModel(floor_texture=png)
        try:
            mat = crab.model.mat("user_floor_mat")
            if int(np.asarray(mat.texid).ravel()[1]) < 0:
                failures.append("floor texture is not in the diffuse slot")
            claw = {n: HOME[n] for n in HOME if n.startswith(("LF", "RF"))}
            crab.pose_dance(crab.z_stand, 0, 0, 0, claw)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.0, 0.0, 0.07]
            cam.distance = 0.48
            cam.azimuth = 135.0
            cam.elevation = -18.0
            frame = crab.render(camera=cam, width=320, height=240)
            r, g, b = (int(v) for v in frame[200, 16])
            if not (r > 80 and g < 40 and b < 40):
                failures.append(
                    f"custom floor did not render red: RGB=({r},{g},{b})")
        finally:
            crab.close()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("floor texture checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
