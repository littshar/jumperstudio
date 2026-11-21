"""Generate a synthetic stick-figure dance video for end-to-end testing.

A white skeleton on black, 30 fps, 12 s:
  - trunk sways +/-20 deg (sin, 0.5 Hz)
  - arms alternate overhead (0.8 Hz, opposite phase)
  - periodic squat to 50% (0.33 Hz)

RTMPose detects stick figures reliably enough to validate the full pipeline.

Run:  PYTHONPATH=. python -m tests.make_synth_video [out.mp4]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 720
FPS = 30
DUR = 12.0
TORSO = 150.0


def frame_pose(t: float) -> dict[str, np.ndarray]:
    lean_deg = 20.0 * math.sin(2 * math.pi * 0.5 * t)
    crouch = 0.75 + 0.25 * math.sin(2 * math.pi * 0.33 * t - math.pi / 2)  # 0.5..1
    elev_l = 90 + 80 * math.sin(2 * math.pi * 0.8 * t)
    elev_r = 90 + 80 * math.sin(2 * math.pi * 0.8 * t + math.pi)
    bend_l = 30 + 25 * math.sin(2 * math.pi * 0.8 * t)
    bend_r = 30 + 25 * math.sin(2 * math.pi * 0.8 * t + math.pi)

    lean = math.radians(lean_deg)
    hip = np.array([W / 2, 520.0])
    neck = hip + TORSO * np.array([math.sin(lean), -math.cos(lean)])
    sho = neck + np.array([0.0, -0.15 * TORSO])

    def arm(elev_deg, bend_deg, direction):
        elev = math.radians(elev_deg)
        bend = math.radians(bend_deg)
        perp = np.array([direction, 0.0])
        down = np.array([math.sin(lean), math.cos(lean)])
        u = math.cos(elev) * down + math.sin(elev) * perp
        elbow = sho + 65 * u
        f = -math.cos(bend) * u + math.sin(bend) * perp
        wrist = elbow + 60 * f
        return elbow, wrist

    elbow_l, wrist_l = arm(elev_l, bend_l, -1)
    elbow_r, wrist_r = arm(elev_r, bend_r, +1)

    seg, total = 100.0, 200.0
    drop = crouch * total

    def leg(sx):
        hip_s = hip + np.array([sx * 30, 0.0])
        ank = hip_s + np.array([sx * 20, drop])
        mid = (hip_s + ank) / 2
        off = math.sqrt(max(seg * seg - (drop / 2) ** 2, 0.0))
        knee = mid + np.array([sx * off, 0.0])
        return hip_s, knee, ank

    hip_l, knee_l, ank_l = leg(-1)
    hip_r, knee_r, ank_r = leg(+1)
    nose = neck + np.array([-30 * math.sin(lean), -40.0])

    return {
        "nose": nose, "neck": neck,
        "shoulder_l": sho + [-35, 0], "shoulder_r": sho + [35, 0],
        "elbow_l": elbow_l, "wrist_l": wrist_l,
        "elbow_r": elbow_r, "wrist_r": wrist_r,
        "hip_l": hip_l, "hip_r": hip_r,
        "knee_l": knee_l, "knee_r": knee_r,
        "ankle_l": ank_l, "ankle_r": ank_r,
    }


EDGES = (
    ("neck", "nose"),
    ("shoulder_l", "shoulder_r"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("hip_l", "hip_r"),
    ("shoulder_l", "hip_l"), ("shoulder_r", "hip_r"),
    ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"),
)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_synth.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (W, H))
    if not writer.isOpened():
        print("cannot open VideoWriter")
        return 1
    n = int(DUR * FPS)
    for i in range(n):
        t = i / FPS
        pose = frame_pose(t)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Filled torso makes the figure detectable for RTMPose.
        torso = np.array([pose["shoulder_l"], pose["shoulder_r"],
                          pose["hip_r"], pose["hip_l"]], dtype=np.int32)
        cv2.fillPoly(img, [torso], (200, 200, 200))
        # Head.
        cv2.circle(img, tuple(((pose["nose"] + pose["neck"]) / 2).astype(int)),
                   34, (255, 255, 255), -1)
        for a, b in EDGES:
            pa, pb = pose[a], pose[b]
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     (255, 255, 255), 16, cv2.LINE_AA)
        for name, p in pose.items():
            if name in ("nose", "neck"):
                continue
            cv2.circle(img, tuple(p.astype(int)), 12, (255, 255, 255), -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        writer.write(img)
    writer.release()
    print(f"wrote {out} ({n} frames, {DUR}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
