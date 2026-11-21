"""Unit tests for the multi-person track repair pass (no GUI, no Sports2D).

Covers the three failure modes measured on real group-dance clips:
  * one human split into sequential fragments (tracker slot retirement)
  * one human tracked twice at once (duplicate slots)
  * fragmentary / frozen tracks that should not become crabs

Run:  .venv/Scripts/python -m tests.test_person_repair
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.retarget import KP_NAMES, Keypoints
from app.core.s2d_pipeline import _repair_persons

# Simple standing skeleton offsets (x, y) around a body centre.
_OFFSETS = {
    "nose": (0, -120), "neck": (0, -100),
    "shoulder_l": (-18, -95), "shoulder_r": (18, -95),
    "elbow_l": (-25, -50), "elbow_r": (25, -50),
    "wrist_l": (-28, -10), "wrist_r": (28, -10),
    "hip_l": (-15, 0), "hip_r": (15, 0),
    "knee_l": (-15, 50), "knee_r": (15, 50),
    "ankle_l": (-15, 100), "ankle_r": (15, 100),
}


def make_person(n: int, x0: float, y0: float, window=None,
                moving: bool = True, phase: float = 0.0) -> Keypoints:
    """A synthetic track: torso ~100 px, dancing in place when moving.

    window: (i0, i1) valid frame slice; None = valid throughout. Non-moving
    tracks hold the exact same pose every frame (Sports2D's occlusion fill).
    """
    t = np.arange(n) / 30.0
    xyz, conf = {}, {}
    for name, (dx, dy) in _OFFSETS.items():
        col = np.full((n, 2), np.nan)
        c = np.zeros(n)
        lo, hi = window if window is not None else (0, n)
        for i in range(lo, hi):
            if moving:
                col[i] = (x0 + dx + 6.0 * np.sin(2 * np.pi * 0.5 * t[i] + phase)
                          + 8 * np.sin(0.7 * t[i] + phase),
                          y0 + dy + 2 * np.sin(2 * np.pi * 0.25 * t[i]))
            else:   # occlusion filler: exact same pose every frame
                col[i] = (x0 + dx, y0 + dy)
            c[i] = 1.0
        xyz[name] = col
        conf[name] = c
    return Keypoints(xyz=xyz, conf=conf, times=t)


def valid_frames(kps: Keypoints) -> np.ndarray:
    hip = kps.xy("hip_l")
    return ~np.isnan(hip[:, 0])


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"{name}: {detail}")

    # ---- 1. sequential fragments of one human stitch into one track ----
    n = 300
    frag_a = make_person(n, 400, 300, window=(0, 90), phase=0.0)
    frag_b = make_person(n, 410, 300, window=(120, 300), phase=1.0)
    kept, stats = _repair_persons([frag_a, frag_b])
    check("stitch count", len(kept) == 1,
          f"got {len(kept)} (stats {stats})")
    check("stitch merged flag", stats["merged"] == 1, f"stats {stats}")
    both = valid_frames(kept[0])
    check("stitch coverage", both[:90].all() and both[120:].all()
          and not both[90:120].any(),
          f"valid pattern wrong")

    # ---- 2. distant neighbours stay separate ----
    left = make_person(200, 150, 300)
    right = make_person(200, 650, 300, phase=0.7)   # 500 px apart, 5 torso
    kept, stats = _repair_persons([left, right])
    check("neighbours count", len(kept) == 2, f"got {len(kept)} (stats {stats})")
    check("neighbours unmerged", stats["merged"] == 0, f"stats {stats}")

    # ---- 3. duplicates (two slots, same human) collapse, live wins ----
    n = 200
    real = make_person(n, 400, 300, window=(0, 150))          # dances 0-5 s
    dup = make_person(n, 405, 300, window=(50, 200), phase=0.3)  # second slot
    kept, stats = _repair_persons([real, dup])
    check("duplicate count", len(kept) == 1, f"got {len(kept)} (stats {stats})")
    v = valid_frames(kept[0])
    # Live-pass preference: each frame comes from a member that was valid
    # there, so the union of both windows is covered.
    check("duplicate coverage", v[:150].all() and v[150:].any(),
          "merged track lost frames")

    # ---- 4. frozen filler track dropped, healthy kept ----
    n = 300
    healthy = make_person(n, 400, 300)
    frozen = make_person(n, 800, 300, window=(0, n), moving=False)  # statue
    kept, stats = _repair_persons([healthy, frozen])
    check("gate count", len(kept) == 1, f"got {len(kept)} (stats {stats})")
    check("gate dropped flag", stats["dropped"] == 1, f"stats {stats}")

    # ---- 5. ordering: most-tracked first; a lone garbage track survives ----
    short = make_person(300, 400, 300, window=(0, 60))
    long_ = make_person(300, 600, 300, window=(0, 290))
    kept, _ = _repair_persons([short, long_])
    check("ranking", valid_frames(kept[0]).sum() >= 280,
          f"first kept only {valid_frames(kept[0]).sum()} frames")
    lone, _ = _repair_persons([make_person(100, 0, 0, window=(0, 10))])
    check("lone survivor", len(lone) == 1, "the only track must be kept")

    print("FAILURES:" if failures else "person-repair unit tests passed")
    for f in failures:
        print(" -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
