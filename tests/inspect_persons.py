"""Diagnose cached multi-person TRCs: fragmentation, swaps, held-pose ratio.

Run:  .venv/Scripts/python -m tests.inspect_persons <result_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.s2d_pipeline import load_persons, KP_NAMES


def torso(kps) -> np.ndarray:
    """(N,2) mid-hip -> neck vector / centroid + torso size per frame."""
    hip = (kps.xy("hip_l") + kps.xy("hip_r")) / 2
    sho = (kps.xy("shoulder_l") + kps.xy("shoulder_r")) / 2
    return hip, sho, np.linalg.norm(sho - hip, axis=1)


def main() -> int:
    d = Path(sys.argv[1])
    persons = load_persons(d)
    n = len(persons[0].times)
    print(f"{len(persons)} usable persons, {n} frames")

    stats = []
    for i, kps in enumerate(persons):
        hip, sho, tsize = torso(kps)
        core_nan = np.isnan(hip[:, 0]) | np.isnan(sho[:, 0])
        valid = ~core_nan
        # static (held) frames: zero total displacement over all keypoints
        allxy = np.concatenate([kps.xy(name) for name in KP_NAMES], axis=1)
        disp = np.nansum(np.abs(np.diff(allxy, axis=0)), axis=1)
        static = np.concatenate([[0], disp < 1e-9])
        live = valid & ~static
        # teleports: centroid jump vs typical speed
        c = (hip + sho) / 2
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        med_ts = np.nanmedian(tsize) or 1.0
        tele = (step > 1.0 * med_ts).sum()
        stats.append({
            "i": i, "valid%": 100 * valid.mean(), "live%": 100 * live.mean(),
            "static%": 100 * static.mean(),
            "med_x": float(np.nanmedian(c[:, 0])), "med_ts": med_ts,
            "teleports": int(tele),
            "t0": float(kps.times[~core_nan][0]) if valid.any() else -1,
            "t1": float(kps.times[~core_nan][-1]) if valid.any() else -1,
        })

    for s in stats:
        print(f"  p{s['i']}: valid {s['valid%']:5.1f}%  live {s['live%']:5.1f}%  "
              f"static {s['static%']:5.1f}%  x@{s['med_x']:7.0f}  "
              f"torso {s['med_ts']:6.0f}px  teleports {s['teleports']:3d}  "
              f"t [{s['t0']:6.2f}..{s['t1']:6.2f}]s")

    # pairwise: overlap in time + median distance when co-valid (duplicate?)
    print("\npairwise (co-valid overlap / median px distance / torso units):")
    for a in range(len(persons)):
        for b in range(a + 1, len(persons)):
            ca = (torso(persons[a])[0] + torso(persons[a])[1]) / 2
            cb = (torso(persons[b])[0] + torso(persons[b])[1]) / 2
            va = ~np.isnan(ca[:, 0])
            vb = ~np.isnan(cb[:, 0])
            both = va & vb
            ts = min(stats[a]["med_ts"], stats[b]["med_ts"]) or 1.0
            if both.any():
                dist = np.linalg.norm(ca[both] - cb[both], axis=1)
                print(f"  p{a}-p{b}: co {100 * both.mean():5.1f}%  "
                      f"med {np.median(dist):7.0f}px  "
                      f"{np.median(dist) / ts:5.2f} torso")
    return 0


if __name__ == "__main__":
    sys.exit(main())
