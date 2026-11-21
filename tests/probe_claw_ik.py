"""GMR-style claw IK feasibility probe.

The question (user, 2026-11-16): the crab's claws have too few DoF to cover
human arm motions -- the analytic channel mapping saturates whenever a pose
falls outside its hand-tuned gains. GMR (github.com/YanjieZe/GMR, ICRA 2026)
solves this differently: a per-frame multi-objective damped-least-squares IK
over the available joints, inside joint limits, warm-started from the
previous frame -- an under-actuated chain then always lands on the CLOSEST
reachable pose instead of a saturated one.

This probe re-implements that solver core against the jumper claw chain
(J0..J3 position the palm; J4 stays the gripper channel) and compares it
with the current analytic mapping for a battery of human arm poses:

  target definition (scale-invariant: the human ghost is ~30 cm tall,
  the crab ~10 cm, so absolute positions are matched in DIRECTION +
  RELATIVE EXTENT around the crab's own shoulder):
      u_dir = (wrist - shoulder) / |wrist - shoulder|       (ghost basis)
      rho   = |wrist - shoulder| / (2 L)                     (0..1)
      target_palm  = crab_shoulder + u_dir * clamp(rho) * R_claw
      target_elbow = crab_shoulder + u_dir_e * clamp(rho_e) * R_elbow
  The elbow objective runs at a lower weight -- that is what makes the
  whole claw arm read as "the same pose", not just the tip (GMR matches
  intermediate links the same way).

Outputs: a residual table on stdout and tests/claw_ik_sheet.png
(one row per pose: analytic mapping | IK solve, ghost target overlaid).

Run:  .venv\\Scripts\\python tests\\probe_claw_ik.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from app.constants import CLAWS, HOME  # noqa: E402
from app.core.crab_model import CrabModel  # noqa: E402
from app.core.retarget import (  # noqa: E402
    arm_channels_from_points, claw_joints_from_channels,
    clip_claw_joints, ghost_arm_points,
)

OUT_PNG = Path(__file__).resolve().parent / "claw_ik_sheet.png"

# Ghost basis, exactly as CrabView._ghost_segments defines it at yaw 0
# (user-calibrated: d_h renders screen-left = the dancer's image-right).
ZH = np.array([0.0, 0.0, 1.0])
D_H = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
D_F = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)     # toward the camera
GHOST_LEN = 0.065          # human upper/fore-arm segment length (m), view value
MANUAL_TORSO = 0.10        # channel normaliser, view value

# IK weights (the "multi-objective" part). Position errors are in metres,
# so the palm weight scales a 2 cm miss to ~1 solver unit.
W_PALM = 50.0
W_ELBOW = 18.0
W_POSTURE = 1.0            # stay near HOME in the null space
MAX_STEP = 0.35            # rad per IK iteration (GMR-style trust region)
IK_ITERS = 12
RHO_MIN, RHO_MAX = 0.15, 1.0
VSTEP_MAX = 0.30           # rad/frame velocity cap (GMR: 3*pi rad/s @ 30 fps)

# Human arm poses (shoulder pitch, shoulder roll, elbow bend), per side.
# roll is + toward the figure's image-right; the LEFT ghost shoulder sits
# at +d_h, so "left arm out sideways" is roll_l = -90.
POSES: list[tuple[str, tuple[float, float, float]]] = [
    ("hang        ", (0.0, 0.0, 180.0)),        # arms straight down
    ("tpose       ", (0.0, -90.0, 180.0)),      # arms straight out
    ("overhead    ", (0.0, -160.0, 180.0)),     # arms up, slightly out
    ("front raise ", (90.0, 0.0, 180.0)),       # pointing at the camera
    ("cross chest ", (70.0, -60.0, 45.0)),      # folded in front
    ("open V      ", (0.0, -120.0, 150.0)),     # wide dance V
    ("heart top   ", (0.0, -175.0, 60.0)),      # both hands meeting overhead
    ("point down  ", (0.0, -40.0, 170.0)),      # down and out
    ("shrug       ", (20.0, -80.0, 90.0)),      # mid everything
]


# --------------------------------------------------------------- solver ----
def claw_ik(crab: CrabModel, leg: str, q_init: np.ndarray,
            palm_t: np.ndarray, elbow_t: np.ndarray,
            q_ref: np.ndarray, q_prev: np.ndarray | None = None,
            iters: int = IK_ITERS) -> tuple[np.ndarray, float]:
    """Damped least-squares IK over J0..J3 for palm + elbow point targets.

    Objectives (stacked, exactly GMR/mink shaped):
      [W_PALM  * J_palm  ; W_ELBOW * J_elbow ; W_POSTURE * 0.3 * I] dq
        = [W_PALM * e_palm; W_ELBOW * e_elbow; W_POSTURE * 0.3 * (q_ref - q)]
    Each iteration's step is trust-region clipped, then projected into the
    joint box; an optional q_prev caps the per-frame travel (velocity limit).
    Returns (q, palm residual in m after the final mj_forward).
    """
    m, d = crab.model, crab.data
    names = [f"{leg}_J{i}_joint" for i in range(4)]
    adr = [crab.joint_adr[n] for n in names]
    dof = [int(m.joint(n).dofadr[0]) for n in names]
    site_id = m.site(leg).id
    forearm_id = m.body(f"{leg}_forearm_link").id
    lo = np.array([crab.joint_range[n][0] for n in names])
    hi = np.array([crab.joint_range[n][1] for n in names])

    q = np.clip(np.array(q_init, dtype=float), lo, hi)
    if q_prev is not None:
        q = np.clip(q, q_prev - VSTEP_MAX, q_prev + VSTEP_MAX)

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    eye4 = np.eye(4)
    for _ in range(iters):
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        J = W_PALM * jacp[:, dof]
        e = W_PALM * (palm_t - np.asarray(d.site(leg).xpos))
        mujoco.mj_jacBody(m, d, jacp, jacr, forearm_id)
        J = np.vstack([J, W_ELBOW * jacp[:, dof], W_POSTURE * 0.3 * eye4])
        e = np.concatenate([
            e, W_ELBOW * (elbow_t - np.asarray(d.body(f"{leg}_forearm_link").xpos)),
            W_POSTURE * 0.3 * (q_ref - q)])
        dq = np.linalg.lstsq(J, e, rcond=None)[0]
        dq = np.clip(dq, -MAX_STEP, MAX_STEP)
        q = np.clip(q + dq, lo, hi)

    for a, v in zip(adr, q):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    res = float(np.linalg.norm(palm_t - np.asarray(d.site(leg).xpos)))
    return q, res


def measure_reach(crab: CrabModel, leg: str) -> tuple[float, float]:
    """Coarse envelope of the claw chain from its shoulder.

    Returns (R_palm, R_forearm): the max |palm site - shoulder| and
    |forearm origin - shoulder| over a joint grid -- the claw's practical
    reach, used to scale human extent onto the crab.
    """
    m, d = crab.model, crab.data
    names = [f"{leg}_J{i}_joint" for i in range(4)]
    adr = [crab.joint_adr[n] for n in names]
    shoulder = np.asarray(d.body(f"{leg}_shoulder_link").xpos).copy()
    best_p = best_e = 0.0
    # J0 rotates about a vertical axis THROUGH the shoulder anchor, so it
    # cannot change the radius -- grid J1..J3 only, J0 stays at HOME.
    d.qpos[adr[0]] = HOME[names[0]]
    grid = [np.linspace(*crab.joint_range[n], 9) for n in names[1:]]
    for j1 in grid[0]:
        for j2 in grid[1]:
            for j3 in grid[2]:
                for a, v in zip(adr[1:], (j1, j2, j3)):
                    d.qpos[a] = v
                mujoco.mj_forward(m, d)
                best_p = max(best_p, float(np.linalg.norm(
                    np.asarray(d.site(leg).xpos) - shoulder)))
                best_e = max(best_e, float(np.linalg.norm(
                    np.asarray(d.body(f"{leg}_forearm_link").xpos) - shoulder)))
    return best_p, best_e


def analytic_claw_q(crab: CrabModel, leg: str, human_side: str,
                    shoulder: np.ndarray, elbow: np.ndarray,
                    wrist: np.ndarray, centre: np.ndarray,
                    signs: dict) -> np.ndarray:
    """Current production mapping for the same pose (J0..J3 only)."""
    ch = arm_channels_from_points(shoulder, elbow, wrist, centre,
                                  D_H, ZH, MANUAL_TORSO)
    raw = claw_joints_from_channels(
        leg, ch["forearm"], ch["bend"], ch["side"], ch["up"], signs)
    q = clip_claw_joints(raw, crab.joint_range)
    return np.array([q[f"{leg}_J{i}_joint"] for i in range(4)])


def palm_pos(crab: CrabModel, leg: str, q: np.ndarray) -> np.ndarray:
    m, d = crab.model, crab.data
    for i in range(4):
        d.qpos[crab.joint_adr[f"{leg}_J{i}_joint"]] = q[i]
    mujoco.mj_forward(m, d)
    return np.asarray(d.site(leg).xpos).copy()


def ghost_pose_for(human_side: str, pitch: float, roll: float,
                   bend: float, shoulder_anchor: np.ndarray,
                   seg_len: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ghost elbow/wrist for the pose, anchored at the crab's shoulder.

    Right-side roll mirrors the left (roll is + toward image-right for both
    sides in the channel convention). seg_len sets the visual scale.
    """
    r = roll if human_side == "l" else -roll
    elbow, wrist = ghost_arm_points(pitch, r, bend, shoulder_anchor,
                                    seg_len, D_H, D_F, ZH)
    return shoulder_anchor, elbow, wrist


# ----------------------------------------------------------------- main ----
def main() -> None:
    crab = CrabModel()
    signs = crab.calibrate_claw_signs()

    # Standing HOME body once; every solve below only rewrites claw joints.
    home_claw = {n: HOME[n] for n in HOME if n.startswith(CLAWS)}
    crab.pose_dance(crab.z_stand, 0.0, 0.0, 0.0, home_claw)

    anchors, reach = {}, {}
    for leg in CLAWS:
        anchors[leg] = np.asarray(
            crab.data.body(f"{leg}_shoulder_link").xpos).copy()
        reach[leg] = measure_reach(crab, leg)
        print(f"{leg}: shoulder z={anchors[leg][2]*1000:.1f} mm, "
              f"reach palm={reach[leg][0]*1000:.0f} mm "
              f"forearm={reach[leg][1]*1000:.0f} mm")
    centre = float(np.mean([anchors[leg][2] for leg in CLAWS])) * np.array(
        [0.0, 0.0, 1.0])

    # The ghost is drawn with human proportions in the residual table but
    # rendered at claw scale for the sheet (visual direction comparison).
    seg_vis = {leg: reach[leg][0] / 2.0 for leg in CLAWS}

    q_prev: dict[str, np.ndarray | None] = {leg: None for leg in CLAWS}
    rows, cells = [], []
    for name, (pitch, roll, bend) in POSES:
        row = {"name": name}
        q_an_all: dict[str, np.ndarray] = {}
        for human_side, leg in (("l", "LF"), ("r", "RF")):
            # Human target in ghost space, then rescaled onto the crab.
            s_g, e_g, w_g = ghost_pose_for(
                human_side, pitch, roll, bend, anchors[leg], GHOST_LEN)
            u_dir = (w_g - s_g) / max(np.linalg.norm(w_g - s_g), 1e-9)
            rho = np.clip(np.linalg.norm(w_g - s_g) / (2 * GHOST_LEN),
                          RHO_MIN, RHO_MAX)
            palm_t = anchors[leg] + u_dir * rho * reach[leg][0]
            # Physical clamp the pure direction target misses: at "arms
            # hanging down" the straight-line palm target sits 118 mm UNDER
            # the floor (shoulder 108 mm, claw 226 mm). A real integration
            # must keep the palm above the deck -- fold instead of pierce.
            palm_t = palm_t.copy()
            palm_t[2] = max(palm_t[2], 0.008)
            u_e = (e_g - s_g) / max(np.linalg.norm(e_g - s_g), 1e-9)
            rho_e = np.clip(np.linalg.norm(e_g - s_g) / (2 * GHOST_LEN),
                            RHO_MIN, RHO_MAX)
            elbow_t = anchors[leg] + u_e * rho_e * reach[leg][1]

            q_home = np.array([HOME[f"{leg}_J{i}_joint"] for i in range(4)])
            q_an = analytic_claw_q(crab, leg, human_side, s_g, e_g, w_g,
                                   centre, signs)
            q_an_all[leg] = q_an
            # Static battery: fresh solve per pose (no velocity cap between
            # unrelated poses). Two starts -- HOME and the analytic solution
            # -- keep the DLS out of local minima.
            best = None
            for q0 in (q_home, q_an):
                q_try, res_try = claw_ik(crab, leg, q0, palm_t, elbow_t,
                                         q_ref=q_home)
                if best is None or res_try < best[1]:
                    best = (q_try, res_try)
            q_ik, res_ik = best
            q_prev[leg] = q_ik
            res_an = float(np.linalg.norm(palm_pos(crab, leg, q_an) - palm_t))
            row[f"an_{leg}"] = res_an
            row[f"ik_{leg}"] = res_ik
            row[f"clip_{leg}"] = int(sum(
                (q_an[i] <= crab.joint_range[f"{leg}_J{i}_joint"][0] + 1e-3)
                or (q_an[i] >= crab.joint_range[f"{leg}_J{i}_joint"][1] - 1e-3)
                for i in range(4)))
        rows.append(row)

        # Render the row: ghost target + crab posed each way.
        def ghost_segments(vis_len: dict) -> dict:
            segs = {}
            for human_side, leg in (("l", "LF"), ("r", "RF")):
                part = "l" if leg == "LF" else "r"
                s, e, w = ghost_pose_for(
                    human_side, pitch, roll, bend, anchors[leg], vis_len[leg])
                segs[f"ghost_{part}u"] = (s, e)
                segs[f"ghost_{part}f"] = (e, w)
            segs["ghost_head"] = (anchors["LF"], anchors["LF"])
            segs["ghost_bar"] = (anchors["LF"], anchors["RF"])
            return segs

        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.10]
        cam.distance = 0.55
        cam.azimuth = 135
        cam.elevation = -15

        for leg in CLAWS:   # analytic cell
            for i in range(4):
                crab.data.qpos[crab.joint_adr[f"{leg}_J{i}_joint"]] = \
                    q_an_all[leg][i]
        crab.set_ghost_pose(ghost_segments(seg_vis))
        mujoco.mj_forward(crab.model, crab.data)
        cell = crab.render(camera=cam, width=320, height=240)

        for leg in CLAWS:   # IK cell
            for i in range(4):
                crab.data.qpos[crab.joint_adr[f"{leg}_J{i}_joint"]] = \
                    q_prev[leg][i]
        mujoco.mj_forward(crab.model, crab.data)
        cell_ik = crab.render(camera=cam, width=320, height=240)
        cells.append((name.strip(), cell, cell_ik))

    # ---------------------------------------------------------------- out --
    print(f"\n{'pose':<12} {'LF an/ik mm':>16} {'RF an/ik mm':>16} "
          f"{'an clipped J':>12}")
    for row in rows:
        print(f"{row['name']:<12} "
              f"{row['an_LF']*1000:6.0f}/{row['ik_LF']*1000:5.0f}   "
              f"{row['an_RF']*1000:6.0f}/{row['ik_RF']*1000:5.0f}   "
              f"{row['clip_LF'] + row['clip_RF']:>6}/8")
    an_all = np.array([[r["an_LF"], r["an_RF"]] for r in rows]) * 1000
    ik_all = np.array([[r["ik_LF"], r["ik_RF"]] for r in rows]) * 1000
    print(f"\nresidual mm  analytic mean {an_all.mean():5.1f} max "
          f"{an_all.max():5.1f}   IK mean {ik_all.mean():5.1f} max "
          f"{ik_all.max():5.1f}")

    # Contact sheet, 2 columns (analytic | IK) with pose labels via cv2.
    import cv2
    pad, label_h = 6, 22
    sheet = np.full((len(cells) * (240 + pad) + label_h,
                     2 * (320 + pad) + pad, 3), 24, dtype=np.uint8)
    for i, (name, an, ik) in enumerate(cells):
        y = label_h + i * (240 + pad)
        sheet[y - 18:y - 4, pad:pad + 300] = 0
        cv2.putText(sheet, name, (pad, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        sheet[y:y + 240, pad:pad + 320] = cv2.cvtColor(an, cv2.COLOR_RGB2BGR)
        x2 = pad + 320 + pad
        sheet[y:y + 240, x2:x2 + 320] = cv2.cvtColor(ik, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT_PNG), sheet)
    print(f"\nsheet written: {OUT_PNG}")


if __name__ == "__main__":
    main()
