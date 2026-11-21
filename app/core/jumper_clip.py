"""Export a recognised crab dance as a Jumper `jumper.dance` motion clip.

The Jumper official repo's dance task is reference-guided (mjlab's
BeyondMimic-style tracking). Its clip format (see
`tasks/jumper/common/dance/motion.py`) wants, per frame:

    meas_body_x/y/z, meas_body_roll/pitch/yaw    measured base pose (m, rad)
    meas_L{k}_j{j}                                measured joint positions (rad)
    dt, phase

The leg numbering in the file is NOT the robot's joint order: L0=LF,
L1=LM, L2=LR, L3=RR, L4=RM, L5=RF (legs go round the body). The entity
order is LF, RF, LM, RM, LR, RR; the Jumper converter remaps with
`LEG_JOINTS`.

We build the measured channels by posing the app's own MuJoCo model through
the recognised command table (the same pose_dance the preview uses), so the
reference is a kinematically consistent trajectory with planted support feet
and the user's stance / J0 / J1 limits baked in. Turns and jumps are already
in the table (yaw curve + jump hop + foot lifts), so they land directly in
the clip as base-pose / joint targets -- the "control commands in the
training" the dance task expects.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from app.constants import HOME, JUMP_HOP_GAIN, STAND_Z_DOC
from app.core.crab_model import CrabModel

#: Jumper refuses a reference whose joints move faster than its servos' corner
#: speed (30.7 rad/s) at the 50 Hz control rate. The app's leg IK can re-solve a
#: leg between two video frames (a foot re-plant, a jump), which is a step no
#: servo can follow; the export reflects that by slew-rate limiting the measured
#: channels. 26 rad/s leaves margin for the 30 -> 50 Hz resample.
MAX_JOINT_SPEED_RAD_S = 26.0

#: The four legs Jumper's dance task checks for ground contact.
SUPPORT_LEGS = ("LM", "RM", "LR", "RR")


def _limit_speed(values: np.ndarray, dt: float,
                 vmax: float = MAX_JOINT_SPEED_RAD_S) -> np.ndarray:
    """Slew-rate limit one channel to |dq| <= vmax * dt between frames.

    Alternating forward/backward passes project onto the pairwise speed bounds;
    the feasible set is convex and non-empty (a constant channel), so this
    converges, and it keeps the trajectory as close to the original as the
    bound allows instead of merely lagging it. Values stay inside the original
    [min, max], so joint limits are preserved.
    """
    x = np.asarray(values, dtype=np.float64).copy()
    n = len(x)
    if n < 2:
        return x
    lim = float(vmax) * float(dt)
    for _ in range(64):
        changed = False
        prev = x[0]
        for i in range(1, n):
            d = x[i] - prev
            if d > lim:
                x[i] = prev = prev + lim
                changed = True
            elif d < -lim:
                x[i] = prev = prev - lim
                changed = True
            else:
                prev = x[i]
        nxt = x[-1]
        for i in range(n - 2, -1, -1):
            d = x[i] - nxt
            if d > lim:
                x[i] = nxt = nxt + lim
                changed = True
            elif d < -lim:
                x[i] = nxt = nxt - lim
                changed = True
            else:
                nxt = x[i]
        if not changed:
            break
    return x


def _home_site_z(crab: CrabModel) -> dict[str, float]:
    """Support-foot site height at the robot's documented standing pose.

    The app's `z_stand` is measured from the foot *sites*, which sit ~8.7 mm
    above Jumper's `STAND_Z` (there the foot collision meshes are what touch
    the floor). The exported base height is shifted onto `STAND_Z_DOC`, so the
    foot targets have to move up with it or the crab is asked to train 8.7 mm
    below the stance its asset was calibrated in.
    """
    m, d = crab.model, crab.data
    d.qpos[:] = 0.0
    d.qpos[crab.free_adr + 2] = STAND_Z_DOC
    d.qpos[crab.free_adr + 3] = 1.0
    for name, value in HOME.items():
        d.qpos[crab.joint_adr[name]] = value
    mujoco.mj_forward(m, d)
    return {leg: float(d.site(leg).xpos[2]) for leg in SUPPORT_LEGS}


def _plant_leg(crab: CrabModel, leg: str, target, iters: int = 10) -> None:
    """Solve J0/J1/J2 so the foot site reaches `target` (world, metres).

    The planar `CrabModel.leg_ik` holds J0 at its HOME azimuth plus the dance
    stance; when the body pitches or rolls, that fixed leg plane cannot reach
    the floor -- 26 mm out of plane at 20 deg of pitch, which Jumper's dance
    task reads as a reference the robot cannot stand in. The third joint is
    exactly what the planar solve leaves out, so solve all three against the
    site Jacobian (damped), warm-started from the caller's pose and clamped to
    the joint ranges.
    """
    m, d = crab.model, crab.data
    sid = m.site(leg).id
    names = [f"{leg}_J{j}_joint" for j in range(3)]
    q_adr = [crab.joint_adr[n] for n in names]
    dofs = [int(m.jnt_dofadr[m.joint(n).id]) for n in names]
    lo = [crab.joint_range[n][0] for n in names]
    hi = [crab.joint_range[n][1] for n in names]
    target = np.asarray(target, dtype=float)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target - d.site_xpos[sid]
        if float(np.linalg.norm(err)) < 5.0e-5:
            return
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        jac = jacp[:, dofs]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1.0e-6 * np.eye(3), err)
        for a, v, l, h in zip(q_adr, dq, lo, hi):
            d.qpos[a] = float(np.clip(d.qpos[a] + v, l, h))


# Source-file leg order (round the body), matching Jumper's LEG_JOINTS.
_JUMPER_LEGS = (
    ("LF", 5),   # L0
    ("LM", 3),   # L1
    ("LR", 3),   # L2
    ("RR", 3),   # L3
    ("RM", 3),   # L4
    ("RF", 5),   # L5
)


def build_dance_clip(table, crab: CrabModel, out_path: str | Path) -> Path:
    """Write the table as a `jumper.dance` source `.npz`; returns the path.

    `crab` is used only for its kinematics (pose_dance + joint addresses); the
    caller owns it. The current pose is not restored -- callers that care
    should pass a scratch model.
    """
    times = np.asarray(table.times, dtype=float)
    n = len(times)
    if n < 2:
        raise ValueError("the dance is too short to export")
    dt = float(np.median(np.diff(times)))
    zs = crab.z_stand
    z_lo, z_hi = 0.5 * zs, crab.z_max_ik

    claw = table.claw
    foot = table.foot_lift or {}
    # The app stands 8.7 mm lower than Jumper's calibrated stance (its
    # `z_stand` is measured from the foot sites, Jumper's from the foot
    # collision meshes); shift the whole reference onto Jumper's ground.
    z_shift = STAND_Z_DOC - zs
    home_z = _home_site_z(crab)
    zs_out = np.zeros(n, dtype=np.float32)
    rpy = np.zeros((n, 3), dtype=np.float64)
    joint = {leg: np.zeros((n, k), dtype=np.float32)
             for leg, k in _JUMPER_LEGS}

    for i in range(n):
        hop = JUMP_HOP_GAIN * zs * float(table.jump[i])
        # Crouches (negative hop) belong to the stance; an upward hop is the
        # body leaving the ground. Pose the legs at the grounded height, then
        # move the base: solving the leg IK at the hop peak makes it saturate
        # and re-solve, which is a servo-impossible step Jumper's continuity
        # check rejects.
        z_leg = float(np.clip(float(table.base_z[i]) + min(hop, 0.0),
                              z_lo, z_hi))
        rise = max(hop, 0.0)
        z = z_leg + z_shift + rise
        roll = float(table.roll[i])
        pitch = float(table.pitch[i])
        yaw = float(table.yaw[i])
        claw_i = {name: float(arr[i]) for name, arr in claw.items()}
        lift_i = {leg: float(arr[i]) for leg, arr in foot.items()}
        crab.pose_dance(z_leg, roll, pitch, yaw, claw_i,
                        leg_sway=float(table.leg_sway[i]),
                        yaw_offset_deg=0.0,
                        foot_lift=lift_i)
        # Keep the app's horizontal foot placement (stance, sway, far stance
        # for the fronts) but plant the foot where it actually belongs
        # vertically -- raised with a hop, tucked for a leg lift.
        targets = {
            leg: (float(crab.data.site(leg).xpos[0]),
                  float(crab.data.site(leg).xpos[1]),
                  home_z[leg] + rise + lift_i.get(leg, 0.0))
            for leg in SUPPORT_LEGS
        }
        crab.set_body(z, math.radians(roll), math.radians(pitch),
                      math.radians(yaw))
        for leg in SUPPORT_LEGS:
            _plant_leg(crab, leg, targets[leg])
        zs_out[i] = z
        rpy[i] = (np.radians(roll), np.radians(pitch), np.radians(yaw))
        for leg, k in _JUMPER_LEGS:
            for j in range(k):
                name = f"{leg}_J{j}_joint"
                joint[leg][i, j] = float(crab.data.qpos[crab.joint_adr[name]])

    # Foot re-plants and leg-raise transitions can still change the IK branch
    # between neighbouring video frames; make every channel followable.
    for leg, _ in _JUMPER_LEGS:
        for j in range(joint[leg].shape[1]):
            joint[leg][:, j] = _limit_speed(joint[leg][:, j], dt)

    data = {
        "time": (np.arange(n, dtype=np.float64) * dt).astype(np.float32),
        "dt": np.float64(dt),
        "phase": np.ones(n, dtype=np.float32),
        "meas_body_x": np.zeros(n, dtype=np.float32),
        "meas_body_y": np.zeros(n, dtype=np.float32),
        "meas_body_z": zs_out,
        "meas_body_roll": rpy[:, 0].astype(np.float32),
        "meas_body_pitch": rpy[:, 1].astype(np.float32),
        "meas_body_yaw": rpy[:, 2].astype(np.float32),
    }
    # The converter reads the measured channels by default; the commanded
    # duplicates keep the file valid for `REFERENCE="command"` too.
    for prefix in ("", "meas_"):
        if prefix == "":
            data["body_x"] = np.zeros(n, dtype=np.float32)
            data["body_y"] = np.zeros(n, dtype=np.float32)
            data["body_z"] = zs_out.copy()
            data["body_roll"] = data["meas_body_roll"].copy()
            data["body_pitch"] = data["meas_body_pitch"].copy()
            data["body_yaw"] = data["meas_body_yaw"].copy()
        for k, (leg, nj) in enumerate(_JUMPER_LEGS):
            arr = joint[leg]
            for j in range(nj):
                data[f"{prefix}L{k}_j{j}"] = arr[:, j].astype(np.float32)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    return out
