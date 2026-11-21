"""Reference-tracking trainer prototype — REAL MuJoCo physics rollout (CPU ES).

The reference track (recognised joints, jumps included) is tracked by a PD
controller in the actual MuJoCo dynamics: gravity, floor friction, claw
self-collision contacts. Terminations (base height error / body tilt) end a
rollout like the anti-fall backstop. The reward is

    0.40 tracking + 0.25 stability (height/tilt) + 0.15 hand separation
    (+ LF-RF contact penalty) + 0.10 smoothness + 0.10 survival

and a small residual policy is trained with an elitist evolution strategy
(1+lambda): per-group PD gain scales, gated upper-body claw residuals,
a claw velocity lead and a jump feed-forward. Everything shown in the
curves/log (return, components, terminations) comes from the physics sim.

The weights + the reference NPZ are exported at the end.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import QThread, Signal

from app.constants import (
    CACHE_DIR, CLAW_J1_MIN_DEG, CRAB_FRONT_YAW_OFFSET_DEG, JUMP_HOP_GAIN,
    TRAIN_CLAW_J0_FWD_DEG, TRAIN_CLAW_J0_SAFE_DEG, TRAIN_STANCE_PITCH_DEG,
)
from app.core.crab_model import CrabModel
from app.core.retarget import sample_commands

CLAW_NAMES = tuple(
    f"{leg}_J{i}_joint" for leg in ("LF", "RF") for i in range(5))

# Base PD gains, tuned on this model (total mass ~2.66 kg, tiny inertias).
KP_LEG_BASE = 2.5
KP_CLAW_BASE = 0.6
KD_BASE = 0.10
SIM_DT = 0.01               # physics timestep inside a sample interval
JUMP_FORCE_SCALE = 20.0     # N per unit of the learned jump feed-forward

# Reward weights (sum = 1 -> normalised return in 0..1). The user asked
# for a stronger rear-foot-air penalty, a stronger body-YAW penalty and a
# post-jump LANDING posture reward (better posture -> higher reward).
W_TRACK = 0.28
W_STAB = 0.14
W_HANDS = 0.08
W_SMOOTH = 0.05
W_SURVIVE = 0.10
W_YAW = 0.15
W_REAR = 0.12
W_LAND = 0.08
LAND_FRAMES = 6              # post-landing posture window (sampled frames)

# Policy layout.
I_KP_LEG, I_KP_CLAW, I_KD = 0, 1, 2
I_RESIDUAL = 3
I_CLAW_LEAD = I_RESIDUAL + len(CLAW_NAMES)
I_JUMP_FF = I_CLAW_LEAD + 1
POLICY_DIM = I_JUMP_FF + 1
POLICY_LAYOUT = (["kp_leg_scale", "kp_claw_scale", "kd_scale"]
                 + list(CLAW_NAMES) + ["claw_velocity_lead",
                                       "jump_feedforward"])

_SPEED_CACHE: dict[str, float] = {}


@dataclass
class TrainParams:
    """Trainable settings shown in the parameter dialog."""
    # Which training method the page's start button uses:
    #   "jumper_ppo" -- Jumper/mjlab rsl_rl PPO reference tracking (available)
    #   "supermimic" -- allmimic's world-model method (G1 only; the crab port
    #                   is not done, so the dialog keeps it disabled)
    #   "es"         -- local CPU MuJoCo + evolution strategy trial
    method: str = "jumper_ppo"
    # Jumper PPO fine-tune config. The slow base was trained offline and is
    # bundled at assets/pretrained/dance_base.pt; every app run starts there.
    # On the 8 GB 4060, 2048 envs measure ~1.6 s/iter, so the default 300
    # extra iterations take about 8 minutes.
    sm_num_envs: int = 2048
    sm_batch_size: int = 256      # inherited default; rsl_rl has no batch_size
    sm_iterations: int = 300
    iterations: int = 24          # 本机试跑轮数
    population: int = 8           # 本机试跑数量 (candidates per epoch)
    elites: int = 3
    sigma: float = 0.18
    precision_deg: float = 8.0    # 训练动作精度 (tracking tolerance)
    stride: int = 3
    max_frames: int = 300
    render: bool = True
    seed: int = 7
    hand_safe_m: float = 0.035
    height_term_m: float = 0.060
    tilt_term_deg: float = 55.0   # terminates real falls, not dance leans
    substeps: int = 10
    device: str = "auto"
    # Reward terms (PPO and ES sets) from the reward-weight dialog; empty =
    # backend defaults. A saved profile name goes into the run manifest.
    reward_weights: dict = field(default_factory=dict)
    reward_profile: str = ""
    show_weight_versions: bool = False
    curriculum: bool = False


def cached_frame_seconds(substeps: int = 10) -> float | None:
    """Previously measured per-frame cost, or None (never forces a build)."""
    return _SPEED_CACHE.get(f"frame_{substeps}")


def benchmark_frame_seconds(substeps: int = 10) -> float:
    """Measured physics cost of one reference frame (cached, machine-local)."""
    key = f"frame_{substeps}"
    if key in _SPEED_CACHE:
        return _SPEED_CACHE[key]
    crab = CrabModel()
    m, d = crab.model, crab.data
    m.opt.timestep = SIM_DT
    mujoco.mj_resetData(m, d)
    crab.set_home_joints()
    crab.set_body(crab.z_stand, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    qadr = np.arange(7, m.nq)
    dadr = np.arange(6, m.nv)
    ref = d.qpos[qadr].copy()
    steps = 150
    t0 = time.perf_counter()
    for _ in range(steps):
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[dadr] = KP_LEG_BASE * (ref - d.qpos[qadr]) - KD_BASE * d.qvel[dadr]
        mujoco.mj_step(m, d)
    dt = (time.perf_counter() - t0) / steps
    per_frame = dt * substeps + 8e-5   # + reward bookkeeping
    _SPEED_CACHE[key] = per_frame
    crab.close()
    return per_frame


class TrainWorker(QThread):
    """Physics training loop off the GUI thread."""

    progress = Signal(int, int)          # epoch (1-based), total
    status = Signal(str)                 # log line
    metrics = Signal(dict)               # per-epoch history
    physics_frame = Signal(object)       # QImage: physics training render
    preview_frame = Signal(object)       # QImage: static policy preview
    dance_ready = Signal(object)         # {times, qpos, terminated}: full physics dance
    reference_saved = Signal(str)
    weights_saved = Signal(str)

    def __init__(self, table, meta: dict, params: TrainParams, parent=None):
        super().__init__(parent)
        self.table = table
        self.meta = dict(meta or {})
        self.params = params
        self.weights_path: str | None = None
        self.reference_path: str | None = None
        self.history: list[dict] = []
        self._lf_rf_geoms: list[tuple[int, bool]] | None = None
        self._cam = (135.0, -16.0, 0.60)   # azimuth, elevation, distance
        # Editable reward weights (the reward dialog's ES tab).
        defaults = {
            "track": W_TRACK, "stab": W_STAB, "hands": W_HANDS,
            "smooth": W_SMOOTH, "survive": W_SURVIVE, "yaw": W_YAW,
            "rear": W_REAR, "land": W_LAND,
        }
        custom = (getattr(params, "reward_weights", None) or {}).get("es", {})
        self.rw = {**defaults, **{k: float(v) for k, v in custom.items()
                                   if k in defaults}}
        self._rw_sum = sum(self.rw.values()) or 1.0

    def set_camera(self, az: float, el: float, dist: float) -> None:
        """Camera for the physics training renders (mouse orbit/zoom)."""
        self._cam = (float(az), float(el), float(dist))

    # ------------------------------------------------------------ helpers ----
    def _jump_segments(self) -> list[tuple[float, float]]:
        mask = getattr(self.table, "jump_mask", None)
        times = self.table.times
        if mask is None or not len(mask):
            return []
        segs, start = [], None
        for i, on in enumerate(mask):
            if on and start is None:
                start = i
            elif not on and start is not None:
                segs.append((float(times[start]), float(times[i - 1])))
                start = None
        if start is not None:
            segs.append((float(times[start]), float(times[-1])))
        return segs

    def _build_reference(self, crab: CrabModel, stride: int | None = None,
                         cap_frames: int | None = None):
        p = self.params
        n = len(self.table.times)
        # Cover the WHOLE clip (jumps anywhere) by raising the stride
        # until the frame cap fits.
        cap = int(cap_frames if cap_frames is not None else p.max_frames)
        base_stride = int(stride if stride is not None else p.stride)
        stride = max(1, base_stride, int(math.ceil(n / max(1, cap))))
        idx = list(range(0, n, stride))
        times = np.asarray(self.table.times)[idx] if idx else np.zeros(0)
        m = crab.model
        qpos_ref = np.zeros((m.nq, len(idx)))
        jump_val_ref = np.zeros(len(idx))
        jump_ref = np.zeros(len(idx), dtype=bool)
        hand_ref = np.zeros(len(idx))
        sway_ref = np.zeros(len(idx))
        foot_ref: list[dict] = []
        for i, k in enumerate(idx):
            cmd = sample_commands(self.table, float(times[i]), jump=True)
            hop = JUMP_HOP_GAIN * crab.z_stand * float(cmd["jump"])
            z = float(np.clip(cmd["base_z"] + min(hop, 0.0),
                              0.5 * crab.z_stand, crab.z_max_ik))
            z += max(hop, 0.0)
            claw = {name: float(cmd["claw"][name]) for name in CLAW_NAMES}
            # User stance: nose-up pitch + front ground legs shifted forward,
            # so the claws ride clear of the deck during training.
            crab.pose_dance(
                z, cmd["roll"], float(cmd["pitch"]) + TRAIN_STANCE_PITCH_DEG,
                cmd["yaw"], claw,
                leg_sway=cmd["leg_sway"],
                yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG,
                foot_lift=cmd["foot_lift"])
            # Claw J0 swings FORWARD (user); 50 deg clears the leg links.
            for leg, sign in (("LF", -1.0), ("RF", 1.0)):
                adr = crab.joint_adr[f"{leg}_J0_joint"]
                crab.data.qpos[adr] += sign * math.radians(TRAIN_CLAW_J0_FWD_DEG)
            # Keep the swung-forward arms inside the collision-free window.
            for leg in ("LF", "RF"):
                lo, hi = TRAIN_CLAW_J0_SAFE_DEG[leg]
                adr = crab.joint_adr[f"{leg}_J0_joint"]
                crab.data.qpos[adr] = np.clip(
                    crab.data.qpos[adr], math.radians(lo), math.radians(hi))
            mujoco.mj_forward(crab.model, crab.data)
            qpos_ref[:, i] = crab.data.qpos.copy()
            jump_val_ref[i] = float(cmd["jump"])
            jump_ref[i] = bool(cmd["jump"] > 0.5)
            hand_ref[i] = float(np.linalg.norm(
                np.asarray(crab.data.site("LF").xpos)
                - np.asarray(crab.data.site("RF").xpos)))
            sway_ref[i] = float(cmd["leg_sway"])
            foot_ref.append(dict(cmd["foot_lift"] or {}))
        dt_ref = (float(np.median(np.diff(times)))
                  if len(times) > 1 else 0.1)
        return {
            "times": times, "qpos": qpos_ref, "jump_val": jump_val_ref,
            "jump_ref": jump_ref, "hand_ref": hand_ref, "sway": sway_ref,
            "foot": foot_ref, "dt": dt_ref,
        }

    def _save_reference(self, crab: CrabModel, ref: dict) -> str:
        stem = str(self.meta.get("video_stem") or "motion")[:40]
        tag = str(self.meta.get("tag") or "ref")[:16]
        out_dir = CACHE_DIR / "supermimic"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}_{tag}_ref.npz"
        dt = np.median(np.diff(ref["times"])) if len(ref["times"]) > 1 else 0.1
        joint_names = [crab.model.joint(i).name for i in range(crab.model.njnt)]
        np.savez_compressed(
            path,
            times=ref["times"],
            qpos=ref["qpos"],
            joint_names=np.array(joint_names, dtype=object),
            jump_mask=ref["jump_ref"],
            jump_segments=np.array(self._jump_segments(), dtype=float).reshape(-1, 2),
            fps=float(1.0 / max(dt, 1e-6)),
            source_video=str(self.meta.get("video", "")),
            person=str(self.meta.get("person", "")),
            note="reference joint data (app format)",
            supermimic_config=json.dumps({
                "num_envs": self.params.sm_num_envs,
                "batch_size": self.params.sm_batch_size,
                "iterations": self.params.sm_iterations,
            }),
        )
        return str(path)

    # ------------------------------------------------------- physics loop ----
    def _prepare_model(self, crab: CrabModel):
        m = crab.model
        m.opt.timestep = SIM_DT
        hinge_ids = [j for j in range(m.njnt)
                     if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        hinge_names = [m.joint(j).name for j in hinge_ids]
        qadr = np.array([int(m.jnt_qposadr[j]) for j in hinge_ids])
        dadr = np.array([int(m.jnt_dofadr[j]) for j in hinge_ids])
        name_to_pos = {n: i for i, n in enumerate(hinge_names)}
        self._claw_pos = [name_to_pos[n] for n in CLAW_NAMES
                          if n in name_to_pos]
        # Training targets keep the claw forearm above the user's -110 deg.
        claw_lo = np.array([m.jnt_range[j, 0] for j in hinge_ids],
                           dtype=float)[self._claw_pos].copy()
        claw_hi = np.array([m.jnt_range[j, 1] for j in hinge_ids],
                           dtype=float)[self._claw_pos].copy()
        for k, name in enumerate(CLAW_NAMES):
            if name.endswith("_J1_joint"):
                claw_lo[k] = max(claw_lo[k], math.radians(CLAW_J1_MIN_DEG))
        self._claw_lo, self._claw_hi = claw_lo, claw_hi
        self._leg_mask = np.array(
            [n.startswith(("LM_", "RM_", "LR_", "RR_")) for n in hinge_names])
        self._qadr, self._dadr = qadr, dadr
        self._lo = np.array([m.jnt_range[j, 0] for j in hinge_ids])
        self._hi = np.array([m.jnt_range[j, 1] for j in hinge_ids])
        self._floor_id = m.geom("dance_floor").id
        self._rear_geoms = {
            leg: {g for g in range(m.ngeom)
                  if (m.geom(g).name or "").startswith(leg + "_")}
            for leg in ("LR", "RR")}
        if self._lf_rf_geoms is None:
            g = []
            for i in range(m.ngeom):
                name = m.geom(i).name or ""
                if name.startswith("LF_"):
                    g.append((i, True))
                elif name.startswith("RF_"):
                    g.append((i, False))
            self._lf_rf_geoms = g

    def _hand_contacts(self, d) -> tuple[int, float]:
        """(number of LF<->RF contacts, worst penetration in m)."""
        n_contact, pen = 0, 0.0
        if self._lf_rf_geoms is None:
            return 0, 0.0
        side = {i: is_lf for i, is_lf in self._lf_rf_geoms}
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in side and g2 in side and side[g1] != side[g2]:
                n_contact += 1
                if c.dist < 0:
                    pen = max(pen, -float(c.dist))
        return n_contact, pen

    def _rear_grounded(self, d) -> tuple[bool, bool]:
        """Whether the two REAR feet (LR/RR) touch the floor."""
        on = [False, False]
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            for i, leg in enumerate(("LR", "RR")):
                if on[i]:
                    continue
                if ((g1 in self._rear_geoms[leg] and g2 == self._floor_id)
                        or (g2 in self._rear_geoms[leg]
                            and g1 == self._floor_id)):
                    on[i] = True
        return on[0], on[1]

    @staticmethod
    def _yaw_deg(q) -> float:
        w, x, y, z = q
        return math.degrees(math.atan2(2 * (w * z + x * y),
                                       1 - 2 * (y * y + z * z)))

    @staticmethod
    def _wrap_deg(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0

    def _tilt_deg(self, quat_a: np.ndarray, quat_b: np.ndarray) -> float:
        """Angle between the two bodies' up axes (deg)."""
        def up(q):
            w, x, y, z = q
            return np.array([2 * (x * z + w * y),
                             2 * (y * z - w * x),
                             1 - 2 * (x * x + y * y)])
        a, b = up(quat_a), up(quat_b)
        a /= max(np.linalg.norm(a), 1e-9)
        b /= max(np.linalg.norm(b), 1e-9)
        return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))

    def _evaluate(self, crab: CrabModel, ref: dict, policy: np.ndarray,
                  stop_frame: int | None = None,
                  capture_frame: int | None = None):
        m, d = crab.model, crab.data
        p = self.params
        qpos_ref = ref["qpos"]
        n = qpos_ref.shape[1]
        if n < 2:
            return 0.0, np.zeros(5), 0.0, True
        kp_leg = KP_LEG_BASE * float(np.clip(1.0 + policy[I_KP_LEG], 0.3, 3.0))
        kp_claw = KP_CLAW_BASE * float(np.clip(1.0 + policy[I_KP_CLAW], 0.3, 3.0))
        kd = KD_BASE * float(np.clip(1.0 + policy[I_KD], 0.3, 3.0))
        kp = np.where(self._leg_mask, kp_leg, kp_claw)
        safe = max(p.hand_safe_m, 1e-6)
        prec = max(p.precision_deg, 0.5)
        qadr, dadr = self._qadr, self._dadr
        lo, hi = self._lo, self._hi
        claw_pos = self._claw_pos
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos_ref[:, 0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        total = 0.0
        comps = np.zeros(8)
        steps = 0
        terminated = False
        land_remaining = 0
        land_values: list[float] = []
        prev_jump = float(ref["jump_val"][0]) if n > 0 else 0.0
        substeps = max(2, min(int(p.substeps),
                              int(round(float(ref.get("dt", 0.1)) / SIM_DT))))
        for i in range(1, n):
            prev, cur = qpos_ref[:, i - 1], qpos_ref[:, i]
            # Reference claw residual/lead for this frame (gated by how
            # close the recorded hands are: upper-body residual action).
            gate = min(1.0, max(0.0, (safe - float(ref["hand_ref"][i])) / safe))
            lead = policy[I_CLAW_LEAD] * (cur[qadr][claw_pos]
                                          - prev[qadr][claw_pos])
            add = policy[I_RESIDUAL:I_CLAW_LEAD] * gate + lead
            jt = float(ref["jump_val"][i])
            if prev_jump > 0.30 and jt <= 0.15:
                land_remaining = LAND_FRAMES   # touchdown detected
            prev_jump = jt
            for s in range(1, substeps + 1):
                a = s / substeps
                tgt = (1 - a) * prev[qadr] + a * cur[qadr]
                tgt[claw_pos] = np.clip(tgt[claw_pos] + add,
                                        self._claw_lo, self._claw_hi)
                d.qfrc_applied[:] = 0.0
                d.qfrc_applied[dadr] = kp * (tgt - d.qpos[qadr]) - kd * d.qvel[dadr]
                if jt > 0.0:
                    # Learned jump assist: vertical force on the base during
                    # detected jumps (the policy's jump feed-forward).
                    d.qfrc_applied[2] += (policy[I_JUMP_FF] * jt
                                          * JUMP_FORCE_SCALE)
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if capture_frame is not None and i == capture_frame:
                try:
                    self.physics_frame.emit(self._render_state(crab))
                except Exception:  # noqa: BLE001 - preview best effort
                    pass
            if not np.isfinite(d.qpos).all():
                terminated = True
                break
            err = np.degrees(np.abs(d.qpos[qadr] - cur[qadr]))
            r_track = math.exp(-float(np.mean(err) / prec) ** 2)
            z_err = abs(float(d.qpos[2]) - float(cur[2]))
            tilt = self._tilt_deg(d.qpos[3:7], cur[3:7])
            r_stab = math.exp(-(z_err / 0.03) ** 2 - (tilt / 15.0) ** 2)
            hand_d = float(np.linalg.norm(
                np.asarray(d.site("LF").xpos) - np.asarray(d.site("RF").xpos)))
            n_contact, penetration = self._hand_contacts(d)
            overlap = max(0.0, safe - hand_d)
            contact_pen = min(1.0, n_contact / 3.0 + penetration / 0.005)
            r_hands = max(0.0, 1.0 - (overlap / safe) ** 2 - 0.5 * contact_pen)
            r_smooth = math.exp(-float(np.mean(d.qvel[dadr] ** 2)) / 4.0)
            # Body-yaw tracking (user: stronger turn/yaw penalty).
            yaw_err = abs(self._wrap_deg(
                self._yaw_deg(d.qpos[3:7]) - self._yaw_deg(cur[3:7])))
            r_yaw = math.exp(-(yaw_err / 12.0) ** 2)
            # REAR feet must stay on the floor (exempt planned jumps and
            # reference foot-lift gestures).
            rear_l, rear_r = self._rear_grounded(d)
            foot_i = ref["foot"][i] if i < len(ref["foot"]) else {}
            lift_l = float(foot_i.get("LR", 0.0))
            lift_r = float(foot_i.get("RR", 0.0))
            ok_l = jt > 0.2 or rear_l or lift_l > 0.01
            ok_r = jt > 0.2 or rear_r or lift_r > 0.01
            r_rear = 0.5 * (float(ok_l) + float(ok_r))
            # Landing posture reward: upright, at the reference height and
            # with both rear feet planted -> higher reward.
            if land_remaining > 0:
                r_land = (math.exp(-(tilt / 12.0) ** 2
                                   - (z_err / 0.02) ** 2)
                          * (0.4 + 0.6 * r_rear))
                land_values.append(r_land)
                land_remaining -= 1
            else:
                r_land = 0.0
            terminated = (z_err > p.height_term_m
                          or tilt > p.tilt_term_deg)
            r_surv = 0.0 if terminated else 1.0
            total += (self.rw["track"] * r_track
                      + self.rw["stab"] * r_stab
                      + self.rw["hands"] * r_hands
                      + self.rw["smooth"] * r_smooth
                      + self.rw["survive"] * r_surv
                      + self.rw["yaw"] * r_yaw
                      + self.rw["rear"] * r_rear
                      + self.rw["land"] * r_land) / self._rw_sum
            comps += (r_track, r_stab, r_hands, r_smooth, r_surv,
                      r_yaw, r_rear, r_land)
            steps += 1
            if stop_frame is not None and i >= stop_frame:
                break
            if terminated:
                break
        comps /= max(steps, 1)
        # Landing curve = average post-touchdown posture score (0 if no
        # jump landed in this rollout), so "better posture -> higher".
        comps[7] = float(np.mean(land_values)) if land_values else 0.0
        return total / max(n - 1, 1), comps, steps / max(n - 1, 1), terminated

    # ------------------------------------------------------------ renders ----
    def _camera(self) -> "mujoco.MjvCamera":
        az, el, dist = self._cam
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.10]
        cam.distance = float(dist)
        cam.azimuth = float(az)
        cam.elevation = float(el)
        return cam

    def _render_state(self, crab: CrabModel, width: int = 430,
                      height: int = 300):
        """QImage of the CURRENT physics state (contacts/friction visible)."""
        from PySide6.QtGui import QImage
        rgb = crab.render(camera=self._camera(), width=width, height=height)
        return QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                      QImage.Format.Format_RGB888).copy()

    def _record_dance(self, crab: CrabModel, policy: np.ndarray):
        """Run the trained policy over the WHOLE dance in physics and
        record every simulated state for playback."""
        n = len(self.table.times)
        stride = 1 if n <= 2400 else 2
        ref = self._build_reference(crab, stride=stride, cap_frames=10 ** 9)
        m, d = crab.model, crab.data
        p = self.params
        qpos_ref = ref["qpos"]
        nf = qpos_ref.shape[1]
        kp_leg = KP_LEG_BASE * float(np.clip(1.0 + policy[I_KP_LEG], 0.3, 3.0))
        kp_claw = KP_CLAW_BASE * float(np.clip(1.0 + policy[I_KP_CLAW], 0.3, 3.0))
        kd = KD_BASE * float(np.clip(1.0 + policy[I_KD], 0.3, 3.0))
        kp = np.where(self._leg_mask, kp_leg, kp_claw)
        safe = max(p.hand_safe_m, 1e-6)
        qadr, dadr = self._qadr, self._dadr
        lo, hi = self._lo, self._hi
        claw_pos = self._claw_pos
        substeps = max(2, min(int(p.substeps),
                              int(round(float(ref.get("dt", 0.1)) / SIM_DT))))
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos_ref[:, 0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        records = [d.qpos.copy()]
        terminated = False
        for i in range(1, nf):
            if self.isInterruptionRequested():
                break
            prev, cur = qpos_ref[:, i - 1], qpos_ref[:, i]
            gate = min(1.0, max(0.0, (safe - float(ref["hand_ref"][i])) / safe))
            lead = policy[I_CLAW_LEAD] * (cur[qadr][claw_pos]
                                          - prev[qadr][claw_pos])
            add = policy[I_RESIDUAL:I_CLAW_LEAD] * gate + lead
            jt = float(ref["jump_val"][i])
            for s in range(1, substeps + 1):
                a = s / substeps
                tgt = (1 - a) * prev[qadr] + a * cur[qadr]
                tgt[claw_pos] = np.clip(tgt[claw_pos] + add,
                                        self._claw_lo, self._claw_hi)
                d.qfrc_applied[:] = 0.0
                d.qfrc_applied[dadr] = kp * (tgt - d.qpos[qadr]) - kd * d.qvel[dadr]
                if jt > 0.0:
                    d.qfrc_applied[2] += policy[I_JUMP_FF] * jt * JUMP_FORCE_SCALE
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if not np.isfinite(d.qpos).all():
                terminated = True
                break
            z_err = abs(float(d.qpos[2]) - float(cur[2]))
            tilt = self._tilt_deg(d.qpos[3:7], cur[3:7])
            records.append(d.qpos.copy())
            if z_err > p.height_term_m or tilt > p.tilt_term_deg:
                terminated = True
                break
        times = np.asarray(ref["times"])[:len(records)]
        return {"times": times, "qpos": np.stack(records, axis=1),
                "terminated": bool(terminated)}

    def _render_preview(self, crab: CrabModel, ref: dict, policy: np.ndarray):
        """Static kinematic pose of the current policy (separate pane)."""
        from PySide6.QtGui import QImage
        n = ref["qpos"].shape[1]
        i = n // 2
        safe = max(self.params.hand_safe_m, 1e-6)
        gate = min(1.0, max(0.0, (safe - float(ref["hand_ref"][i])) / safe))
        q = ref["qpos"][:, i].copy()
        add = policy[I_RESIDUAL:I_CLAW_LEAD] * gate
        for k, pos in enumerate(self._claw_pos):
            q[self._qadr[pos]] += add[k]
        crab.data.qpos[:] = q
        mujoco.mj_forward(crab.model, crab.data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.10]
        cam.distance = 0.60
        cam.azimuth = 135.0
        cam.elevation = -16.0
        rgb = crab.render(camera=cam, width=470, height=330)
        return QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                      QImage.Format.Format_RGB888).copy()

    # ------------------------------------------------------------ training ----
    def run(self) -> None:  # noqa: C901
        try:
            crab = CrabModel()
            crab.claw_geometry
            self._prepare_model(crab)
            ref = self._build_reference(crab)
            if ref["qpos"].shape[1] < 2:
                self.status.emit("参考动作太短，无法训练")
                return
            self.reference_path = self._save_reference(crab, ref)
            self.reference_saved.emit(self.reference_path)

            p = self.params
            rng = np.random.default_rng(p.seed)
            pop = max(int(p.population), int(p.elites) + 1, 2)
            best = np.zeros(POLICY_DIM)
            sigma_vec = np.array(
                [0.25, 0.25, 0.25]
                + [p.sigma] * len(CLAW_NAMES)
                + [0.30, 0.50])
            best_ret = None
            best_comps = np.zeros(5)
            best_alive = 0.0
            segs = self._jump_segments()
            n_frames = ref["qpos"].shape[1]
            self.status.emit(
                f"参考轨道 {n_frames} 帧 · 跳跃段 {len(segs)} 个")
            self.status.emit(
                f"GPU 配置 · num_envs={p.sm_num_envs} · "
                f"batch={p.sm_batch_size} · 轮数={p.sm_iterations}"
                f" · 已写入参考 NPZ")
            self.status.emit(
                f"Smoke：精度 {p.precision_deg:.0f}° · 数量 {pop} · "
                f"轮数 {p.iterations} · 终止 {p.height_term_m*1000:.0f}mm/"
                f"{p.tilt_term_deg:.0f}° · 双手安全 {p.hand_safe_m*1000:.0f}mm")
            self.status.emit(
                "物理仿真：MuJoCo 动力学 + 地面摩擦 + 钳自碰撞；"
                "MuJoCo 物理在 CPU · 渲染走显卡")
            t_start = time.perf_counter()
            last_render = 0.0
            for it in range(1, max(int(p.iterations), 1) + 1):
                if self.isInterruptionRequested():
                    self.status.emit("训练已停止")
                    return
                scale = max(0.35, 1.0 - 0.6 * (it - 1) / max(p.iterations, 1))
                cands = [best] + [
                    best + sigma_vec * scale * rng.standard_normal(POLICY_DIM)
                    for _ in range(pop - 1)]
                results = []
                cap = (n_frames // 2) if (
                    p.render and time.perf_counter() - last_render >= 0.25
                ) else None
                for k, cand in enumerate(cands):
                    if self.isInterruptionRequested():
                        self.status.emit("训练已停止")
                        return
                    results.append(self._evaluate(
                        crab, ref, cand, capture_frame=cap if k == 0 else None))
                if cap is not None:
                    last_render = time.perf_counter()
                order = int(np.argmax([r[0] for r in results]))
                if best_ret is not None and best_ret >= results[order][0]:
                    ret_best = best_ret
                    comps = best_comps
                    alive = best_alive
                else:
                    best = cands[order]
                    ret_best = results[order][0]
                    best_ret = ret_best
                    comps = results[order][1]
                    best_comps = comps
                    alive = results[order][2]
                    best_alive = alive
                mean_ret = float(np.mean([r[0] for r in results]))
                term_frac = float(np.mean([r[3] for r in results]))
                rec = {
                    "iter": it,
                    "return_best": float(ret_best),
                    "return_mean": mean_ret,
                    "track": float(comps[0]), "stab": float(comps[1]),
                    "hands": float(comps[2]), "smooth": float(comps[3]),
                    "survive": float(comps[4]), "term_frac": term_frac,
                    "alive": float(alive), "yaw": float(comps[5]),
                    "rear": float(comps[6]), "land": float(comps[7]),
                }
                self.history.append(rec)
                self.metrics.emit(rec)
                self.progress.emit(it, int(p.iterations))
                self.status.emit(
                    f"epoch {it:03d}/{p.iterations}  return={ret_best:.4f}  "
                    f"track={comps[0]:.3f} stab={comps[1]:.3f} "
                    f"hands={comps[2]:.3f} smooth={comps[3]:.3f} "
                    f"survive={comps[4]:.3f} yaw={comps[5]:.3f} "
                    f"rear={comps[6]:.3f} land={comps[7]:.3f}  "
                    f"term={term_frac*100:.0f}%  "
                    f"alive={alive*100:.0f}%  "
                    f"t={time.perf_counter()-t_start:.1f}s")
                if p.render and (it % 2 == 1 or it == p.iterations):
                    try:
                        self.preview_frame.emit(self._render_preview(crab, ref, best))
                    except Exception as exc:  # noqa: BLE001 - preview best effort
                        self.status.emit(f"渲染失败：{type(exc).__name__}: {exc}")

            # ---- full-dance physics playback with the trained policy ----
            self.status.emit("生成完整舞蹈回放…")
            dance = self._record_dance(crab, best)
            if dance["terminated"]:
                self.status.emit(
                    f"回放：策略在 {len(dance['times'])} 帧后触发终止条件"
)
            else:
                self.status.emit(
                    f"回放：完整舞蹈 {len(dance['times'])} 帧 · 物理仿真")
            self.dance_ready.emit(dance)
            self.weights_path = self._save_weights(best, ref, dance)
            self.weights_saved.emit(self.weights_path)
            self.status.emit(f"训练完成 · 权重已写入 {self.weights_path}")
            crab.close()
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            import traceback
            self.status.emit(f"训练失败: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())

    def _save_weights(self, policy: np.ndarray, ref: dict,
                      dance: dict | None = None) -> str:
        stem = str(self.meta.get("video_stem") or "motion")[:40]
        tag = str(self.meta.get("tag") or "policy")[:16]
        out_dir = CACHE_DIR / "train"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}_{tag}_policy.npz"
        np.savez_compressed(
            path,
            policy=policy,
            policy_layout=np.array(POLICY_LAYOUT, dtype=object),
            config=json.dumps(asdict(self.params)),
            history=json.dumps(self.history),
            reference=self.reference_path or "",
            source_video=str(self.meta.get("video", "")),
            person=str(self.meta.get("person", "")),
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            frames=int(ref["times"].size),
            physics="MuJoCo dynamics, contacts + friction + claw self-collision",
            dance_times=(dance["times"] if dance else np.zeros(0)),
            dance_qpos=(dance["qpos"] if dance else np.zeros((0, 0))),
            dance_terminated=(dance["terminated"] if dance else False),
        )
        return str(path)
