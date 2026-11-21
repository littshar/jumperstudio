"""MuJoCo wrapper for the jumper crab robot.

Kinematic puppet: joint angles are written straight into qpos, mj_forward is
called, and the scene is rendered offscreen. No dynamics, no actuators, no
physics -- the goal is a stable, always-upright dance preview.

The four walking legs are solved with a closed-form 2-link IK (J1/J2 share a
parallel axis, J0 is held at HOME), so the body can bob up and down while the
feet stay planted under the hips. The two claws are pure command passthrough.

All geometry is *calibrated* from the loaded model at HOME via mj_forward --
nothing is copied from CAD numbers, so a model revision cannot silently break
the maths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - hard dependency
    raise SystemExit("mujoco is required: pip install mujoco") from exc

from app.constants import CLAWS, FRONT_LEG_J0_STANCE_DEG, HOME, JUMPER_XML, LEGS

_EPS = 1e-9


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) as a MuJoCo (w, x, y, z) quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


@dataclass
class LegCalib:
    """Planar 2-link geometry of one walking leg, measured at HOME.

    The leg plane has orthonormal basis (t1, t2) with normal n (the joint
    axis shared by J1 and J2). Basis chosen so the HOME thigh direction is
    t1, hence azimuth_home(J1) = 0 and azimuth_home(calf) = beta_home.

    axis1 / axis2 are the empirical signs mapping joint deltas onto that
    plane: the model's J1/J2 axes may point along -n (the left legs are
    mirrored), in which case positive q rotates the leg the other way.
    """
    name: str
    p1: np.ndarray        # J1 anchor in base frame
    a: float              # thigh length (J1 -> J2)
    b: float              # calf length (J2 -> foot site)
    n: np.ndarray         # plane normal (joint axis, unit)
    t1: np.ndarray        # HOME thigh direction (unit)
    t2: np.ndarray        # n x t1
    beta_home: float      # signed bend (calf azimuth - thigh azimuth) at HOME
    axis1: float          # +1/-1: sign of J1 rotation about n
    axis2: float          # +1/-1: sign of J2 rotation about n (vs. thigh)
    q0_home: float
    q1_home: float
    q2_home: float
    foot_xy: np.ndarray   # foot site xy in base frame at HOME
    q1_range: tuple[float, float]
    q2_range: tuple[float, float]


class CrabModel:
    """Loads jumper + floor + lights; calibrates legs; poses and renders."""

    def __init__(self, xml_path=None, render_width: int = 960,
                 render_height: int = 720, floor_texture=None):
        xml = str(xml_path or JUMPER_XML)
        spec = mujoco.MjSpec.from_file(xml)
        self._add_environment(spec, floor_texture=floor_texture)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.free_adr = int(self.model.joint("floating_base").qposadr[0])
        self.joint_adr = {
            name: int(self.model.joint(name).qposadr[0]) for name in HOME
        }
        self.joint_range = self._usable_ranges()
        self._ghost_bodies = tuple(
            f"ghost_{part}" for part in ("lu", "lf", "ru", "rf", "bar", "head"))

        self._calibrate_home()
        self.calib = {leg: self._calibrate_leg(leg) for leg in LEGS}
        self.z_max_ik = self._max_ik_height()
        self._claw_geom: dict | None = None
        self._claw_ik_ctx: dict | None = None

        self._floor_default = np.array(self.model.geom("dance_floor").rgba)

        self._renderer = None
        self._render_width = render_width
        self._render_height = render_height

    # ------------------------------------------------------------ setup ----
    @staticmethod
    def _add_environment(spec: "mujoco.MjSpec", floor_texture=None) -> None:
        world = spec.worldbody
        # Optional user-imported floor image as a 2D texture (PNG path; the
        # caller converts other formats). Without it: plain tinted plane.
        if floor_texture is not None:
            spec.add_texture(name="user_floor",
                             type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=str(floor_texture))
            # The texture list is indexed by material slot, and slot 0 is
            # *emission*; the diffuse slot (`texture=` in MJCF) is slot 1.
            # Passing ["user_floor"] painted the floor emissive but left the
            # diffuse white, so the plane rendered pure white under the lights.
            # Low specular keeps a photo floor from washing out at a low camera
            # angle.
            spec.add_material(name="user_floor_mat",
                              textures=["", "user_floor"], texuniform=1,
                              texrepeat=[4, 4],
                              specular=0.05, shininess=0.1, reflectance=0.0)
            world.add_geom(
                name="dance_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[2, 2, 0.05], material="user_floor_mat", rgba=[1, 1, 1, 1],
            )
        else:
            world.add_geom(
                name="dance_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[2, 2, 0.05], rgba=[0.23, 0.26, 0.30, 1],
            )
        world.add_light(pos=[0.9, -1.1, 1.2], dir=[-0.4, 0.5, -0.8])
        world.add_light(pos=[-1.0, 0.9, 1.0], dir=[0.5, -0.4, -0.8])

        # Ghost-arm reference display: mocap bodies positioned per frame by
        # the UI (set_ghost_pose). Parked at z=-5 until then.
        ghost = (
            ("ghost_lu", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.006, 0.0325, 0.006],
             [0.35, 0.80, 1.00, 0.9]),
            ("ghost_lf", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.005, 0.0325, 0.005],
             [0.35, 0.80, 1.00, 0.9]),
            ("ghost_ru", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.006, 0.0325, 0.006],
             [1.00, 0.62, 0.25, 0.9]),
            ("ghost_rf", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.005, 0.0325, 0.005],
             [1.00, 0.62, 0.25, 0.9]),
            ("ghost_bar", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.007, 0.05, 0.007],
             [0.85, 0.90, 1.00, 0.8]),
            ("ghost_head", mujoco.mjtGeom.mjGEOM_SPHERE, [0.026, 0.0, 0.0],
             [0.85, 0.90, 1.00, 0.8]),
        )
        for name, gtype, size, rgba in ghost:
            body = world.add_body(name=name, mocap=True, pos=[0, 0, -5])
            body.add_geom(type=gtype, size=size, rgba=rgba,
                          group=2, contype=0, conaffinity=0)

    def _usable_ranges(self) -> dict[str, tuple[float, float]]:
        """Usable joint ranges.

        The MJCF carries CAD-export placeholder limits (every joint +/-2 rad);
        the URDF carries the real hardware limits but at least one entry
        (LM_J0) disagrees with the trainer's HOME pose. So: URDF limits,
        widened wherever needed so HOME is always feasible. mj_forward does
        not enforce limits, so commanding the wider URDF range is safe.
        """
        from app.constants import JOINT_RANGE

        ranges = {}
        for name in HOME:
            urdf = JOINT_RANGE.get(name, (-2.0, 2.0))
            lo, hi = urdf
            home = HOME[name]
            lo = min(lo, home - 0.05)
            hi = max(hi, home + 0.05)
            ranges[name] = (lo, hi)
        return ranges

    def clamp(self, name: str, value: float) -> float:
        lo, hi = self.joint_range[name]
        return float(np.clip(value, lo, hi))

    # -------------------------------------------------------- claw geometry ----
    @property
    def claw_geometry(self) -> dict:
        """Claw-chain facts the GMR-style IK retarget needs (lazy, cached).

        anchor[leg]: shoulder-link position in the BASE frame at HOME -- no
          claw joint can move it, only the free joint, so the IK series can
          place it per frame as anchor_offset + base pose.
        reach[leg]: (palm, forearm) max radius from the shoulder over a
          J1..J3 grid (J0 spins about a vertical axis through the anchor, so
          it cannot change the radius) -- the practical envelope the human
          hand target is scaled onto.
        seg_len[leg]: (L1, L2) -- upper-arm length (shoulder -> forearm
          origin, rigid) and claw length (forearm origin -> palm site,
          near-rigid), the denominators of the direction objectives.
        """
        if self._claw_geom is None:
            geom: dict = {"anchor": {}, "reach": {}, "seg_len": {}}
            m, d = self.model, self.data
            mujoco.mj_resetData(m, d)
            self.set_home_joints()
            self.set_body(self.z_stand, 0.0, 0.0, 0.0)
            mujoco.mj_forward(m, d)
            for leg in CLAWS:
                shoulder = np.array(d.body(f"{leg}_shoulder_link").xpos)
                geom["anchor"][leg] = shoulder - np.array([0.0, 0.0, self.z_stand])
                fore = np.asarray(d.body(f"{leg}_forearm_link").xpos)
                palm = np.asarray(d.site(leg).xpos)
                geom["seg_len"][leg] = (
                    float(np.linalg.norm(fore - shoulder)),
                    float(np.linalg.norm(palm - fore)))
                names = [f"{leg}_J{i}_joint" for i in range(4)]
                adr = [self.joint_adr[n] for n in names]
                d.qpos[adr[0]] = HOME[names[0]]
                best_p = best_e = 0.0
                grid = [np.linspace(*self.joint_range[n], 9) for n in names[1:]]
                for j1 in grid[0]:
                    for j2 in grid[1]:
                        for j3 in grid[2]:
                            for a, v in zip(adr[1:], (j1, j2, j3)):
                                d.qpos[a] = v
                            mujoco.mj_forward(m, d)
                            best_p = max(best_p, float(np.linalg.norm(
                                np.asarray(d.site(leg).xpos) - shoulder)))
                            best_e = max(best_e, float(np.linalg.norm(
                                np.asarray(d.body(f"{leg}_forearm_link").xpos)
                                - shoulder)))
                geom["reach"][leg] = (best_p, best_e)
            mujoco.mj_resetData(m, d)
            self._claw_geom = geom
        return self._claw_geom

    # ------------------------------------------------------- calibration ----
    def _calibrate_home(self) -> None:
        """Stand at HOME on the origin; measure z_stand from the foot sites."""
        mujoco.mj_resetData(self.model, self.data)
        self.set_home_joints()
        self.data.qpos[self.free_adr + 2] = 0.20  # any positive height
        mujoco.mj_forward(self.model, self.data)
        zs = [
            float(self.data.site(leg).xpos[2]) for leg in (*LEGS, *CLAWS)
        ]
        # Base sits at z=0.20; z_stand lowers it until the lowest foot site
        # (all six: 4 legs + both claw pads) touches the floor.
        self.z_stand = 0.20 - min(zs)
        self._home_foot_world = {
            leg: np.array(self.data.site(leg).xpos[:2]).copy() for leg in LEGS
        }
        mujoco.mj_resetData(self.model, self.data)

    def _calibrate_leg(self, leg: str) -> LegCalib:
        mujoco.mj_resetData(self.model, self.data)
        self.set_home_joints()
        # Base at the origin: world coordinates ARE base-frame coordinates.
        # (mj_forward is purely kinematic; intersecting the floor harms nothing.)
        mujoco.mj_forward(self.model, self.data)

        p1 = np.array(self.data.body(f"{leg}_thigh_link").xpos)
        p2 = np.array(self.data.body(f"{leg}_calf_link").xpos)
        pf = np.array(self.data.site(leg).xpos)

        e1 = p2 - p1
        e2 = pf - p2
        a = float(np.linalg.norm(e1))
        b = float(np.linalg.norm(e2))
        e1u, e2u = e1 / a, e2 / b
        n = np.cross(e1u, e2u)
        n /= np.linalg.norm(n)
        t1 = e1u
        t2 = np.cross(n, t1)
        beta_home = math.atan2(float(np.dot(e2u, t2)), float(np.dot(e2u, t1)))

        # Empirical axis signs: perturb J1/J2 by a small positive delta and
        # watch which way the thigh / calf azimuths actually move in the
        # (t1, t2) plane. The left legs mount mirrored, so their axes flip.
        def _azim(vec: np.ndarray) -> float:
            return math.atan2(float(np.dot(vec, t2)), float(np.dot(vec, t1)))

        q1_adr = self.joint_adr[f"{leg}_J1_joint"]
        q2_adr = self.joint_adr[f"{leg}_J2_joint"]

        self.data.qpos[q1_adr] += 0.15
        mujoco.mj_forward(self.model, self.data)
        thigh_az = _azim(np.asarray(self.data.body(f"{leg}_calf_link").xpos)
                         - np.asarray(self.data.body(f"{leg}_thigh_link").xpos))
        axis1 = 1.0 if thigh_az > 0.0 else -1.0
        self.data.qpos[q1_adr] -= 0.15

        self.data.qpos[q2_adr] += 0.15
        mujoco.mj_forward(self.model, self.data)
        calf_az = _azim(np.asarray(self.data.site(leg).xpos)
                        - np.asarray(self.data.body(f"{leg}_calf_link").xpos))
        axis2 = 1.0 if calf_az > beta_home else -1.0
        self.data.qpos[q2_adr] -= 0.15

        mujoco.mj_resetData(self.model, self.data)
        return LegCalib(
            name=leg, p1=p1, a=a, b=b, n=n, t1=t1, t2=t2,
            beta_home=beta_home, axis1=axis1, axis2=axis2,
            q0_home=HOME[f"{leg}_J0_joint"],
            q1_home=HOME[f"{leg}_J1_joint"],
            q2_home=HOME[f"{leg}_J2_joint"],
            foot_xy=self._home_foot_world[leg],
            q1_range=self.joint_range[f"{leg}_J1_joint"],
            q2_range=self.joint_range[f"{leg}_J2_joint"],
        )

    # ------------------------------------------------------------- poses ----
    def _max_ik_height(self) -> float:
        """Highest base z where every walking leg still reaches the floor.

        Scans upward until some leg's in-plane target distance hits its
        2-link reach (a + b, minus a 1 mm margin so knees never lock dead
        straight). Bounds the user height slider.
        """
        def reach(leg: str, z: float) -> float:
            c = self.calib[leg]
            tv = np.array([c.foot_xy[0], c.foot_xy[1], -z]) - c.p1
            return math.hypot(float(tv @ c.t1), float(tv @ c.t2))

        z = self.z_stand
        limit = min(c.a + c.b - 1e-3 for c in self.calib.values())
        while z < 0.4 and all(reach(leg, z + 5e-4) <= limit for leg in LEGS):
            z += 5e-4
        return z

    def set_home_joints(self) -> None:
        for name, value in HOME.items():
            self.data.qpos[self.joint_adr[name]] = value

    def set_body(self, z: float, roll: float = 0.0, pitch: float = 0.0,
                 yaw: float = 0.0, xy: tuple[float, float] = (0.0, 0.0)) -> None:
        """World position (xy, z) with Rz(yaw)Ry(pitch)Rx(roll) attitude."""
        adr = self.free_adr
        self.data.qpos[adr + 0:adr + 3] = (xy[0], xy[1], z)
        self.data.qpos[adr + 3:adr + 7] = quat_from_rpy(roll, pitch, yaw)

    def leg_ik(self, leg: str, base_z: float, base_quat: np.ndarray,
               yaw: float = 0.0,
               foot_z: float = 0.0,
               base_xy: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        """J1/J2 so the foot reaches its (yaw-rotated) HOME xy at height foot_z.

        foot_z = 0 plants the foot on the floor; a positive value lifts it
        (leg-raise gesture). The HOME foot position is rotated by the body's
        yaw, so a turning crab pivots its feet in place instead of twisting
        the leg planes. base_xy shifts the whole stance (stage placement).
        """
        c = self.calib[leg]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, base_quat)
        rot = rot.reshape(3, 3)

        # Foot target: HOME xy rotated by yaw at the requested height,
        # relative to the base's stage position.
        fx, fy = c.foot_xy
        target_w = np.array([
            base_xy[0] + fx * math.cos(yaw) - fy * math.sin(yaw),
            base_xy[1] + fx * math.sin(yaw) + fy * math.cos(yaw), foot_z])
        base_w = np.array([base_xy[0], base_xy[1], base_z])
        target_b = rot.T @ (target_w - base_w)
        t_vec = target_b - c.p1

        tx = float(np.dot(t_vec, c.t1))
        ty = float(np.dot(t_vec, c.t2))
        # Out-of-plane part is unsolvable (planar mechanism); projection only.
        r = math.hypot(tx, ty)
        r = min(max(r, abs(c.a - c.b) + 1e-4), c.a + c.b - 1e-4)
        psi = math.atan2(ty, tx)

        cos_beta = (r * r - c.a * c.a - c.b * c.b) / (2 * c.a * c.b)
        cos_beta = min(1.0, max(-1.0, cos_beta))
        beta = math.copysign(math.acos(cos_beta), c.beta_home)
        gamma = math.atan2(c.b * math.sin(beta), c.a + c.b * math.cos(beta))
        theta1 = psi - gamma

        # azimuth_home(J1) = 0; joint deltas reach the (t1, t2) plane through
        # the measured axis signs (mirrored legs rotate the opposite way).
        q1 = c.q1_home + c.axis1 * _wrap(theta1)
        q2 = c.q2_home + c.axis2 * _wrap(beta - c.beta_home)

        q1 = float(np.clip(q1, *c.q1_range))
        q2 = float(np.clip(q2, *c.q2_range))
        return q1, q2

    def pose_dance(self, base_z: float, roll: float, pitch: float, yaw: float,
                   claw_joints: dict[str, float],
                   leg_sway: float = 0.0,
                   yaw_offset_deg: float = 0.0,
                   foot_lift: dict[str, float] | None = None,
                   base_xy: tuple[float, float] = (0.0, 0.0)) -> None:
        """Write a full dance pose. claw_joints: 10 entries, LF/RF x J0..J4.

        yaw_offset_deg is a constant added to the commanded yaw (used by the
        UI so "yaw 0" faces the preview camera). Calibration callers keep 0.
        foot_lift: optional {leg: height_m} raising that leg's foot target
        off the floor (leg-raise gesture; walking legs only). base_xy moves
        the whole stance across the stage (multi-crab placement).
        """
        roll, pitch, yaw = (math.radians(v) for v in (roll, pitch, yaw))
        yaw += math.radians(yaw_offset_deg)
        self.set_body(base_z, roll, pitch, yaw, xy=base_xy)
        quat = self.data.qpos[self.free_adr + 3:self.free_adr + 7].copy()

        for leg in LEGS:
            # Front-leg stance (user): the two front walking legs reach
            # further forward (LM negative, RM positive).
            stance = 0.0
            if leg == "LM":
                stance = -math.radians(FRONT_LEG_J0_STANCE_DEG)
            elif leg == "RM":
                stance = math.radians(FRONT_LEG_J0_STANCE_DEG)
            j0 = self.clamp(f"{leg}_J0_joint",
                            self.calib[leg].q0_home
                            + leg_sway * (1.0 if leg in ("LM", "RR") else -1.0)
                            + stance)
            q1, q2 = self.leg_ik(leg, base_z, quat, yaw=yaw,
                                 foot_z=float((foot_lift or {}).get(leg, 0.0)),
                                 base_xy=base_xy)
            self.data.qpos[self.joint_adr[f"{leg}_J0_joint"]] = j0
            self.data.qpos[self.joint_adr[f"{leg}_J1_joint"]] = q1
            self.data.qpos[self.joint_adr[f"{leg}_J2_joint"]] = q2

        for name, value in claw_joints.items():
            self.data.qpos[self.joint_adr[name]] = self.clamp(name, value)

        mujoco.mj_forward(self.model, self.data)

    def apply_qpos(self, qpos: np.ndarray) -> None:
        """Write a precomputed full qpos (used by the timeline playback)."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ render ----
    def set_floor_rgba(self, rgba) -> None:
        """Tint the dance floor; None restores the compiled default."""
        if rgba is None:
            rgba = self._floor_default
        self.model.geom("dance_floor").rgba[:] = rgba

    # -------------------------------------------------------- ghost arms ----
    def set_ghost_pose(self, segments: dict | None) -> None:
        """Position the mocap ghost-arm bodies; None parks them out of sight.

        segments maps body name -> (a, b) world endpoints of a capsule whose
        local z axis spans the segment. Call before pose_dance() so the
        trailing mj_forward picks the mocap transform up.
        """
        if segments is None:
            segments = {}
        for name in self._ghost_bodies:
            mid = self.model.body(name).mocapid
            seg = segments.get(name)
            if seg is None:
                self.data.mocap_pos[mid] = (0.0, 0.0, -5.0)
                self.data.mocap_quat[mid] = (1.0, 0.0, 0.0, 0.0)
                continue
            a, b = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
            self.data.mocap_pos[mid] = (a + b) / 2.0
            d = b - a
            n = float(np.linalg.norm(d))
            # NB: mocap_quat[i] hands out a copy in this build -- the result
            # must be assigned back with setitem to reach mjData.
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            if n > 1e-9:
                mujoco.mju_quatZ2Vec(quat, d / n)
            self.data.mocap_quat[mid] = quat

    # Ceiling for the offscreen framebuffer. GL buffers scale with it, so it
    # bounds GPU memory; panels larger than this get a smaller render that the
    # UI upscales (aspect is preserved).
    _MAX_OFFSCREEN = 4096

    def _ensure_offscreen(self, w: int, h: int) -> tuple[int, int]:
        """Make the offscreen framebuffer at least w x h; return usable size.

        jumper.xml declares 1280x960, which is smaller than the 3D panel on a
        large/high-DPI window. mujoco.Renderer raises ValueError in that case
        and the whole view stays blank, so grow the model's framebuffer to the
        requested size before (re)creating the renderer.
        """
        s = min(1.0, self._MAX_OFFSCREEN / max(w, h))
        if s < 1.0:
            w, h = max(1, int(w * s)), max(1, int(h * s))
        if (w > self.model.vis.global_.offwidth
                or h > self.model.vis.global_.offheight):
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth), w)
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight), h)
        return w, h

    def render(self, camera: "mujoco.MjvCamera | None" = None,
               width: int | None = None, height: int | None = None) -> np.ndarray:
        """Offscreen RGB uint8 frame of the current data."""
        w = width or self._render_width
        h = height or self._render_height
        if self._renderer is None or (w, h) != (self._render_width, self._render_height):
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None  # drop the old GL context first
            w, h = self._ensure_offscreen(w, h)
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._render_width, self._render_height = w, h
        # This mujoco build wants -1 (free camera), not None, for the default.
        self._renderer.update_scene(
            self.data, camera=-1 if camera is None else camera)
        return self._renderer.render()

    def render_with_mask(self, camera: "mujoco.MjvCamera | None" = None,
                         width: int | None = None,
                         height: int | None = None,
                         robot_only: bool = False
                         ) -> tuple[np.ndarray, np.ndarray, tuple | None]:
        """RGB frame plus a boolean mask (True on robot/floor, False on sky).

        Used by the UI to composite gradient backgrounds behind the crab.
        robot_only strips the floor from the mask, so a second crab layered
        on top does not erase the first one with its own floor plane. Also
        returns the robot's pixel bbox (x0, y0, x1, y1) or None -- the UI
        uses it for hit-testing (drag a crab) and number badges.
        """
        rgb = self.render(camera=camera, width=width, height=height)
        renderer = self._renderer  # created/refreshed by render() above
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        floor_id = self.model.geom("dance_floor").id
        is_geom = seg[..., 0] >= 0
        mask = is_geom
        robot_only_mask = is_geom & (seg[..., 0] != floor_id)
        if robot_only:
            mask = robot_only_mask
        bbox = None
        ys, xs = np.where(robot_only_mask)
        if len(xs):
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return rgb, mask, bbox

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def calibrate_claw_signs(self) -> dict[str, dict[str, float]]:
        """Empirically find which direction each claw joint moves its tip.

        Perturbs each joint by +0.25 rad from HOME and watches the claw site
        (the palm-pad contact site) and the finger tip. Returns, per claw:
          lift : sign of J1 that raises the claw
          lat  : sign of J0 that moves the claw outward (away from body, +y
                 for LF / -y for RF)
          ccw  : sign of J0 that sweeps the claw counter-clockwise (viewed
                 from above, rotation about +z) -- used for the J0 rest bias
          fold : sign of J2 (and J3) that pulls the claw toward the body
          open : sign of J4 that opens the gripper (tip further from palm)
        No CAD math, no hardcoded mirror tables -- survives model revisions.
        """
        signs = {}
        for leg in CLAWS:
            out = self.joint_adr
            self.pose_dance(self.z_stand, 0, 0, 0, {n: HOME[n] for n in HOME if n.startswith(leg)})
            anchor0 = np.array(self.data.body(f"{leg}_shoulder_link").xpos)
            palm0 = np.array(self.data.site(leg).xpos)
            tip0 = np.array(self.data.body(f"{leg}_finger_tip_link").xpos)

            def probe(jname: str) -> tuple[np.ndarray, np.ndarray]:
                save = self.data.qpos[out[jname]]
                self.data.qpos[out[jname]] = save + 0.25
                mujoco.mj_forward(self.model, self.data)
                palm = np.array(self.data.site(leg).xpos)
                tip = np.array(self.data.body(f"{leg}_finger_tip_link").xpos)
                self.data.qpos[out[jname]] = save
                mujoco.mj_forward(self.model, self.data)
                return palm - palm0, tip - tip0

            def sweep(jname: str) -> float:
                """Signed azimuth change of the palm about the shoulder
                anchor, in the xy plane (deg): + = counter-clockwise."""
                save = self.data.qpos[out[jname]]
                self.data.qpos[out[jname]] = save + 0.25
                mujoco.mj_forward(self.model, self.data)
                palm = np.array(self.data.site(leg).xpos)
                self.data.qpos[out[jname]] = save
                mujoco.mj_forward(self.model, self.data)
                a0 = math.atan2(palm0[1] - anchor0[1], palm0[0] - anchor0[0])
                a1 = math.atan2(palm[1] - anchor0[1], palm[0] - anchor0[0])
                return math.degrees(_wrap(a1 - a0))

            dp1, _ = probe(f"{leg}_J1_joint")
            dp0, _ = probe(f"{leg}_J0_joint")
            dp2, _ = probe(f"{leg}_J2_joint")
            j0_ccw = sweep(f"{leg}_J0_joint")

            outward_y = 1.0 if leg == "LF" else -1.0  # LF sits on +y side
            grip0 = np.linalg.norm(tip0 - palm0)

            # Opening = distance finger-tip to palm-pad increases.
            save = self.data.qpos[out[f"{leg}_J4_joint"]]
            self.data.qpos[out[f"{leg}_J4_joint"]] = save + 0.25
            mujoco.mj_forward(self.model, self.data)
            grip_up = np.linalg.norm(
                np.array(self.data.body(f"{leg}_finger_tip_link").xpos)
                - np.array(self.data.site(leg).xpos)
            ) - grip0
            self.data.qpos[out[f"{leg}_J4_joint"]] = save
            mujoco.mj_forward(self.model, self.data)

            signs[leg] = {
                "lift": float(np.sign(dp1[2]) or 1.0),
                "lat": float(np.sign(dp0[1] * outward_y) or 1.0),
                "ccw": float(np.sign(j0_ccw) or 1.0),
                "fold": float(np.sign(-(dp2[0:2] @ palm0[0:2]) / (np.linalg.norm(palm0[0:2]) + 1e-9)) or 1.0),
                "open": float(np.sign(grip_up) or 1.0),
            }
        return signs

    # ----------------------------------------------------------- selftest ----
    def selftest(self) -> list[str]:
        """Sanity checks; returns a list of failure messages (empty = pass)."""
        errors = []

        if not (0.07 <= self.z_stand <= 0.16):
            errors.append(f"z_stand={self.z_stand:.4f} m out of [0.07, 0.16]")

        # The IK-height bound must sit well above standing, with the legs
        # short of dead-straight.
        zf = self.z_max_ik / self.z_stand
        if not (1.3 <= zf <= 1.75):
            errors.append(f"z_max_ik={zf:.2f}x stand outside [1.3, 1.75]")

        # Leg IK round-trip: asking for the standing height must return HOME.
        # Tolerance 0.02 rad: the six foot sites sit at slightly different
        # heights, so z_stand (set by the lowest) is a few tenths of a mm
        # below some legs' HOME height.
        for leg in LEGS:
            c = self.calib[leg]
            q1, q2 = self.leg_ik(leg, self.z_stand,
                                  np.array([1.0, 0.0, 0.0, 0.0]))
            if abs(q1 - c.q1_home) > 0.02 or abs(q2 - c.q2_home) > 0.02:
                errors.append(
                    f"{leg} IK round-trip: home=({c.q1_home:.4f},{c.q2_home:.4f}) "
                    f"got=({q1:.4f},{q2:.4f})"
                )

        # Body at half crouch must produce sane leg angles inside limits.
        for z in (0.6 * self.z_stand, 1.0 * self.z_stand):
            for leg in LEGS:
                q1, q2 = self.leg_ik(leg, z, np.array([1.0, 0.0, 0.0, 0.0]))
                lo1, hi1 = self.calib[leg].q1_range
                lo2, hi2 = self.calib[leg].q2_range
                if not (lo1 <= q1 <= hi1 and lo2 <= q2 <= hi2):
                    errors.append(f"{leg} IK out of range at z={z:.3f}")

        # Feet must actually stay planted through a crouch. A joint-axis
        # sign inversion passes the round-trip above (theta ~ 0 at standing
        # height) yet folds the mirrored legs the wrong way, sinking the
        # feet tens of mm under the floor -- exactly what this catches.
        for zf in (1.0, 0.85, 0.7):
            mujoco.mj_resetData(self.model, self.data)
            self.set_home_joints()
            self.set_body(zf * self.z_stand, 0.0, 0.0, 0.0)
            quat = self.data.qpos[self.free_adr + 3:self.free_adr + 7].copy()
            for leg in LEGS:
                q1, q2 = self.leg_ik(leg, zf * self.z_stand, quat)
                self.data.qpos[self.joint_adr[f"{leg}_J1_joint"]] = q1
                self.data.qpos[self.joint_adr[f"{leg}_J2_joint"]] = q2
            mujoco.mj_forward(self.model, self.data)
            for leg in LEGS:
                fz = float(self.data.site(leg).xpos[2])
                if abs(fz) > 0.002:
                    errors.append(
                        f"{leg} foot off floor at z={zf:.2f}: {fz * 1000:.1f} mm")

        # Turning in place (yaw + front offset) must keep the feet planted
        # too -- the pivoting foot targets rotate with the body.
        for yaw_deg in (60.0, 135.0, 225.0):
            self.pose_dance(self.z_stand, 0.0, 0.0, 0.0,
                            {n: HOME[n] for n in
                             (f"{p}_J{i}_joint" for p in CLAWS for i in range(5))},
                            yaw_offset_deg=yaw_deg)
            for leg in LEGS:
                fz = float(self.data.site(leg).xpos[2])
                if abs(fz) > 0.003:
                    errors.append(
                        f"{leg} foot off floor at yaw={yaw_deg:.0f}: "
                        f"{fz * 1000:.1f} mm")

        # Stage placement: the whole stance shifts with base_xy, feet still
        # planted (multi-crab mode relies on this). The stage shift is measured
        # against the same pose at the origin, so the front-leg stance (which
        # deliberately moves LM/RM feet off their HOME xy) is not mistaken for
        # a placement error.
        claws_home = {n: HOME[n] for n in
                      (f"{p}_J{i}_joint" for p in CLAWS for i in range(5))}
        self.pose_dance(self.z_stand, 0.0, 0.0, 0.0, claws_home)
        at_origin = {leg: np.asarray(self.data.site(leg).xpos).copy()
                     for leg in LEGS}
        self.pose_dance(self.z_stand, 0.0, 0.0, 0.0, claws_home,
                        base_xy=(0.2, -0.15))
        for leg in LEGS:
            site = np.asarray(self.data.site(leg).xpos)
            want = np.concatenate([at_origin[leg][:2] + [0.2, -0.15],
                                   at_origin[leg][2:]])
            off = float(np.linalg.norm(site - want))
            if off > 0.003:
                errors.append(f"{leg} misplaced at base_xy: "
                              f"off {off*1000:.1f} mm, "
                              f"z {site[2]*1000:.1f} mm")

        # A lifted foot target must place that foot at the requested height
        # while the others stay planted (0.35x is inside the folded-leg
        # envelope; ~0.46x is where the 2-link clamps out).
        lift = 0.35 * self.z_stand
        self.pose_dance(self.z_stand, 0.0, 0.0, 0.0,
                        {n: HOME[n] for n in
                         (f"{p}_J{i}_joint" for p in CLAWS for i in range(5))},
                        foot_lift={"LM": lift})
        fz = {leg: float(self.data.site(leg).xpos[2]) for leg in LEGS}
        if abs(fz["LM"] - lift) > 0.003:
            errors.append(f"LM lift target {lift*1000:.0f} mm, got {fz['LM']*1000:.1f} mm")
        for leg in ("RM", "LR", "RR"):
            if abs(fz[leg]) > 0.002:
                errors.append(f"{leg} left floor during LM lift: {fz[leg]*1000:.1f} mm")

        # Render one frame headless.
        try:
            self.pose_dance(self.z_stand, 0.0, 0.0, 0.0,
                            {n: HOME[n] for n in
                             (f"{p}_J{i}_joint" for p in CLAWS for i in range(5))})
            import mujoco as mj
            cam = mj.MjvCamera()
            cam.lookat[:] = [0, 0, 0.06]
            cam.distance = 0.5
            cam.azimuth = 135
            cam.elevation = -15
            frame = self.render(camera=cam, width=320, height=240)
            if frame.shape != (240, 320, 3) or frame.mean() == 0:
                errors.append(f"render produced bad frame {frame.shape}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"render failed: {exc}")

        return errors


if __name__ == "__main__":
    crab = CrabModel()
    issues = crab.selftest()
    print(f"z_stand = {crab.z_stand*1000:.1f} mm")
    for leg in LEGS:
        c = crab.calib[leg]
        print(f"{leg}: a={c.a*1000:.1f} b={c.b*1000:.1f} mm "
              f"beta_home={math.degrees(c.beta_home):.1f} deg "
              f"axis=({c.axis1:+.0f},{c.axis2:+.0f})")
    if issues:
        print("SELFTEST FAILURES:")
        for msg in issues:
            print(" -", msg)
        raise SystemExit(1)
    print("selftest passed")
