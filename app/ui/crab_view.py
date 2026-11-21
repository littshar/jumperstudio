"""Right panel: the jumper crab dancing in a MuJoCo offscreen render.

Supports one crab (idle / single-person mode) or one crab per detected
person (multi mode): each actor is its own CrabModel posed at a stage
offset, rendered back-to-front and composited over the backdrop.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

_RENDER_RETRIES = 3   # auto-retries before the click-to-retry state

from app.constants import (
    BG_DEFAULT, BODY_SIGN_PITCH, BODY_SIGN_ROLL, BODY_SIGN_YAW, CACHE_DIR,
    CLAW_J0_LIMIT_DEG, CLAW_J0_OFFSET_DEG, CLAW_J0_SPREAD_DEG,
    CLAW_J1_MIN_DEG,
    CLAW_J1_OFFSET_DEG, CLAW_J3_OFFSET_DEG,
    CLAW_J0_CROSS_LIMIT_DEG, CRAB_BACKGROUNDS,
    CRAB_FRONT_YAW_OFFSET_DEG, CRAB_RENDER_FPS, HEIGHT_FACTOR_MAX,
    HEIGHT_FACTOR_MIN, HOME, JUMP_HOP_GAIN, K_PITCH, K_ROLL, K_YAW,
    STAGE_SPREAD,
)
from app.core.crab_model import CrabModel
from app.core.retarget import (
    CrabCommandTable, arm_channels_from_points,
    clip_claw_joints, ghost_arm_points, sample_commands,
    solve_claw_ik_pose, wrap180,
)


class MappingOverlay(QWidget):
    """Semi-transparent panel annotating the human -> crab retarget mapping.

    Informational only: transparent to the mouse (orbit/zoom keep working
    through it) and repainted with every rendered frame so the live command
    values track the crab.
    """

    _PANEL = QColor(8, 12, 18, 216)
    _BORDER = QColor(48, 54, 61)
    _TITLE = QColor(230, 180, 80)
    _TEXT = QColor(201, 209, 217)
    _VALUE = QColor(97, 175, 239)
    _NOTE = QColor(139, 148, 158)

    _WIDTH = 330

    def __init__(self, view: "CrabView"):
        super().__init__(view)
        self._view = view
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._title_font = QFont("Microsoft YaHei", 9)
        self._title_font.setBold(True)
        self._font = QFont("Microsoft YaHei", 9)
        self._content = self._build_content()
        h = sum({"title": 24, "sep": 9, "row": 19, "note": 16}[kind]
                for kind, _ in self._content) + 16
        self.setFixedSize(self._WIDTH, h)

    @staticmethod
    def _build_content() -> list[tuple[str, object]]:
        # Rows carry (text, live-key); the live value is shown right-aligned.
        return [
            ("title", "重定向映射 · 人体 → 螃蟹"),
            ("sep", None),
            ("row", ("躯干左右倾 → 身体侧倾 roll", "roll")),
            ("row", ("躯干前后倾 → 身体俯仰 pitch", "pitch")),
            ("row", ("躯干转身 → 身体转向 yaw", "yaw")),
            ("row", ("下蹲程度 → 身体高度 z", "base_z")),
            ("row", ("跳跃 → 螃蟹腾空", "jump")),
            ("row", ("抬腿 → 同侧行走腿抬起", "raise")),
            ("sep", None),
            ("row", ("肩部 yaw 水平指向 → 钳 J0", None)),
            ("row", ("小臂朝向 → 钳 J1", None)),
            ("row", ("肘部弯曲 → 钳 J2 + J3 共同表达", None)),
            ("row", ("手举过肩 → 钳口 J4 张开", None)),
            ("note", "左臂 → 左钳 LF ・ 右臂 → 右钳 RF"),
            ("sep", None),
            ("note", "四条行走腿：IK 自动保持站姿"),
            ("note", f"增益 roll ×{K_ROLL:g} · pitch ×{K_PITCH:g} · yaw ×{K_YAW:g}"),
            ("note", f"方向符号 roll/pitch/yaw "
                     f"{BODY_SIGN_ROLL:+.0f}/{BODY_SIGN_PITCH:+.0f}/{BODY_SIGN_YAW:+.0f}"),
        ]

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._PANEL)
        p.drawRoundedRect(rect, 8, 8)
        p.setPen(QPen(self._BORDER, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 8, 8)

        live = self._view.live_commands or {}
        x, y = 12, 8
        for kind, item in self._content:
            if kind == "sep":
                p.setPen(QPen(QColor(33, 38, 45), 1))
                p.drawLine(x, y + 4, self._WIDTH - x, y + 4)
                y += 9
                continue
            if kind == "title":
                p.setFont(self._title_font)
                p.setPen(self._TITLE)
                p.drawText(x, y, self._WIDTH - 2 * x, 24, Qt.AlignmentFlag.AlignVCenter, str(item))
                y += 24
            elif kind == "row":
                text, key = item
                p.setFont(self._font)
                p.setPen(self._TEXT)
                p.drawText(x, y, self._WIDTH - 2 * x - 70, 19,
                           Qt.AlignmentFlag.AlignVCenter, text)
                if key:
                    p.setPen(self._VALUE)
                    value = live.get(key)
                    if key == "base_z":
                        shown = f"{value * 1000:4.0f} mm" if value is not None else "—"
                    elif key in ("jump", "raise"):
                        shown = f"{value * 100:3.0f} %" if value is not None else "—"
                    else:
                        shown = f"{value:+5.1f}°" if value is not None else "—"
                    p.drawText(self._WIDTH - x - 66, y, 66, 19,
                               Qt.AlignmentFlag.AlignVCenter
                               | Qt.AlignmentFlag.AlignRight, shown)
                y += 19
            else:  # note
                p.setFont(self._font)
                p.setPen(self._NOTE)
                p.drawText(x, y, self._WIDTH - 2 * x, 16,
                           Qt.AlignmentFlag.AlignVCenter, str(item))
                y += 16


@dataclass(eq=False)
class _Actor:
    """One crab on the stage: its model, command table and stage slot."""
    crab: CrabModel
    table: CrabCommandTable
    xy: tuple[float, float] = (0.0, 0.0)
    spin: object = False
    jump: object = True
    yaw_prev: float | None = None
    channels: object = None     # actor 0 only (ghost figure source)


class CrabView(QWidget):
    """Kinematic puppet view driven by one or more command tables + a time."""

    _LOOKAT_BASE = (0.0, 0.0, 0.07)
    # Ghost-arm display (separate 3D reconstruction of the human arms),
    # floating above the crab like a marionette reference.
    _GHOST_CENTER = np.array([0.0, 0.0, 0.30])
    _GHOST_LEN = 0.065

    def __init__(self, crab: CrabModel, parent=None):
        super().__init__(parent)
        self.crab = crab
        self.actors: list[_Actor] = []
        self.table: CrabCommandTable | None = None
        self._channels = None
        self._ghost_visible = False

        self._label = QLabel("crab", alignment=Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(320, 240)
        self._label.setStyleSheet(
            "background:#0d1117; color:#7d8590; border-radius:6px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.0, 0.0, 0.07]
        self.camera.distance = 0.48
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0

        self._pixmap: QPixmap | None = None
        self._render_error = False
        self._render_dead = False
        self._render_retries = 0
        self._render_detail = ""
        self._min_render_interval = 1.0 / CRAB_RENDER_FPS
        self._last_render = 0.0
        self._last_t = None
        self._live: dict | None = None
        self._spin = False
        self._height_factor = 1.0
        self._yaw_prev: float | None = None

        # Extra crab models for multi-person mode, pooled across videos
        # (each CrabModel compile is ~0.3 s, so never throw them away).
        self._pool: list[CrabModel] = []
        self._pool_floor: list = []   # floor rgba applied per pooled model
        # Per-actor robot pixel boxes from the last paint, painter order
        # (far -> near); drives drag hit-testing and number badges.
        self._hit_boxes: list = []
        self._drag_actor: _Actor | None = None
        self._press_pos: tuple[float, float] | None = None
        self._jump: object = True
        self._probe: tuple | None = None   # (joint_idx, delta) calibration
        self._manual: dict | None = None   # manual ghost-arm posing mode

        # Background preset + gradient/image cache (rebuilt on size change).
        self._bg = CRAB_BACKGROUNDS[0]
        self._grad_key: tuple | None = None
        self._grad: np.ndarray | None = None
        # User-imported backdrop image (RGB) and floor texture (PNG path).
        self._bg_image: np.ndarray | None = None
        self._floor_tex: Path | None = None

        self._overlay = MappingOverlay(self)

        # Click-to-retry surface for a dead renderer. Real buttons (not
        # rich-text links): the old link text was overwritten by the dead
        # branch, which left the click surface with nothing to click.
        self._retry_link = QWidget(self)
        self._retry_link.setStyleSheet(
            "background:rgba(13,17,23,235); border-radius:6px;")
        retry_lay = QVBoxLayout(self._retry_link)
        retry_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        retry_msg = QLabel("3D 渲染不可用")
        retry_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        retry_msg.setStyleSheet(
            "color:#e6b450; font-size:15px; font-weight:600;")
        retry_lay.addWidget(retry_msg)
        retry_row = QHBoxLayout()
        retry_row.addStretch(1)
        self.retry_btn = QPushButton("⟳ 重试")
        self.retry_btn.setObjectName("accentBtn")
        self.retry_btn.clicked.connect(self.retry_render)
        self.diag_btn = QPushButton("查看诊断")
        self.diag_btn.clicked.connect(self._show_render_diag)
        retry_row.addWidget(self.retry_btn)
        retry_row.addWidget(self.diag_btn)
        retry_row.addStretch(1)
        retry_lay.addLayout(retry_row)
        self._retry_link.setVisible(False)

        self.setMouseTracking(True)
        self.setMinimumHeight(240)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Keep the mapping annotation anchored to the top-right corner.
        self._overlay.move(self.width() - self._overlay.width() - 10, 10)
        if self._retry_link.isVisible():
            self._retry_link.setGeometry(self._label.geometry())

    # ------------------------------------------------------ render retry ----
    def _show_render_diag(self) -> None:
        QMessageBox.information(
            self, "3D 渲染诊断",
            self._render_detail or "无详细信息")

    def retry_render(self) -> None:
        """Recreate every renderer (fresh GPU context) and repaint."""
        for model in (self.crab, *self._pool):
            try:
                if model._renderer is not None:
                    model._renderer.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
            model._renderer = None
        self._render_error = False
        self._render_dead = False
        self._render_retries = 0
        self._render_detail = ""
        self._retry_link.setVisible(False)
        self._last_render = 0.0
        self._repaint_soon()

    # ---------------------------------------------------------- playback ----
    def _pooled_model(self, i: int) -> CrabModel:
        """Model for actor i (0 = the view's original crab)."""
        if i == 0:
            return self.crab
        while len(self._pool) < i:
            model = CrabModel(floor_texture=self._floor_tex)
            model.set_floor_rgba(self._pool_floor or None)
            self._pool.append(model)
        return self._pool[i - 1]

    def set_actors(self, actors: list[dict]) -> None:
        """Build the cast. Each dict: {table, channels?, xy?, spin?}.

        xy is the stage offset (multi-person mode lines crabs up along the
        axis across the camera view). Extra models come from the pool; the
        camera pulls back so the whole line-up fits.
        """
        self.actors = [
            _Actor(crab=self._pooled_model(i),
                   table=a["table"],
                   xy=tuple(a.get("xy", (0.0, 0.0))),
                   channels=a.get("channels"))
            for i, a in enumerate(actors)
        ]
        # Dropping actors (multi -> single): release the now-idle pooled
        # models' renderers. They hold real GPU contexts; leaving six of
        # them around froze the preview on some machines (the renderers
        # rebuild lazily on the next render, so a later re-cast is fine).
        for model in self._pool[len(self.actors) - 1:]:
            try:
                if model._renderer is not None:
                    model._renderer.close()
                model._renderer = None
            except Exception:  # noqa: BLE001 - best effort
                pass
        self.table = self.actors[0].table if self.actors else None
        self._channels = self.actors[0].channels if self.actors else None
        self._last_t = None
        self._yaw_prev = None
        for a in self.actors:
            a.yaw_prev = None
        # Frame the whole stage: pull back proportionally to the spread.
        span = max((abs(a.xy[0]) + abs(a.xy[1]) for a in self.actors),
                   default=0.0)
        want = float(np.clip(0.48 * (1.0 + 1.6 * span / STAGE_SPREAD),
                             0.48, 1.1))
        if len(self.actors) > 1:
            self.camera.distance = want
        elif self.actors:
            # Back to a single crab: restore the close-up framing the
            # multi-mode pull-back left behind.
            self.camera.distance = float(np.clip(self.camera.distance, 0.3, 0.55))
        # Park ghost bodies on every model so stray arms never show up.
        for a in self.actors:
            if a.channels is None:
                a.crab.set_ghost_pose(None)
        self._repaint_soon()

    def clear_actors(self) -> None:
        self.actors = []
        self.table = None
        self._channels = None

    def set_table(self, table: CrabCommandTable) -> None:
        """Single-actor convenience API (tests / single-person mode)."""
        self.set_actors([{"table": table}])

    def set_channels(self, channels) -> None:
        """Human channels for actor 0; drives the ghost-arm overlay."""
        self._channels = channels
        if self.actors:
            self.actors[0].channels = channels

    def set_ghost_visible(self, visible: bool) -> None:
        if visible == self._ghost_visible:
            return
        self._ghost_visible = visible
        # Frame the ghost too: lift the lookat, pull the camera back a bit.
        if visible:
            self.camera.lookat[2] = 0.16
            self.camera.distance = float(
                np.clip(self.camera.distance * 1.42, 0.2, 1.5))
        else:
            self.camera.lookat[:] = self._LOOKAT_BASE
            self.camera.distance = float(
                np.clip(self.camera.distance / 1.42, 0.2, 1.5))
            self.crab.set_ghost_pose(None)  # park bodies out of sight
        self._repaint_soon()

    def set_spin(self, spin) -> None:
        """Spin source selection, applied to every actor: False/None = yaw
        curve; True = all scripted turns; ndarray = per-frame enable mask."""
        self._spin = spin
        for a in self.actors:
            a.spin = spin
            a.yaw_prev = None
        self._repaint_soon()

    def set_jump(self, jump) -> None:
        """Jump gating for every actor: True keeps all detected hops, an
        ndarray is a per-frame enable mask, False/None mutes them."""
        self._jump = jump
        for a in self.actors:
            a.jump = jump
        self._repaint_soon()

    def set_height_factor(self, factor: float) -> None:
        """User height bias multiplying every commanded base_z."""
        self._height_factor = float(
            np.clip(factor, HEIGHT_FACTOR_MIN, HEIGHT_FACTOR_MAX))
        self._repaint_soon()

    def _z_command(self, z: float) -> float:
        """Apply the height bias and keep the IK-reachable envelope.

        Upper bound = the measured height where the walking legs still
        reach the floor (~1.6x standing), so the tallest slider setting
        keeps the feet planted.
        """
        return float(np.clip(z * self._height_factor,
                             0.5 * self.crab.z_stand, self.crab.z_max_ik))

    # -------------------------------------------------------- ghost arms ----
    def _ghost_segments(self, t: float) -> dict | None:
        """Mocap body placements reconstructing both human arms in 3D.

        Uses the same shoulder pitch/roll/elbow channels the claw mapping
        consumes, plus the commanded yaw: at yaw 0 the figure faces the
        camera (the crab's reference direction), and it turns with the
        crab. Horizontal axis mirrored to match the dancer's on-screen
        appearance (user-calibrated; negate d_h to flip).
        """
        ch = self._channels
        if not self._ghost_visible or ch is None or not len(ch.times):
            return None

        zh = np.array([0.0, 0.0, 1.0])
        d_h = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        d_f = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)   # toward camera

        # Turn the ghost with the crab's commanded yaw.
        psi = np.radians(float((self._live or {}).get("yaw") or 0.0))
        c, s = np.cos(psi), np.sin(psi)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        d_h, d_f = rot @ d_h, rot @ d_f

        center = self._GHOST_CENTER + np.array([0.0, 0.0, self.crab.z_stand * 0.25])
        L = self._GHOST_LEN

        head = center + zh * 0.075
        s_l = center + 0.05 * d_h
        s_r = center - 0.05 * d_h
        segs = {"ghost_head": (head, head), "ghost_bar": (s_l, s_r)}

        for side, shoulder in (("l", s_l), ("r", s_r)):
            part = side
            tt = np.clip(t, ch.times[0], ch.times[-1])
            pitch = float(np.nan_to_num(
                np.interp(tt, ch.times, getattr(ch, f"shoulder_pitch_{side}")),
                nan=0.0))
            roll = float(np.nan_to_num(
                np.interp(tt, ch.times, getattr(ch, f"shoulder_roll_{side}")),
                nan=0.0))
            # Video bend is 0 straight / 180 folded; the ghost renderer
            # takes 180 straight / 0 folded.
            bend = 180.0 - float(np.nan_to_num(
                np.interp(tt, ch.times, getattr(ch, f"bend_{side}")),
                nan=0.0))
            # Shoulder pitch = depth raise (toward camera), roll = signed
            # lateral raise; the roll channel is + = image right while d_h
            # points screen-left, hence the mirrored construction (shared
            # with the manual ghost mode via ghost_arm_points).
            elbow, wrist = ghost_arm_points(
                pitch, roll, bend, shoulder, L, d_h, d_f, zh)
            segs[f"ghost_{part}u"] = (shoulder, elbow)
            segs[f"ghost_{part}f"] = (elbow, wrist)
        return segs

    @property
    def live_commands(self) -> dict | None:
        """Last command dict painted (drives the mapping overlay values)."""
        return self._live

    def idle_tick(self, t: float) -> None:
        """Stand-and-wave animation shown before any video is loaded."""
        if self._probe is not None or self._manual is not None:
            return  # probe / manual-ghost modes drive their own poses
        signs = self._probe_signs()
        claw = self._rest_claw_pose()
        z = self.crab.z_stand * 1.06 * (1.0 + 0.012 * np.sin(2 * np.pi * 0.45 * t))
        roll = 1.8 * np.sin(2 * np.pi * 0.45 * t + 1.2)
        wave = 0.5 + 0.45 * np.sin(2 * np.pi * 0.9 * t)

        claw["LF_J1_joint"] += signs["LF"]["lift"] * 1.15 * wave
        claw["LF_J4_joint"] += signs["LF"]["open"] * (0.25 + 0.3 * wave)

        self.crab.pose_dance(self._z_command(z), roll, 0.0, 0.0, claw,
                             yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG)
        self._live = {"base_z": float(z), "roll": float(roll),
                      "pitch": 0.0, "yaw": 0.0, "leg_sway": 0.0, "claw": claw}
        self._last_t = None
        self._paint_frame()

    def _probe_signs(self):
        if not hasattr(self, "_signs"):
            self._signs = self.crab.calibrate_claw_signs()
        return self._signs

    def _rest_claw_pose(self) -> dict:
        """Claw pose at rest with the mapping's static biases applied:
        mirror-symmetric at HOME (J0 bias 0), arm plane levelled (J1 bias),
        J3 outward bias."""
        signs = self._probe_signs()
        claw = {name: value for name, value in HOME.items()
                if name.startswith(("LF", "RF"))}
        for leg, spread in (("LF", +1.0), ("RF", -1.0)):
            claw[f"{leg}_J0_joint"] -= signs[leg]["ccw"] * np.radians(
                CLAW_J0_OFFSET_DEG)
            # Anti-crossing spread (user): LF +, RF -.
            claw[f"{leg}_J0_joint"] += spread * np.radians(
                CLAW_J0_SPREAD_DEG)
            claw[f"{leg}_J1_joint"] -= np.radians(CLAW_J1_OFFSET_DEG)
            claw[f"{leg}_J3_joint"] -= signs[leg]["fold"] * np.radians(
                CLAW_J3_OFFSET_DEG)
        return claw

    # ------------------------------------------------- joint probe mode ----
    def show_probe(self, claw_deltas: dict | None) -> None:
        """Calibration pose: both claws at rest plus per-joint deltas.

        claw_deltas maps joint name -> delta rad (added to the rest pose,
        clamped into range); the probe dialog's ten sliders build it.
        None ends probe mode. While a video is loaded the ghost arms (the
        recognised human skeleton) stay live so the user can compare the
        posed claws against the dancer's actual arms.
        """
        if claw_deltas is None:
            self._probe = None
            # Restore the ghost visibility the probe took over.
            if getattr(self, "_probe_ghost_saved", None) is not None:
                self._ghost_visible = self._probe_ghost_saved
                self._probe_ghost_saved = None
            self._repaint_soon()
            return
        self._probe = dict(claw_deltas)
        if getattr(self, "_probe_ghost_saved", None) is None:
            self._probe_ghost_saved = self._ghost_visible
            self._ghost_visible = True   # show the human reference arms
        self._probe_frame(self._last_t if self._last_t is not None else 0.0)

    def _probe_frame(self, t: float) -> None:
        claw = self._rest_claw_pose()
        for name, delta in self._probe.items():
            if name in claw:
                claw[name] = self.crab.clamp(name, claw[name] + delta)
        self.crab.set_ghost_pose(self._ghost_segments(t))
        self.crab.pose_dance(self._z_command(self.crab.z_stand * 1.02),
                             0.0, 0.0, 0.0, claw,
                             yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG)
        self._live = {"base_z": self.crab.z_stand, "roll": 0.0, "pitch": 0.0,
                      "yaw": 0.0, "leg_sway": 0.0, "claw": claw}
        self._paint_frame()

    # ------------------------------------------------- manual ghost mode ----
    # The user poses the ghost arms with SLIDERS (shoulder pitch/roll euler
    # + elbow bend, per side); the crab claws follow through the SAME IK
    # solver the video table uses (solve_claw_ik_pose), so the claw always
    # lands in the world direction the ghost shows -- the tool for judging
    # reachability and joint limits against the ghost, no video needed.
    _MANUAL_TORSO = 0.10   # ghost shoulder-bar span, normalises side/up

    def show_manual_ghost(self, on: bool) -> None:
        """Enter/leave manual ghost-arm control (probe tool, second mode)."""
        if on and self._manual is None:
            self._manual_ghost_saved = self._ghost_visible
            self._ghost_visible = True
            self._manual = {"angles": self.ghost_angles_at(
                self._last_t if self._last_t is not None else 0.0),
                "channels": {}, "override": None}
        elif not on and self._manual is not None:
            self._manual = None
            if getattr(self, "_manual_ghost_saved", None) is not None:
                self._ghost_visible = self._manual_ghost_saved
                self._manual_ghost_saved = None
            if not self._ghost_visible:
                self.crab.set_ghost_pose(None)
        self._repaint_soon()

    def _manual_basis(self) -> tuple:
        """Fixed ghost frame at yaw 0 (mirrored to match the dancer)."""
        zh = np.array([0.0, 0.0, 1.0])
        d_h = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)   # screen-left axis
        d_f = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)  # toward camera
        center = self._GHOST_CENTER + np.array(
            [0.0, 0.0, self.crab.z_stand * 0.25])
        return zh, d_h, d_f, center

    def ghost_angles_at(self, t: float) -> dict:
        """Ghost shoulder pitch/roll + elbow bend per side at time t.

        The video-synced channels when they exist, else a neutral hang
        (slightly out, elbows bent). Slider entry pose + reset both read
        this."""
        ch = self._channels
        out = {}
        for side in ("l", "r"):
            pitch, roll, bend = 12.0, 25.0 if side == "l" else -25.0, 100.0
            if ch is not None and len(ch.times):
                tt = np.clip(t, ch.times[0], ch.times[-1])
                pitch = float(np.nan_to_num(np.interp(
                    tt, ch.times, getattr(ch, f"shoulder_pitch_{side}")),
                    nan=0.0))
                roll = float(np.nan_to_num(np.interp(
                    tt, ch.times, getattr(ch, f"shoulder_roll_{side}")),
                    nan=0.0))
                bend = 180.0 - float(np.nan_to_num(np.interp(
                    tt, ch.times, getattr(ch, f"bend_{side}")), nan=0.0))
            out[side] = (pitch, roll, bend)
        return out

    def set_manual_ghost_angles(self, angles: dict) -> None:
        """Slider drive: {"l": (pitch, roll, bend), "r": (...)} in degrees."""
        if self._manual is None:
            return
        self._manual["angles"] = {side: tuple(map(float, angles[side]))
                                  for side in ("l", "r")}
        self._manual_frame()

    @property
    def manual_angles(self) -> dict:
        """Live {side: (pitch_deg, roll_deg, bend_deg)} (probe UI)."""
        return (self._manual or {}).get("angles", {})

    def set_manual_claw_override(self, claw: dict | None) -> None:
        """Pose the claws directly while keeping the manual ghost visible.

        None returns to the ghost-driven IK. Used by the keyframe
        calibration mode, where the user matches both sides by hand.
        """
        if self._manual is None:
            return
        self._manual["override"] = None if claw is None else {
            name: float(v) for name, v in claw.items()
            if name in self.crab.joint_adr}
        self._manual_frame()

    def _manual_frame(self) -> None:
        """Pose the crab from the slider ghost-arm angles through the same
        direction-tracking IK the video table uses; also publishes the
        live channels."""
        zh, d_h, d_f, center = self._manual_basis()
        s_l = center + 0.05 * d_h
        s_r = center - 0.05 * d_h
        head = center + zh * 0.075
        segs = {"ghost_head": (head, head), "ghost_bar": (s_l, s_r)}
        channels = {}
        for side, shoulder in (("l", s_l), ("r", s_r)):
            pitch, roll, bend = self._manual["angles"][side]
            # Side swing is mirror-antisymmetric: the right ghost arm at
            # +roll must mirror the left at +roll (user: the right arm's
            # swing direction was reversed). The claw SOLVE keeps the
            # slider value as-is so the user anchor (90/90/180) stays exact.
            seg_roll = -roll if side == "r" else roll
            elbow, wrist = ghost_arm_points(
                pitch, seg_roll, bend, shoulder, self._GHOST_LEN, d_h, d_f, zh)
            channels[side] = arm_channels_from_points(
                shoulder, elbow, wrist, center, d_h, zh, self._MANUAL_TORSO)
            segs[f"ghost_{side}u"] = (shoulder, elbow)
            segs[f"ghost_{side}f"] = (elbow, wrist)
        self._manual["channels"] = channels
        self.crab.set_ghost_pose(segs)

        # Pose the body first (the IK anchors live on the shoulders), then
        # solve the claws for the slider angles with the SAME staged IK as
        # the video table -- including the user-anchor calibration offset,
        # so (90, 90, 180) lands on the pose the user specified.
        z = self._z_command(self.crab.z_stand * 1.02)
        claw = self._rest_claw_pose()
        self.crab.pose_dance(z, 0.0, 0.0, 0.0, claw,
                             yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG)
        ik = solve_claw_ik_pose(self.crab, self._manual["angles"])
        for leg_out in ik.values():
            claw.update(leg_out)
        # Anti-crossing spread (user): LF J0 +, RF J0 -.
        for leg, sgn in (("LF", 1.0), ("RF", -1.0)):
            claw[f"{leg}_J0_joint"] += sgn * np.radians(CLAW_J0_SPREAD_DEG)
        # Usable shoulder-yaw limits: keep the arms clear of the leg links.
        for leg in ("LF", "RF"):
            lo, hi = CLAW_J0_LIMIT_DEG[leg]
            claw[f"{leg}_J0_joint"] = float(np.clip(
                claw[f"{leg}_J0_joint"], np.radians(lo), np.radians(hi)))
            name = f"{leg}_J1_joint"
            claw[name] = float(np.clip(
                claw[name], np.radians(CLAW_J1_MIN_DEG),
                self.crab.joint_range[name][1]))
        # J4 (gripper) is out of the mapping (user): rest value.
        for leg in ("LF", "RF"):
            claw[f"{leg}_J4_joint"] = HOME[f"{leg}_J4_joint"]
        # Same no-cross window as the video table, so a keyframe correction
        # computed against this base lines up with the runtime mapping.
        cross = np.radians(CLAW_J0_CROSS_LIMIT_DEG)
        claw["LF_J0_joint"] = max(claw["LF_J0_joint"], -cross)
        claw["RF_J0_joint"] = min(claw["RF_J0_joint"], cross)
        override = (self._manual or {}).get("override")
        if override:
            for name, value in override.items():
                if name in claw:
                    claw[name] = self.crab.clamp(name, value)
        claw = clip_claw_joints(claw, self.crab.joint_range)
        self.crab.pose_dance(z, 0.0, 0.0, 0.0, claw,
                             yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG)
        self._live = {"base_z": self.crab.z_stand, "roll": 0.0, "pitch": 0.0,
                      "yaw": 0.0, "leg_sway": 0.0, "claw": claw}
        self._paint_frame()

    @property
    def manual_channels(self) -> dict:
        """Live {side: {forearm, bend, side, up}} while posing (probe UI)."""
        return (self._manual or {}).get("channels", {})

    def _paint_frame(self) -> None:
        """Render the posed crab(s) into the label (shared code path).

        Failures auto-retry a few times (transient GPU/context hiccups);
        once the budget is spent the label switches to a click-to-retry
        state that also offers the captured traceback for diagnosis.
        """
        if self._render_dead:
            return
        if self._render_error:
            self._render_retries += 1
            if self._render_retries > _RENDER_RETRIES:
                self._render_dead = True
                # Keep the label's geometry stable: a cleared pixmap would
                # shrink its size hint and the splitter would reflow.
                self._label.setText("")
                if self._label.pixmap().isNull():
                    ph = QPixmap(self._label.size().expandedTo(
                        self._label.minimumSize()))
                    ph.fill(QColor("#0d1117"))
                    self._label.setPixmap(ph)
                self._retry_link.setGeometry(self._label.geometry())
                self._retry_link.raise_()
                self._retry_link.setVisible(True)
                return
        try:
            w = max(self._label.width(), 320)
            h = max(self._label.height(), 240)
            # Single crab on the plain backdrop renders directly; anything
            # else (gradient preset, several crabs) goes through compositing.
            if len(self.actors) <= 1 and self._bg["id"] == BG_DEFAULT:
                model = self.actors[0].crab if self.actors else self.crab
                frame = model.render(camera=self.camera, width=w, height=h)
                self._hit_boxes = []
            else:
                order = sorted(self.actors, key=self._depth_key,
                               reverse=True)
                layers = []
                boxes = []
                if not order:
                    # Idle crab (no video) still needs compositing on a
                    # gradient / custom backdrop.
                    frame, mask, _bbox = self.crab.render_with_mask(
                        camera=self.camera, width=w, height=h)
                    layers.append((frame, mask))
                for i, a in enumerate(order):
                    # Farthest layer carries the (tinted) floor; nearer
                    # layers carry only their crab, or each one's floor
                    # plane would erase everything painted before it.
                    frame, mask, bbox = a.crab.render_with_mask(
                        camera=self.camera, width=w, height=h,
                        robot_only=(i > 0))
                    layers.append((frame, mask))
                    if bbox:
                        boxes.append((a, bbox))
                frame = self._composite(layers)
                self._hit_boxes = boxes  # painter order: far -> near
                frame = self._draw_badges(frame, boxes)
        except Exception as exc:  # noqa: BLE001 - surfaced via the retry UI
            import traceback
            self._render_error = True
            self._render_detail = (f"{type(exc).__name__}: {exc}\n"
                                   + traceback.format_exc())
            print(self._render_detail, file=sys.stderr)
            self._label.setText(f"3D 渲染重试中… ({self._render_retries + 1}"
                                f"/{_RENDER_RETRIES})")
            return
        # Success: clear the retry state.
        if self._render_error or self._retry_link.isVisible():
            self._render_error = False
            self._render_retries = 0
            self._render_detail = ""
            self._retry_link.setVisible(False)
        img = QImage(frame.data, frame.shape[1], frame.shape[0],
                     frame.strides[0], QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(img)
        scaled = self._pixmap.scaled(
            self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        if self._overlay.isVisible():
            self._overlay.update()

    def _draw_badges(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """Number badge over each crab in multi mode (matches the video
        panel's person N labels)."""
        if len(self.actors) < 2:
            return frame
        img = QImage(frame.data, frame.shape[1], frame.shape[0],
                     frame.strides[0], QImage.Format.Format_RGB888).copy()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for a, (x0, y0, x1, y1) in boxes:
            number = self.actors.index(a) + 1
            cx = (x0 + x1) // 2
            cy = y0 - 14
            r = 11
            p.setPen(QPen(QColor(230, 180, 80), 2))
            p.setBrush(QColor(13, 17, 23, 215))
            p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
            p.setPen(QColor(240, 246, 252))
            f = QFont("Microsoft YaHei", 9)
            f.setBold(True)
            p.setFont(f)
            p.drawText(cx - r, cy - r - 1, 2 * r, 2 * r,
                       Qt.AlignmentFlag.AlignCenter, str(number))
        p.end()
        h, w = img.height(), img.width()
        out = np.asarray(img.constBits(), dtype=np.uint8).reshape(
            h, img.bytesPerLine())[:, :w * 3].reshape(h, w, 3).copy()
        return out

    def _depth_key(self, a: _Actor) -> float:
        """Painter's-order key: how far the actor sits toward the camera.

        MuJoCo parks the camera at the given azimuth from the lookat
        point, so the projection of (actor_xy - lookat_xy) onto the
        camera direction ranks near/far; farthest renders first.
        """
        az = np.radians(self.camera.azimuth)
        d = np.array([np.cos(az), np.sin(az)])
        return float(np.asarray(a.xy) @ d)

    # --------------------------------------------------------- background ----
    def set_background(self, bg_id: str) -> None:
        preset = next((p for p in CRAB_BACKGROUNDS if p["id"] == bg_id),
                      CRAB_BACKGROUNDS[0])
        self._bg = preset
        if bg_id != "custom":
            self._bg_image = None
        # Leaving a custom floor: rebuild WITHOUT the texture, otherwise the
        # imported image stays on the plane and "clear" looks broken.
        if self._floor_tex is not None:
            self._floor_tex = None
            self._pool_floor = []
            self._rebuild_models()
        self.crab.set_floor_rgba(preset.get("floor"))
        # Pooled crabs get the same floor so multi-mode stays uniform.
        self._pool_floor = list(preset["floor"]) if preset.get("floor") else []
        for model in self._pool:
            model.set_floor_rgba(preset.get("floor"))
        self._grad_key = None
        self._repaint_soon()

    def set_custom_background(self, image_path: str | None,
                              floor_path: str | None) -> None:
        """User-imported backdrop image + floor texture.

        The backdrop composites like the gradient presets (cover-cropped).
        The floor image becomes a MuJoCo plane texture, which needs every
        model rebuilt (~0.3 s each) -- actors keep their tables and stage
        slots, only the models are swapped.
        """
        self._bg_image = None
        if image_path:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                self._bg_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._bg = {"id": "custom", "name": "自定义"}
        # A textured floor owns its own colour; a leftover preset tint from a
        # previous background would multiply it (and can wash it to white).
        self._pool_floor = []

        need_rebuild = False
        if floor_path and Path(floor_path).exists():
            png = CACHE_DIR / "user_floor.png"
            img = cv2.imdecode(np.fromfile(floor_path, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                if img.shape[0] > 1024 or img.shape[1] > 1024:
                    k = 1024 / max(img.shape[:2])
                    img = cv2.resize(img, None, fx=k, fy=k)
                cv2.imwrite(str(png), img)   # mujoco textures need PNG
                self._floor_tex = png
                need_rebuild = True
        elif self._floor_tex is not None:
            self._floor_tex = None
            need_rebuild = True
        if need_rebuild:
            self._rebuild_models()
        if self._floor_tex is not None:
            self.crab.set_floor_rgba(None)
            for model in self._pool:
                model.set_floor_rgba(None)
        self._grad_key = None
        self._repaint_soon()

    def _rebuild_models(self) -> None:
        """Recreate every CrabModel with the current floor texture."""
        self.crab.close()
        for m in self._pool:
            m.close()
        self.crab = CrabModel(floor_texture=self._floor_tex)
        self._pool = []
        if hasattr(self, "_signs"):
            del self._signs  # re-probe on the new model
        if self._bg.get("id") != "custom":
            self.crab.set_floor_rgba(self._bg.get("floor"))
            self._pool_floor = list(self._bg["floor"]) if self._bg.get("floor") else []
        for i, a in enumerate(self.actors):
            a.crab = self._pooled_model(i)
            a.yaw_prev = None
        self._hit_boxes = []
        self._last_render = 0.0

    def set_mapping_visible(self, visible: bool) -> None:
        self._overlay.setVisible(visible)

    def close(self) -> None:
        """Release every model's GPU renderer (main + pool)."""
        self.crab.close()
        for model in self._pool:
            model.close()

    def _repaint_soon(self) -> None:
        self._last_render = 0.0
        if self._manual is not None:
            self._manual_frame()
        elif self._last_t is not None:
            self.show_time(self._last_t, force=True)
        # else the idle timer repaints within 120 ms

    def _composite(self, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Blend rendered crab layers back-to-front over the backdrop
        (user-imported image if set, else the preset's vertical gradient)."""
        h, w = layers[0][0].shape[:2]
        if self._grad_key != (self._bg["id"], w, h):
            if self._bg_image is not None:
                bh, bw = self._bg_image.shape[:2]
                scale = max(w / bw, h / bh)          # cover-crop to frame
                nw, nh = max(int(np.ceil(w / scale)), 2), max(int(np.ceil(h / scale)), 2)
                resized = cv2.resize(self._bg_image, (nw, nh))
                x0, y0 = (nw - w) // 2, (nh - h) // 2
                self._grad = resized[y0:y0 + h, x0:x0 + w].astype(np.float32)
            else:
                top = np.asarray(self._bg.get("top", (13, 17, 23)), dtype=np.float32)
                bot = np.asarray(self._bg.get("bottom", (22, 27, 34)), dtype=np.float32)
                t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
                self._grad = top[None, None, :] * (1.0 - t) + bot[None, None, :] * t
            self._grad_key = (self._bg["id"], w, h)
        out = self._grad.copy()
        for frame, mask in layers:
            alpha = cv2.blur(mask.astype(np.float32), (5, 5))[..., None]
            out = frame.astype(np.float32) * alpha + out * (1.0 - alpha)
        return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)

    def show_time(self, t: float, force: bool = False) -> None:
        """Pose the crab(s) at time t; re-render throttled to CRAB_RENDER_FPS.

        In probe mode the crabs freeze at the probe pose but the ghost arms
        keep following t, so the human skeleton stays in sync with the
        playing video for comparison.
        """
        if self._probe is None and self._manual is None and not self.actors:
            return
        now = time.monotonic()
        if not force and (now - self._last_render) < self._min_render_interval:
            return
        self._last_render = now
        self._last_t = t
        if self._probe is not None:
            self._probe_frame(t)
            return
        if self._manual is not None:
            self._manual_frame()   # frozen at the user's dragged pose
            return

        for i, a in enumerate(self.actors):
            cmd = sample_commands(a.table, t, spin=a.spin, jump=a.jump)
            # Nearest-angle yaw tracking: the scripted turn arrays return
            # to the plain curve after a full sweep (a raw ~360 deg step).
            # Rotating the shortest way renders that as a no-move instead
            # of a whip, and also smooths seeks / segment toggles.
            yaw = cmd["yaw"]
            if a.yaw_prev is not None:
                yaw = a.yaw_prev + wrap180(yaw - a.yaw_prev)
            a.yaw_prev = yaw
            # Hop script: the crouch dip (发力) lowers the GROUNDED height
            # (inside the IK clip), the airborne rise is added on top so a
            # jump can leave the IK envelope and the feet leave the floor.
            hop_m = JUMP_HOP_GAIN * a.crab.z_stand * cmd["jump"]
            z = self._z_command(cmd["base_z"] + min(hop_m, 0.0)) \
                + max(hop_m, 0.0)
            cmd = {**cmd, "yaw": yaw, "base_z": z}
            if i == 0:
                self._live = cmd
                # Ghost mocap first: pose_dance's mj_forward bakes it in.
                a.crab.set_ghost_pose(self._ghost_segments(t))
            a.crab.pose_dance(
                z, cmd["roll"], cmd["pitch"],
                yaw, cmd["claw"], leg_sway=cmd["leg_sway"],
                yaw_offset_deg=CRAB_FRONT_YAW_OFFSET_DEG,
                foot_lift=cmd["foot_lift"], base_xy=a.xy,
            )
        self._paint_frame()

    # ------------------------------------------------------------ camera ----
    def _orbit(self, dx: float, dy: float) -> None:
        self.camera.azimuth -= 0.4 * dx
        self.camera.elevation = float(
            np.clip(self.camera.elevation - 0.4 * dy, -80.0, 10.0))
        if self._last_t is not None:
            self._last_render = 0.0  # bypass the throttle
            self.show_time(self._last_t, force=True)

    def _zoom(self, steps: float) -> None:
        self.camera.distance = float(
            np.clip(self.camera.distance * (1.0 - 0.1 * steps), 0.2, 1.5))
        if self._last_t is not None:
            self._last_render = 0.0
            self.show_time(self._last_t, force=True)

    # ------------------------------------------------------- crab drag ----
    def _render_pos(self, pos) -> tuple[float, float] | None:
        """Widget position -> render-pixel coordinates (undo letterbox)."""
        if self._pixmap is None:
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        lw, lh = self._label.width(), self._label.height()
        scale = min(lw / pw, lh / ph)
        ox = (lw - pw * scale) / 2.0
        oy = (lh - ph * scale) / 2.0
        return (pos.x() - ox) / scale, (pos.y() - oy) / scale

    def _actor_at(self, pos):
        """Nearest actor under the position (None = background)."""
        if len(self.actors) < 2 or not self._hit_boxes:
            return None
        xy = self._render_pos(pos)
        if xy is None:
            return None
        x, y = xy
        for a, (x0, y0, x1, y1) in reversed(self._hit_boxes):
            if x0 - 6 <= x <= x1 + 6 and y0 - 6 <= y <= y1 + 26:
                return a  # include the badge area above the crab
        return None

    def _drag_delta(self, dx_px: float, dy_px: float) -> np.ndarray:
        """Screen-pixel drag -> stage-plane world delta (m).

        Screen-x maps onto the camera's right axis, screen-y onto the view
        axis (dragging down pulls the crab toward the viewer). Scale from
        the ~45 deg vertical FOV; near enough that dragging feels 1:1.
        """
        ph = self._pixmap.height() if self._pixmap else self._label.height()
        m_per_px = (2.0 * math.tan(math.radians(22.5))
                    * max(self.camera.distance, 0.15) / max(ph, 1))
        az = math.radians(self.camera.azimuth)
        right = np.array([math.cos(az - math.pi / 2),
                          math.sin(az - math.pi / 2)])
        forward = np.array([math.cos(az), math.sin(az)])
        return dx_px * m_per_px * right - dy_px * m_per_px * forward

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._press_pos = (event.position().x(), event.position().y())
        self._drag_actor = self._actor_at(event.position())
        if self._drag_actor is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is None:
            over = self._actor_at(event.position()) is not None
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if over
                else Qt.CursorShape.ArrowCursor)
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            x, y = event.position().x(), event.position().y()
            dx, dy = x - self._press_pos[0], y - self._press_pos[1]
            self._press_pos = (x, y)
            if self._drag_actor is not None:
                # Move this crab across the stage; playback keeps running.
                delta = self._drag_delta(dx, dy)
                a = self._drag_actor
                a.xy = (a.xy[0] + float(delta[0]),
                        a.xy[1] + float(delta[1]))
                self._last_render = 0.0
                if self._last_t is not None:
                    self.show_time(self._last_t, force=True)
                return
            self._orbit(dx, dy)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._press_pos = None
        self._drag_actor = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        self._zoom(event.angleDelta().y() / 120.0)
