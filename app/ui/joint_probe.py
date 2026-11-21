"""Claw joint probe dialog: pose the claws joint by joint, by slider.

Two modes (user request, both slider-driven -- no mode buttons beyond the
mode switch itself):
  * 关节扫动 -- ten sliders, left claw and right claw separated, one per
    claw joint J0..J4 (degrees relative to the calibrated rest pose, within
    the joint's real range). Watch both claws on the preview and read the
    live J0-J4 angles in the manual readout conventions.
  * 手动幽灵 -- pose the ghost arms with sliders; the crab claws follow
    the video mapping. The right-hand claw sliders can override the claws
    freely, and "记录关键帧" stores matched ghost/crab poses that become a
    piecewise mapping curve (>=2 keyframes). Live J0-J4 readout per claw.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from app.constants import HOME
from app.core import claw_calib

# Manual-ghost sliders: (side, field, label, lo, hi)
# Side-swing ranges are per side (user): the left arm's outward arc lies on
# the negative side, the right arm's on the positive side.
_ANGLE_SPECS = (
    ("l", "pitch", "左肩 前后", -90, 180),
    ("l", "roll", "左肩 侧摆", -120, 0),
    ("l", "bend", "左肘 弯曲", 0, 180),
    ("r", "pitch", "右肩 前后", -90, 180),
    ("r", "roll", "右肩 侧摆", -120, 0),
    ("r", "bend", "右肘 弯曲", 0, 180),
)
_SWEEP, _MANUAL = 0, 1
_SWEEP_WINDOW_DEG = 180.0   # cap on |delta| sliders offer, for drag precision


def _slider_row(label: str, lo: int, hi: int, value: int, on_change,
                label_width: int = 230) -> tuple[QHBoxLayout, QSlider, QLabel]:
    """label + slider + live value; returns (row, slider, value_label)."""
    name = QLabel(label)
    name.setStyleSheet("color:#8b949e;")
    name.setMinimumWidth(label_width)
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(value)
    s.valueChanged.connect(on_change)
    val = QLabel(f"{value:+d}°")
    val.setStyleSheet("color:#61afef;")
    val.setMinimumWidth(48)
    row = QHBoxLayout()
    row.addWidget(name)
    row.addWidget(s, stretch=1)
    row.addWidget(val)
    return row, s, val


class JointProbeDialog(QDialog):
    """Non-modal; drives CrabView.show_probe / show_manual_ghost."""

    manual_changed = Signal(bool)
    calib_changed = Signal()

    def __init__(self, crab_view, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关节映射")
        self.setModal(False)
        self.setMinimumWidth(820)
        self._view = crab_view
        self._mode = _SWEEP

        # mode switch
        self._mode_sweep = QPushButton("关节角度")
        self._mode_manual = QPushButton("识别映射")
        for b in (self._mode_sweep, self._mode_manual):
            b.setCheckable(True)
            b.setObjectName("modeBtn")
        self._mode_sweep.setChecked(True)
        self._mode_sweep.clicked.connect(lambda: self._set_mode(_SWEEP))
        self._mode_manual.clicked.connect(lambda: self._set_mode(_MANUAL))
        row_mode = QHBoxLayout()
        row_mode.addWidget(self._mode_sweep)
        row_mode.addWidget(self._mode_manual)
        row_mode.addStretch(1)

        # ---- sweep panel: ten sliders, LF and RF separated ---------------
        # Deltas are degrees from the rest pose, bounded by each joint's
        # real range (windowed for drag precision).
        self._sweep_sliders: dict[tuple[str, int], QSlider] = {}
        self._sweep_labels: dict[tuple[str, int], QLabel] = {}
        self._sweep_panel = QWidget()
        slay = QVBoxLayout(self._sweep_panel)
        slay.setContentsMargins(0, 0, 0, 0)
        slay.setSpacing(2)
        for leg, head in (("LF", "左钳 LF"), ("RF", "右钳 RF")):
            h = QLabel(head)
            h.setStyleSheet("color:#e6b450; font-weight:600;")
            slay.addWidget(h)
            for j in range(5):
                name = f"{leg}_J{j}_joint"
                lo_r, hi_r = self._view.crab.joint_range[name]
                home = HOME[name]
                lo = int(max(np.degrees(lo_r - home), -_SWEEP_WINDOW_DEG))
                hi = int(min(np.degrees(hi_r - home), _SWEEP_WINDOW_DEG))
                row, s, val = _slider_row(f"J{j}", lo, hi, 0, self._on_sweep,
                                          label_width=36)
                slay.addLayout(row)
                self._sweep_sliders[(leg, j)] = s
                self._sweep_labels[(leg, j)] = val
        self.sweep_reset = QPushButton("全部归零")
        self.sweep_reset.clicked.connect(self._reset_sweep)
        slay.addWidget(self.sweep_reset)

        # ---- manual ghost panel: one slider per ghost-arm euler angle ----
        self._angle_sliders: dict[tuple[str, str], QSlider] = {}
        self._angle_labels: dict[tuple[str, str], QLabel] = {}
        self._manual_panel = QWidget()
        mlay = QVBoxLayout(self._manual_panel)
        mlay.setContentsMargins(0, 0, 0, 0)
        mlay.setSpacing(2)
        self._side_headers: dict[str, QLabel] = {}
        for side, field, label, lo, hi in _ANGLE_SPECS:
            if side not in self._side_headers:
                head = QLabel("左臂" if side == "l" else "右臂")
                head.setStyleSheet("color:#e6b450; font-weight:600;")
                self._side_headers[side] = head
                mlay.addWidget(head)
            row, s, val = _slider_row(label, lo, hi, 0, self._on_angles)
            mlay.addLayout(row)
            self._angle_sliders[(side, field)] = s
            self._angle_labels[(side, field)] = val
        self.reset_btn = QPushButton("↩ 回到视频当前姿势")
        self.reset_btn.clicked.connect(self._reset_angles)
        mlay.addWidget(self.reset_btn)
        self._manual_panel.setVisible(False)

        self._frames = claw_calib.load_frames()
        self._calib_box = QWidget()
        clay = QVBoxLayout(self._calib_box)
        clay.setContentsMargins(0, 4, 0, 0)
        self._calib_info = QLabel()
        self._calib_info.setWordWrap(True)
        self._calib_info.setStyleSheet("color:#8b949e;")
        record = QPushButton("记录关键帧")
        record.setObjectName("accentBtn")
        record.clicked.connect(self._record_frame)
        follow = QPushButton("重新跟随幽灵")
        follow.clicked.connect(self._follow_ghost)
        undo = QPushButton("删除最后一帧")
        undo.clicked.connect(self._undo_frame)
        clear = QPushButton("清除校准")
        clear.clicked.connect(self._clear_frames)
        crow = QHBoxLayout()
        crow.addWidget(record)
        crow.addWidget(follow)
        crow.addWidget(undo)
        crow.addWidget(clear)
        crow.addStretch(1)
        clay.addLayout(crow)
        clay.addWidget(self._calib_info)
        self._calib_box.setVisible(False)

        self.live_lab = QLabel("LF —")
        self.live_lab.setStyleSheet("color:#61afef;")
        self.live_lab.setWordWrap(True)
        self.tip = QLabel()
        self.tip.setWordWrap(True)
        self.tip.setStyleSheet("color:#8b949e;")
        self._set_tip()

        panels = QHBoxLayout()
        panels.addWidget(self._manual_panel, stretch=1)
        panels.addWidget(self._sweep_panel, stretch=1)
        lay = QVBoxLayout(self)
        lay.addLayout(row_mode)
        lay.addLayout(panels)
        lay.addWidget(self._calib_box)
        # Calibration box, live angle readout and tip are hidden for now.
        self.live_lab.setVisible(False)
        self.tip.setVisible(False)

    # ------------------------------------------------------------ driving ----
    def _set_tip(self) -> None:
        # Explanatory text removed; keep the sliders themselves.
        self.tip.setText("")

    def _set_mode(self, mode: int) -> None:
        self._mode = mode
        manual = mode == _MANUAL
        self._mode_sweep.setChecked(not manual)
        self._mode_manual.setChecked(manual)
        # Keyframe calibration is temporarily disabled; the manual mode keeps
        # only the ghost sliders driving the claws.
        self._sweep_panel.setVisible(not manual)
        self._manual_panel.setVisible(manual)
        self._calib_box.setVisible(False)
        if manual:
            self._view.show_probe(None)
            self._view.show_manual_ghost(True)
            self._view.set_manual_claw_override(None)
            self._sync_sliders()
            if self._view.manual_angles:
                self._view.set_manual_ghost_angles(self._view.manual_angles)
            self._update_live()
        else:
            self._view.set_manual_claw_override(None)
            self._view.show_manual_ghost(False)
            self._on_sweep()
        self._set_tip()
        self._update_calib_info()
        self.manual_changed.emit(manual)

    # -------------------------------------------------------- sweep ui ----
    def _on_sweep(self, *args) -> None:
        """Any sweep slider moved -> pose the claws directly."""
        deltas = {}
        for (leg, j), s in self._sweep_sliders.items():
            self._sweep_labels[(leg, j)].setText(f"{s.value():+d}°")
            deltas[f"{leg}_J{j}_joint"] = np.radians(s.value())
        if self._mode == _MANUAL:
            self._apply_calib_override()
        else:
            self._view.show_probe(deltas)
        self._update_live()

    def _reset_sweep(self) -> None:
        for s in self._sweep_sliders.values():
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
        for key, lab in self._sweep_labels.items():
            lab.setText("+0°")
        self._on_sweep()

    # -------------------------------------------------------- manual ui ----
    def _sync_sliders(self) -> None:
        """Sliders <- the view's live manual angles (entry pose / reset)."""
        angles = self._view.manual_angles
        for (side, field), s in self._angle_sliders.items():
            val = angles.get(side, (0, 0, 0))[("pitch", "roll", "bend").index(field)]
            s.blockSignals(True)
            s.setValue(int(round(float(val))))
            s.blockSignals(False)
            self._angle_labels[(side, field)].setText(f"{float(val):+.0f}°")

    def _on_angles(self, *args) -> None:
        """Any slider moved -> push all six angles to the ghost + claws."""
        if self._mode != _MANUAL:
            return
        for key, s in self._angle_sliders.items():
            self._angle_labels[key].setText(f"{s.value():+d}°")
        fields = ("pitch", "roll", "bend")
        angles = {
            side: tuple(self._angle_sliders[(side, f)].value() for f in fields)
            for side in ("l", "r")
        }
        self._view.set_manual_ghost_angles(angles)
        self._update_live()

    def _reset_angles(self) -> None:
        """Back to the video-synced pose at the current playhead."""
        view = self._view
        t = view._last_t if view._last_t is not None else 0.0
        view.set_manual_ghost_angles(view.ghost_angles_at(t))
        self._sync_sliders()
        self._update_live()

    def _sync_sweep_to_live(self) -> None:
        """Crab sliders <- the claw pose the mapping currently produces."""
        live = (self._view.live_commands or {}).get("claw") or {}
        rest = self._view._rest_claw_pose()
        for (leg, j), slider in self._sweep_sliders.items():
            name = f"{leg}_J{j}_joint"
            v = live.get(name)
            if v is None:
                continue
            deg = int(round(np.degrees(v - rest[name])))
            slider.blockSignals(True)
            slider.setValue(max(slider.minimum(),
                                min(slider.maximum(), deg)))
            slider.blockSignals(False)
            self._sweep_labels[(leg, j)].setText(f"{slider.value():+d}°")

    def _calib_target(self) -> dict:
        rest = self._view._rest_claw_pose()
        return {
            f"{leg}_J{j}_joint": self._view.crab.clamp(
                f"{leg}_J{j}_joint",
                rest[f"{leg}_J{j}_joint"] + np.radians(slider.value()))
            for (leg, j), slider in self._sweep_sliders.items()
        }

    def _apply_calib_override(self) -> None:
        if self._view.manual_angles:
            self._view.set_manual_claw_override(self._calib_target())

    def _follow_ghost(self) -> None:
        """Clear the manual claw override and resume mapping-driven pose."""
        if self._mode != _MANUAL:
            return
        self._view.set_manual_claw_override(None)
        if self._view.manual_angles:
            self._view.set_manual_ghost_angles(self._view.manual_angles)
        self._sync_sweep_to_live()
        self._update_live()

    def _record_frame(self) -> None:
        ghost = {
            side: tuple(float(self._angle_sliders[(side, f)].value())
                        for f in ("pitch", "roll", "bend"))
            for side in ("l", "r")
        }
        target = self._calib_target()
        # Temporarily let the mapping pose the claws to capture the base it
        # produces for this ghost; the correction is target - base.
        self._view.set_manual_claw_override(None)
        base = dict((self._view.live_commands or {}).get("claw") or {})
        if not base:
            return
        self._view.set_manual_claw_override(target)
        self._frames.append(claw_calib.make_record(ghost, target, base))
        claw_calib.save_frames(self._frames)
        self._update_calib_info()
        self.calib_changed.emit()

    def _undo_frame(self) -> None:
        if not self._frames:
            return
        self._frames.pop()
        claw_calib.save_frames(self._frames)
        self._update_calib_info()
        self.calib_changed.emit()

    def _clear_frames(self) -> None:
        self._frames = []
        claw_calib.clear()
        self._update_calib_info()
        self.calib_changed.emit()

    def _update_calib_info(self) -> None:
        if not self._frames:
            self._calib_info.setText("记录关键帧即可加偏置")
            return
        n = len(self._frames)
        lines = [f"关键帧 {n} 个 · 偏置已生效"]
        for i in range(max(1, n - 3), n):
            text = claw_calib.ratio_text(self._frames, i - 1, i)
            if text:
                lines.append(f"{i}→{i + 1} · {text}")
        self._calib_info.setText("\n".join(lines))

    def _update_live(self) -> None:
        live = self._view.live_commands or {}
        claw = live.get("claw") or {}
        if self._mode == _MANUAL:
            # Report DELTAS from the rest pose -- the same convention as
            # the 关节扫动 sliders, so the user's anchor reads back verbatim.
            rest = self._view._rest_claw_pose()
            lines = []
            for leg in ("LF", "RF"):
                parts = []
                for j in range(5):
                    name = f"{leg}_J{j}_joint"
                    v = claw.get(name)
                    parts.append(f"J{j} {np.degrees(v - rest[name]):+6.1f}°"
                                 if v is not None else f"J{j} —")
                lines.append(f"{leg}: " + "  ".join(parts))
            self.live_lab.setText("\n".join(lines) if claw else "—")
            return
        # Sweep mode: the sliders carry the values; show absolute commands.
        parts = []
        for leg in ("LF", "RF"):
            vals = []
            for j in range(5):
                v = claw.get(f"{leg}_J{j}_joint")
                vals.append(f"{np.degrees(v):+6.1f}°" if v is not None else "—")
            parts.append(f"{leg} " + " ".join(vals))
        self.live_lab.setText("\n".join(parts) if claw else "LF —")

    # ---------------------------------------------------------------- exit ----
    def closeEvent(self, event) -> None:  # noqa: N802
        self._view.set_manual_claw_override(None)
        self._view.show_manual_ghost(False)
        self._view.show_probe(None)
        self.manual_changed.emit(False)
        super().closeEvent(event)
