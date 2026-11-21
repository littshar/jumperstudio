"""Left panel: the imported video with switchable detection overlays."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget,
)

from app.core.retarget import SKELETON_EDGES

_BONES = [(a, b) for a, b in SKELETON_EDGES]
_BONE_COLOR = (80, 220, 100)     # BGR
_JOINT_COLOR = (60, 140, 255)
_BOX_COLOR = (255, 190, 60)
_BOX_OTHER = (120, 140, 160)     # non-followed people in multi-person clips


class VideoPanel(QWidget):
    """Video frames by time, with independently toggled overlays.

    The person bounding box and the skeleton are two separate switches
    (persisted in settings); both read the same keypoints. Toggling while
    paused repaints the current frame immediately.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel("把舞蹈视频拖到窗口任意位置", alignment=Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(320, 240)
        self._label.setObjectName("videoSurface")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label, stretch=1)

        # Analysis console shown over the import area while Sports2D runs:
        # the same tqdm bar VSCode prints, plus the pipeline stage messages.
        self._progress_box = QWidget(self)
        self._progress_box.setStyleSheet(
            "background:#0b0f14; border:1px solid #21262d; border-radius:8px;")
        play = QVBoxLayout(self._progress_box)
        play.setContentsMargins(12, 10, 12, 10)
        title = QLabel("姿态识别")
        title.setStyleSheet("color:#e6b450; font-weight:600;")
        self._progress_anim = QProgressBar()
        self._progress_anim.setRange(0, 0)   # indeterminate while parsing
        self._progress_anim.setTextVisible(False)
        self._progress_anim.setMaximumHeight(6)
        self._progress_line = QLabel("准备…")
        self._progress_line.setWordWrap(True)
        self._progress_line.setStyleSheet(
            "color:#61afef; font-family:Consolas,'Microsoft YaHei';")
        self._progress_log = QPlainTextEdit()
        self._progress_log.setReadOnly(True)
        self._progress_log.setMaximumBlockCount(300)
        self._progress_log.setStyleSheet(
            "QPlainTextEdit { background:#0b0f14; color:#9fb2c8;"
            " border:none; font-family:Consolas,'Microsoft YaHei'; }")
        play.addWidget(title)
        play.addWidget(self._progress_anim)
        play.addWidget(self._progress_line)
        play.addWidget(self._progress_log, stretch=1)
        self._progress_box.setVisible(False)
        self._progress_last = ""

        self.show_detect = False    # person bounding box overlay
        self.show_skeleton = True   # full skeleton overlay

        self._cap = None
        self._last_idx = -1
        self._last_t = 0.0
        self.fps = 30.0
        self.frame_count = 0
        self.duration = 0.0
        self.frame_w = 0
        self.frame_h = 0
        self._kps = None
        self._pixmap = None
        self._persons = []          # list[Keypoints]; _kps is person 0
        self._follow = 0            # index of the followed person (0-based)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._progress_box.setGeometry(self._label.geometry())

    # ---------------------------------------------------------- progress ----
    def begin_progress(self) -> None:
        self._progress_last = ""
        self._progress_line.setText("准备…")
        self._progress_log.clear()
        self._progress_box.setGeometry(self._label.geometry())
        self._progress_box.show()
        self._progress_box.raise_()

    def append_progress(self, text: str) -> None:
        text = (text or "").strip()
        if not text or text == self._progress_last:
            return
        self._progress_last = text
        if "%" in text or "|" in text or "it/s" in text:
            self._progress_line.setText(text)
            return
        self._progress_line.setText(text)
        self._progress_log.appendPlainText(text)
        bar = self._progress_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def end_progress(self) -> None:
        self._progress_box.setVisible(False)

    # ------------------------------------------------------------- video ----
    def open_video(self, path: str) -> bool:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self._cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 1:
            self.fps = 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = (self.frame_count / self.fps) if self.fps else 0.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._last_idx = -1
        self.show_time(0.0)
        return True

    def set_keypoints(self, kps) -> None:
        self._kps = kps
        self._persons = [kps]
        self._follow = 0

    def set_persons(self, persons: list, follow: int | None = 0) -> None:
        """All detected people + which one the crab follows.

        follow is 0-based; None (multi-crab mode) highlights everyone.
        """
        self._persons = persons or []
        if follow is None:
            self._follow = -1  # everyone is somebody's crab
        else:
            self._follow = max(0, min(follow, len(self._persons) - 1))
        idx = max(self._follow, 0)
        self._kps = self._persons[idx] if self._persons else None
        self._repaint()

    def close_video(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------ overlays --
    def set_detect_visible(self, visible: bool) -> None:
        self.show_detect = visible
        self._repaint()

    def set_skeleton_visible(self, visible: bool) -> None:
        self.show_skeleton = visible
        self._repaint()

    def _repaint(self) -> None:
        """Redraw the current frame now (so paused toggles take effect)."""
        if self._cap is not None and self._last_idx >= 0:
            self.show_time(self._last_t)

    # -------------------------------------------------------------- frame ----
    def _grab(self, idx: int) -> np.ndarray | None:
        if self._cap is None:
            return None
        idx = max(0, min(idx, max(self.frame_count - 1, 0)))
        if idx != self._last_idx + 1:  # need a real seek
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._last_idx = idx
        return frame

    def _draw_overlay(self, frame: np.ndarray, t: float) -> None:
        if not (self.show_detect or self.show_skeleton):
            return
        multi = len(self._persons) > 1
        persons = self._persons if multi else [self._kps] if self._kps else []
        for idx, kps in enumerate(persons):
            pts = self._keypoints_at(t, kps)
            if not pts:
                continue
            self._draw_person(frame, pts, idx,
                              followed=(not multi or self._follow < 0
                                        or idx == self._follow),
                              numbered=multi)

    def _keypoints_at(self, t: float, kps=None) -> dict[str, tuple[float, float]] | None:
        kps = kps if kps is not None else self._kps
        kps = kps if kps is not None else self._kps
        if (kps is None or kps.times is None
                or len(kps.times) == 0):
            return None
        idx = int(np.clip(np.searchsorted(kps.times, t),
                          0, len(kps.times) - 1))
        pts = {}
        for name, arr in kps.xyz.items():
            x, y = arr[idx]
            pts[name] = (float(x), float(y))
        return pts

    def _draw_person(self, frame: np.ndarray, pts: dict, idx: int,
                     followed: bool, numbered: bool) -> None:
        h, w = frame.shape[:2]
        scale = self.frame_w / w if w else 1.0

        def px(p):
            return int(p[0] / scale), int(p[1] / scale)

        valid = [p for p in pts.values()
                 if not any(np.isnan(v) for v in p)]
        if not valid:
            return

        box_color = _BOX_COLOR if followed else _BOX_OTHER
        if self.show_detect:
            xs = [p[0] / scale for p in valid]
            ys = [p[1] / scale for p in valid]
            x0, x1 = max(int(min(xs)) - 14, 0), min(int(max(xs)) + 14, w - 1)
            y0, y1 = max(int(min(ys)) - 14, 0), min(int(max(ys)) + 14, h - 1)
            thick = 2 if followed else 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, thick, cv2.LINE_AA)
            label = f"person {idx + 1}" if numbered else "person"
            cv2.putText(frame, label, (x0, max(y0 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

        if self.show_skeleton:
            for a, b in _BONES:
                pa, pb = pts.get(a), pts.get(b)
                if pa is None or pb is None:
                    continue
                if any(np.isnan(v) for v in (*pa, *pb)):
                    continue
                cv2.line(frame, px(pa), px(pb), _BONE_COLOR, 2, cv2.LINE_AA)

        for p in valid:
            cv2.circle(frame, px(p), 3, _JOINT_COLOR, -1, cv2.LINE_AA)

    def show_time(self, t: float) -> None:
        if self._cap is None:
            return
        idx = int(round(t * self.fps))
        frame = self._grab(idx)
        if frame is None:
            return
        self._last_t = t
        self._draw_overlay(frame, t)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                     rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(img)
        self._update_label()

    def _update_label(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._update_label()
