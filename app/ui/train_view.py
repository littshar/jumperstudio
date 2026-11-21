"""Secondary page: whole-body motion tracking training (Jumper PPO / local ES).

Reached from the main page's top-right button; replaces the whole page.
Contents: big render toggle, TWO separate render panes (physics training
simulation vs. the static policy preview), a big progress bar, the live
training LOG (replaces the old parameter column), reward curves, the
parameter dialog (settability + resource estimation) and the export-weights
button that only enables once training finished.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

import cv2
import matplotlib
import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics, QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QPlainTextEdit, QProgressBar, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from app.core.crab_model import CrabModel
from app.core.jumper_train import (
    JumperPreviewWorker, JumperTrainWorker, jumper_status, latest_run_dir,
    newest_checkpoint, pretrained_checkpoint,
)

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from app.core.train_worker import TrainParams, TrainWorker  # noqa: E402
from app.ui.train_params_dialog import TrainParamsDialog  # noqa: E402
from app.ui.train_history_dialog import TrainHistoryDialog  # noqa: E402

_BG = "#0d1117"
_FG = "#8b949e"


class RewardCanvas(FigureCanvasQTAgg):
    """Top: normalised return; bottom: the main reward components."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 2.6), facecolor=_BG,
                             constrained_layout=True)
        super().__init__(self.figure)
        self.ax_ret = self.figure.add_subplot(2, 1, 1)
        self.ax_comp = self.figure.add_subplot(2, 1, 2)
        self._iters: list[int] = []
        self._hist: list[dict] = []
        self.mode = "es"          # "es" (local trial) or "jumper" (GPU PPO)
        self._style_axes()
        self._placeholder()
        self.setMinimumHeight(170)

    def _style_axes(self) -> None:
        for ax in (self.ax_ret, self.ax_comp):
            ax.set_facecolor(_BG)
            ax.tick_params(colors=_FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#21262d")
            ax.grid(alpha=0.2, color=_FG)
        self.ax_ret.set_ylabel("归一化回报", color=_FG)
        self.ax_comp.set_ylabel("奖励分量", color=_FG)
        self.ax_comp.set_xlabel("训练轮数", color=_FG)
        self.ax_ret.set_ylim(0.0, 1.0)
        self.ax_comp.set_ylim(0.0, 1.0)

    def _placeholder(self) -> None:
        for ax, text in ((self.ax_ret, "归一化回报"),
                         (self.ax_comp,
                          "跟踪 / 稳定 / 双手距离 / 平滑 / 存活 / yaw / 后腿触地 / 落地")):
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                    va="center", fontsize=10, color="#586e88")
            ax.set_xticks([])
            ax.set_yticks([])

    def _placeholder_jumper(self) -> None:
        for ax, text in ((self.ax_ret,
                          "归一化回报"),
                         (self.ax_comp,
                          "跟踪质量")):
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                    va="center", fontsize=10, color="#586e88")
            ax.set_xticks([])
            ax.set_yticks([])

    #: Jumper/mjlab's reward is NOT normalised: the eight positive tracking
    #: terms sum to 6.5, the manager scales every step by dt=0.02 s, and an
    #: episode is 500 steps. So the raw cumulative return tops out near
    #: 6.5*0.02*500 = 65, and a raw "50" is ~77% of full marks, not 50/100.
    _JUMPER_STEP_CEIL = 6.5 * 0.02
    _JUMPER_EPISODE_CEIL = 6.5 * 0.02 * 500.0

    def _add_jumper(self, rec: dict) -> None:
        """Normalised PPO metrics: return vs. the theoretical ceiling + tracking."""
        self._iters.append(int(rec["iter"]))
        self._hist.append(rec)
        self.ax_ret.clear()
        self.ax_comp.clear()
        self._style_axes()
        x = self._iters
        ceil_ep = self._JUMPER_EPISODE_CEIL
        ceil_step = self._JUMPER_STEP_CEIL
        self.ax_ret.plot(
            x, [h.get("mean_reward", 0.0) / ceil_ep for h in self._hist],
            color="#61afef", lw=1.4, label="累计回报 / 满分")
        self.ax_ret.plot(
            x, [(h.get("mean_reward", 0.0)
                 / max(h.get("ep_len", 1.0), 1.0)) / ceil_step
                for h in self._hist],
            color="#e5c07b", lw=1.0, label="单步回报 / 上限")
        leg = self.ax_ret.legend(loc="lower right", fontsize=7,
                                 framealpha=0.3)
        for t in leg.get_texts():
            t.set_color("#c9d1d9")
        self.ax_ret.set_ylabel("归一化回报", color=_FG)
        self.ax_ret.set_ylim(-0.05, 1.05)
        self.ax_comp.plot(
            x, [1.0 / (1.0 + max(h.get("joint_err", 1.0), 0.0))
                for h in self._hist],
            color="#61afef", lw=1.0, label="关节位置")
        self.ax_comp.plot(
            x, [1.0 / (1.0 + 10.0 * max(h.get("body_err", 0.1), 0.0))
                for h in self._hist],
            color="#e5c07b", lw=1.0, label="身体位置")
        self.ax_comp.plot(
            x, [min(max(h.get("ep_len", 0.0) / 500.0, 0.0), 1.0)
                for h in self._hist],
            color="#98c379", lw=1.0, label="存活时长")
        leg = self.ax_comp.legend(loc="lower right", fontsize=7, ncol=3,
                                  framealpha=0.3)
        for t in leg.get_texts():
            t.set_color("#c9d1d9")
        self.ax_comp.set_ylabel("跟踪质量", color=_FG)
        self.ax_comp.set_xlabel("训练轮数", color=_FG)
        self.ax_comp.set_ylim(0.0, 1.0)
        if len(x) == 1:
            for ax in (self.ax_ret, self.ax_comp):
                ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
        self.draw_idle()

    def reset(self) -> None:
        self._iters.clear()
        self._hist.clear()
        for ax in (self.ax_ret, self.ax_comp):
            ax.clear()
        self._style_axes()
        if self.mode == "jumper":
            self.ax_ret.set_ylabel("归一化回报", color=_FG)
            self.ax_ret.set_ylim(-0.05, 1.05)
            self.ax_comp.set_ylabel("跟踪质量", color=_FG)
            self._placeholder_jumper()
        else:
            self._placeholder()
        self.draw_idle()

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self.reset()

    def add(self, rec: dict) -> None:
        if self.mode == "jumper":
            self._add_jumper(rec)
            return
        self._iters.append(int(rec["iter"]))
        self._hist.append(rec)
        self.ax_ret.clear()
        self.ax_comp.clear()
        self._style_axes()
        x = self._iters
        self.ax_ret.plot(x, [h["return_best"] for h in self._hist],
                         color="#98c379", lw=1.6, label="最优回报")
        self.ax_ret.plot(x, [h["return_mean"] for h in self._hist],
                         color="#61afef", lw=1.0, label="种群均值")
        leg = self.ax_ret.legend(loc="lower right", fontsize=7,
                                 framealpha=0.3)
        for t in leg.get_texts():
            t.set_color("#c9d1d9")
        self.ax_ret.set_ylabel("归一化回报", color=_FG)
        self.ax_ret.set_ylim(0.0, 1.0)
        terms = (("track", "跟踪", "#61afef"), ("stab", "稳定", "#e5c07b"),
                 ("hands", "双手距离", "#e06c75"), ("smooth", "平滑", "#c678dd"),
                 ("survive", "存活", "#98c379"), ("yaw", "yaw", "#56b6c2"),
                 ("rear", "后腿触地", "#d19a66"), ("land", "落地姿态", "#e6b450"))
        for key, label, color in terms:
            self.ax_comp.plot(x, [h[key] for h in self._hist], color=color,
                              lw=1.0, label=label)
        leg = self.ax_comp.legend(loc="lower right", fontsize=6, ncol=4,
                                  framealpha=0.3)
        for t in leg.get_texts():
            t.set_color("#c9d1d9")
        self.ax_comp.set_ylabel("奖励分量", color=_FG)
        self.ax_comp.set_xlabel("训练轮数", color=_FG)
        self.ax_comp.set_ylim(0.0, 1.0)
        if len(x) == 1:
            for ax in (self.ax_ret, self.ax_comp):
                ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
        self.draw_idle()


class _OrbitLabel(QLabel):
    """Image label with mouse orbit (drag) + zoom (wheel) signals."""

    dragged = Signal(float, float)
    wheeled = Signal(float)

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._last = None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._last is not None:
            pos = event.position().toPoint()
            delta = pos - self._last
            if delta.x() or delta.y():
                self.dragged.emit(float(delta.x()), float(delta.y()))
            self._last = pos
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._last = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.wheeled.emit(event.angleDelta().y() / 120.0)
        event.accept()


def _render_pane(caption: str, orbit: bool = False
                 ) -> tuple[QWidget, QLabel]:
    """Caption + image label used for the three render panes."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    cap = QLabel(caption)
    cap.setStyleSheet("color:#8b949e; font-weight:600;")
    if orbit:
        img = _OrbitLabel("等待训练…")
        img.setToolTip("拖动旋转 · 滚轮缩放")
    else:
        img = QLabel("等待训练…")
    img.setAlignment(Qt.AlignmentFlag.AlignCenter)
    img.setMinimumSize(260, 180)
    img.setStyleSheet(
        "background:#0f1319; border:1px solid #21262d; border-radius:8px;"
        "color:#586e88;")
    lay.addWidget(cap)
    lay.addWidget(img, stretch=1)
    return box, img


class TrainPage(QWidget):
    """The full-page training UI (replaces the main page while shown)."""

    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table = None
        self._meta: dict = {}
        self._worker: TrainWorker | None = None
        self._jumper: JumperTrainWorker | None = None
        self._preview_worker: JumperPreviewWorker | None = None
        self._run_dir: str | None = None
        # Newest checkpoint of the run this page started/resumed. Start after
        # Stop continues it; importing a new dance clears it.
        self._continue_ckpt: str | None = None
        self._stop_pending = False
        self._auto_preview_after_stop = False
        self._backend = "jumper"
        self._weights: str | None = None
        self._weights_kind = "npz"
        self._frame_count = 0
        self._jump_segs: list[tuple[float, float]] = []
        self._params = TrainParams()
        # Tests may inject a tiny TrainParams here (kept out of the UI).
        self._params_override: TrainParams | None = None
        # Post-training full-dance physics playback (policy preview pane).
        self._dance: dict | None = None
        self._dance_crab: CrabModel | None = None
        self._dance_t0 = 0.0
        self._dance_paused = False
        self._dance_elapsed = 0.0
        self._dance_timer = QTimer(self)
        self._dance_timer.setInterval(40)   # 25 fps
        self._dance_timer.timeout.connect(self._tick_dance)
        # Source video pane (synchronised with the physics playback).
        self._cap = None
        self._cap_path: str | None = None
        self._cap_fps = 30.0
        self._cap_idx = -1
        # Recorded policy replay (SuperMimic GPU backend): an MP4 from
        # Jumper's play.py, shown in the policy pane instead of qpos frames.
        self._policy_cap = None
        self._policy_path: str | None = None
        self._policy_fps = 30.0
        self._policy_count = 0
        self._policy_idx = -1
        self._policy_offset = 0.0
        # Interactive cameras for the two MuJoCo panes (az, el, dist).
        self._phys_cam = [135.0, -16.0, 0.60]
        self._policy_cam = [135.0, -16.0, 0.60]

        # ---- header
        self.back_btn = QPushButton("返回")
        self.back_btn.clicked.connect(self.back_requested.emit)
        title = QLabel("动作训练")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color:#f0f6fc;")
        self.export_btn = QPushButton("导出")
        self.export_btn.setObjectName("accentBtn")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_weights)
        head = QHBoxLayout()
        head.addWidget(self.back_btn)
        head.addSpacing(8)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.export_btn)

        # ---- big render toggle + two independent render panes
        self.render_toggle = QCheckBox("实时画面")
        self.render_toggle.setToolTip("显示训练实时画面")
        self.render_toggle.setChecked(True)
        self.render_toggle.setStyleSheet(
            "QCheckBox { font-size: 16px; font-weight: 600; spacing: 12px; }"
            "QCheckBox::indicator { width: 26px; height: 26px; }")
        self.render_toggle.toggled.connect(self._on_render_toggled)
        physics_box, self.physics_img = _render_pane("训练渲染", orbit=True)
        preview_box, self.preview_img = _render_pane("策略预览", orbit=True)
        physics_box.setToolTip("拖动旋转 · 滚轮缩放")
        preview_box.setToolTip("拖动旋转 · 滚轮缩放")
        self.physics_img.dragged.connect(self._orbit_physics)
        self.physics_img.wheeled.connect(self._zoom_physics)
        self.preview_img.dragged.connect(self._orbit_policy)
        self.preview_img.wheeled.connect(self._zoom_policy)
        self.preview_pause_btn = QPushButton("暂停")
        self.preview_pause_btn.setCheckable(True)
        self.preview_pause_btn.setEnabled(False)
        self.preview_pause_btn.setFixedHeight(24)
        self.preview_pause_btn.clicked.connect(self._toggle_preview_pause)
        preview_box.layout().addWidget(self.preview_pause_btn)
        panes = QHBoxLayout()
        panes.addWidget(physics_box, stretch=1)
        panes.addWidget(preview_box, stretch=1)
        left = QVBoxLayout()
        left.addWidget(self.render_toggle)
        left.addLayout(panes, stretch=1)

        # ---- right column: ONE window reused -- training log while a run
        # is going, the source video when reviewing the result. Two small tabs
        # switch by hand; the page also switches itself (log at start,
        # video when the replay is ready).
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1200)
        self.log.setPlaceholderText("训练日志…")
        self.log.setStyleSheet(
            "QPlainTextEdit { background:#0b0f14; color:#9fb2c8;"
            " border:1px solid #21262d; border-radius:8px;"
            " font-family:Consolas,'Microsoft YaHei'; }")
        self.log.setFont(QFont("Consolas", 9))
        self._log_fit_tick = 0
        self.video_img = QLabel("训练结束后在这里与策略回放同步显示原视频")
        self.video_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_img.setMinimumSize(260, 180)
        self.video_img.setStyleSheet(
            "QLabel { background:#0f1319; border:1px solid #21262d;"
            " border-radius:8px; color:#586e88; }")
        video_page = QWidget()
        video_lay = QVBoxLayout(video_page)
        video_lay.setContentsMargins(0, 0, 0, 0)
        video_lay.setSpacing(4)
        video_cap = QLabel("原视频")
        video_cap.setStyleSheet("color:#8b949e;")
        video_lay.addWidget(video_cap)
        video_lay.addWidget(self.video_img, stretch=1)

        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self.log)         # 0: log
        self.right_stack.addWidget(video_page)       # 1: source video
        self.log_tab = QPushButton("日志")
        self.video_tab = QPushButton("原视频")
        tab_style = (
            "QPushButton { background:#161b22; color:#8b949e;"
            " border:1px solid #30363d; border-radius:6px;"
            " padding:2px 8px; font-size:11px; }"
            "QPushButton:checked { background:#1f6feb; color:#ffffff; }")
        for b in (self.log_tab, self.video_tab):
            b.setCheckable(True)
            b.setStyleSheet(tab_style)
        self.log_tab.setChecked(True)
        self.log_tab.clicked.connect(lambda: self._show_right_page(0))
        self.video_tab.clicked.connect(lambda: self._show_right_page(1))
        tabs = QHBoxLayout()
        tabs.addWidget(self.log_tab)
        tabs.addWidget(self.video_tab)
        tabs.addStretch(1)
        right_box = QGroupBox("日志 / 原视频")
        right_box.setMinimumWidth(380)
        right_lay = QVBoxLayout(right_box)
        right_lay.addLayout(tabs)
        right_lay.addWidget(self.right_stack, stretch=1)
        right = QVBoxLayout()
        right.addWidget(right_box)

        top = QHBoxLayout()
        top.addLayout(left, stretch=3)
        top.addLayout(right, stretch=2)

        # ---- big progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMinimumHeight(18)
        self.progress.setStyleSheet(
            "QProgressBar { background:#161b22; border:1px solid #30363d;"
            " border-radius:9px; }"
            "QProgressBar::chunk { background:#1f6feb; border-radius:9px; }")
        self.progress_pct = QLabel("0%")
        self.progress_pct.setStyleSheet(
            "font-size: 15px; font-weight: 700; color:#61afef;")
        self.progress_pct.setFixedWidth(64)
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress, stretch=1)
        prog_row.addWidget(self.progress_pct)

        # ---- curves
        self.rewards = RewardCanvas()

        # ---- backend selector (which trainer the start button runs)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("PPO", "jumper")
        self.backend_combo.addItem("Smoke", "es")
        ok, why = jumper_status()
        if not ok:
            idx = self.backend_combo.findData("jumper")
            self.backend_combo.model().item(idx).setEnabled(False)
            self.backend_combo.setItemData(idx, why, Qt.ItemDataRole.ToolTipRole)
            self.backend_combo.setCurrentIndex(1)
            self._backend = "es"
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)

        # ---- footer
        self.params_btn = QPushButton("参数")
        self.params_btn.clicked.connect(self._open_params)
        self.start_btn = QPushButton("开始")
        self.start_btn.setObjectName("accentBtn")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.replay_btn = QPushButton("回放")
        self.replay_btn.setEnabled(False)
        self.replay_btn.setToolTip("播放完整策略回放")
        self.replay_btn.clicked.connect(self._start_dance_playback)
        self.quick_preview_btn = QPushButton("快速预览")
        self.quick_preview_btn.setEnabled(False)
        self.quick_preview_btn.setToolTip("用最后检查点录制 8 秒回放")
        self.quick_preview_btn.clicked.connect(self._quick_preview)
        self.history_btn = QPushButton("训练历史")
        self.history_btn.setToolTip("本工程的训练记录")
        self.history_btn.clicked.connect(self._open_train_history)
        self.status_label = QLabel("等待导入动作轨道")
        self.status_label.setStyleSheet("color:#8b949e;")
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("训练方法："))
        backend_row.addWidget(self.backend_combo)
        backend_row.addStretch(1)
        backend_row.addWidget(self.status_label)

        foot = QHBoxLayout()
        foot.addWidget(self.start_btn)
        foot.addWidget(self.params_btn)
        foot.addWidget(self.stop_btn)
        foot.addWidget(self.quick_preview_btn)
        foot.addWidget(self.replay_btn)
        foot.addWidget(self.history_btn)
        foot.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        lay.addLayout(head)
        lay.addLayout(top, stretch=3)
        lay.addLayout(backend_row)
        lay.addLayout(prog_row)
        lay.addWidget(self.rewards, stretch=2)
        lay.addLayout(foot)
        self._on_backend_changed(self.backend_combo.currentIndex())

    # ---------------------------------------------------------------- log ----
    def _log(self, text: str) -> None:
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")
        # Stick to the newest line from the very first message -- the log is
        # a live feed, not something to keep dragging down. Moving the cursor
        # also forces QPlainTextEdit to lay out the appended block before the
        # scrollbar maximum is read.
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.ensureCursorVisible()
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._log_fit_tick += 1
        if self._log_fit_tick % 4 == 0:
            self._fit_log_font()

    def _fit_log_font(self) -> None:
        """Shrink/grow the log font so the longest recent line fits the pane."""
        width = max(80, self.log.viewport().width() - 18)
        lines = self.log.toPlainText().splitlines()[-40:]
        if not lines:
            return
        font = self.log.font()
        metrics = QFontMetrics(font)
        longest = max(metrics.horizontalAdvance(line) for line in lines)
        if longest <= 0:
            return
        point = font.pointSizeF()
        if point <= 0:
            point = 10.0
        target = max(6.0, min(10.5, point * width / float(longest)))
        if abs(target - point) >= 0.4:
            font.setPointSizeF(target)
            self.log.setFont(font)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_log_fit_tick"):
            self._fit_log_font()

    def _open_train_history(self) -> None:
        TrainHistoryDialog(self).exec()

    def _show_right_page(self, index: int) -> None:
        """Switch the shared right window between the log and the video."""
        self.right_stack.setCurrentIndex(0 if index == 0 else 1)
        self.log_tab.setChecked(index == 0)
        self.video_tab.setChecked(index == 1)

    # ------------------------------------------------------------ data in ----
    def set_reference(self, table, meta: dict) -> None:
        """Feed the recognised joint track (+ jumps) into the page."""
        self.stop_training()
        self._stop_dance_playback()
        self._release_policy_video()
        self._dance = None
        self._dance_paused = False
        self.preview_pause_btn.blockSignals(True)
        self.preview_pause_btn.setChecked(False)
        self.preview_pause_btn.setText("暂停")
        self.preview_pause_btn.setEnabled(False)
        self.preview_pause_btn.blockSignals(False)
        self.replay_btn.setEnabled(False)
        self._table = table
        self._meta = dict(meta or {})
        self._weights = None
        self._run_dir = None
        self._continue_ckpt = None
        self.export_btn.setEnabled(False)
        self.rewards.reset()
        self.progress.setValue(0)
        self.progress_pct.setText("0%")
        self.log.clear()
        times = np.asarray(table.times)
        self._frame_count = len(times)
        duration = float(times[-1] - times[0]) if len(times) else 0.0
        self._jump_segs = self._jump_segments(table)
        seg_txt = "、".join(f"{a:.1f}–{b:.1f}s" for a, b in self._jump_segs[:4])
        self._log(
            f"已导入动作轨道：{self._meta.get('video_name', '—')} · "
            f"{self._meta.get('person', '')} · {self._frame_count} 帧 / "
            f"{duration:.1f} s")
        self._log(
            f"跳跃段 {len(self._jump_segs)} 个"
            + (f"：{seg_txt}" if seg_txt else "未检测到跳跃"))
        ok, why = jumper_status()
        self._show_right_page(0)
        if not ok:
            self._log(f"PPO 训练环境不可用 · {why}")
        # Checkpoints from an earlier session are enough for the quick preview:
        # after a restart the app can replay the last saved weights directly.
        run = latest_run_dir() if ok else None
        if run is not None:
            self._run_dir = str(run)
        self._apply_backend_ui()
        self._sync_buttons()
        self._open_video_source()

    @staticmethod
    def _jump_segments(table) -> list[tuple[float, float]]:
        mask = getattr(table, "jump_mask", None)
        times = np.asarray(table.times)
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

    # ----------------------------------------------------------- backend ----
    def _on_backend_changed(self, _index: int = -1) -> None:
        self._backend = self.backend_combo.currentData() or "es"
        self._params.method = (
            "jumper_ppo" if self._backend == "jumper" else "es")
        self.rewards.set_mode("jumper" if self._backend == "jumper" else "es")
        self._apply_backend_ui()
        self._sync_buttons()

    def _apply_backend_ui(self) -> None:
        jumper = self._backend == "jumper"
        self.physics_img.setText("训练渲染")
        self.preview_img.setText("策略预览")
        if self._table is None:
            self.status_label.setText("等待导入动作轨道")
        else:
            self.status_label.setText(
                "动作轨道已导入 · 从预训练权重微调" if jumper
                else "动作轨道已导入 · 点击「开始」")

    # ------------------------------------------------------------ params ----
    def _open_params(self) -> None:
        dlg = TrainParamsDialog(self._params, max(self._frame_count, 2),
                                self.render_toggle.isChecked(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._params = dlg.params(self._params)
        self.render_toggle.blockSignals(True)
        self.render_toggle.setChecked(self._params.render)
        self.render_toggle.blockSignals(False)
        # The method chosen in the dialog drives the backend combo.
        data = "jumper" if self._params.method == "jumper_ppo" else "es"
        idx = self.backend_combo.findData(data)
        if idx >= 0 and idx != self.backend_combo.currentIndex():
            self.backend_combo.setCurrentIndex(idx)
        self._log(
            f"参数更新：方法 "
            f"{'PPO' if self._params.method == 'jumper_ppo' else 'Smoke'}"
            f" · 精度 {self._params.precision_deg:.1f}° · "
            f"数量 {self._params.population} · 轮数 {self._params.iterations}"
            f" · 终止 {self._params.height_term_m * 1000:.0f}mm/"
            f"{self._params.tilt_term_deg:.0f}° · 双手安全 "
            f"{self._params.hand_safe_m * 1000:.0f}mm")

    # ------------------------------------------------------------ training ----
    @staticmethod
    def _running(worker) -> bool:
        return bool(worker is not None
                    and getattr(worker, "isRunning", lambda: False)())

    def _busy(self) -> bool:
        return (self._running(self._worker)
                or self._running(self._jumper)
                or self._running(self._preview_worker))

    def start_training(self) -> None:
        if self._table is None:
            self._log("先导入一个动作轨道")
            return
        if self._busy() or self._stop_pending:
            self._log("训练或回放仍在运行，请先停止")
            return
        self._show_right_page(0)
        self._stop_dance_playback()
        self._release_policy_video()
        self._dance = None
        self.preview_img.setPixmap(QPixmap())
        self.physics_img.setPixmap(QPixmap())
        self.preview_pause_btn.blockSignals(True)
        self.preview_pause_btn.setChecked(False)
        self.preview_pause_btn.setText("暂停")
        self.preview_pause_btn.setEnabled(False)
        self.preview_pause_btn.blockSignals(False)
        self._run_dir = None   # a new run will publish its own checkpoint
        self.replay_btn.setEnabled(False)
        self.quick_preview_btn.setEnabled(False)
        self.rewards.reset()
        self.progress.setValue(0)
        self.progress_pct.setText("0%")
        self.export_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.backend_combo.setEnabled(False)
        if self._backend == "jumper":
            self._start_jumper()
        else:
            self._start_es()
        self._sync_buttons()

    def _start_es(self) -> None:
        params = self._params_override or dataclasses.replace(
            self._params, render=self.render_toggle.isChecked())
        self.progress.setRange(0, params.iterations)
        self._log(
            f"开始 Smoke 训练：精度 {params.precision_deg:.1f}° · 数量 "
            f"{params.population} · 轮数 {params.iterations} · 物理子步 "
            f"{params.substeps} · 渲染{'开' if params.render else '关'}")
        worker = TrainWorker(self._table, self._meta, params, self)
        worker.set_camera(*self._phys_cam)
        worker.progress.connect(self._on_progress)
        worker.status.connect(self._on_status)
        worker.metrics.connect(self._on_metrics)
        worker.physics_frame.connect(self._on_physics_frame)
        worker.preview_frame.connect(self._on_preview_frame)
        worker.dance_ready.connect(self._on_dance_ready)
        worker.reference_saved.connect(self._on_reference_saved)
        worker.weights_saved.connect(self._on_weights_saved)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _start_jumper(self) -> None:
        resume = (self._continue_ckpt
                  if self._continue_ckpt and Path(self._continue_ckpt).is_file()
                  else None)
        if resume is None and pretrained_checkpoint() is None:
            self._log(
                "缺少预训练权重 assets/pretrained/dance_base.pt，"
                "已取消训练。")
            self.stop_training()
            return
        params = self._params_override or self._params
        total = max(int(params.sm_iterations), 1)
        self.status_label.setText("正在准备训练…")
        self.progress.setRange(0, total)
        per_iter = 0.9 + 0.0005 * max(int(params.sm_num_envs), 1)
        self._log(
            f"PPO · {params.sm_num_envs} 环境 · {total} 轮 · "
            f"约 {total * per_iter / 60:.0f} 分钟"
            + (f" · 从 {Path(resume).name} 继续" if resume else ""))
        self.physics_img.setPixmap(QPixmap())
        self.physics_img.setText(
            "等待实时画面…" if self.render_toggle.isChecked()
            else "实时画面已关闭")
        self.preview_img.setText("训练完成后显示策略回放")
        worker = JumperTrainWorker(
            self._table, params,
            render=self.render_toggle.isChecked(),
            checkpoint=Path(resume) if resume else None,
            parent=self)
        worker.log.connect(self._log)
        worker.progress.connect(self._on_progress)
        worker.metrics.connect(self._on_metrics)
        worker.status.connect(self._on_status)
        worker.live_frame.connect(self._on_jumper_frame)
        worker.trained.connect(self._on_jumper_trained)
        worker.preview_ready.connect(self._on_policy_video_ready)
        worker.states_ready.connect(self._on_policy_states_ready)
        worker.failed.connect(self._on_jumper_failed)
        worker.finished.connect(self._on_jumper_finished)
        self._jumper = worker
        worker.start()

    def _on_stop_clicked(self) -> None:
        self._auto_preview_after_stop = False
        if self._running(self._jumper):
            self.status_label.setText("正在停止训练…")
            self._log("正在停止训练…")
        self.stop_training()

    def stop_training(self) -> None:
        had = self._busy()
        if had:
            self._stop_pending = True
        worker = self._preview_worker
        if self._running(worker):
            worker.request_stop()
            worker.wait(5000)
        worker = self._jumper
        if self._running(worker):
            worker.request_stop()
            worker.wait(8000)
        worker = self._worker
        if self._running(worker):
            worker.requestInterruption()
            worker.wait(3000)
        self._sync_buttons()

    # ---------------------------------------------------------- callbacks ----
    def _on_progress(self, i: int, n: int) -> None:
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(i)
        self.progress_pct.setText(f"{100 * i / max(n, 1):.0f}%")

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)
        self._log(text)

    def _on_metrics(self, rec: dict) -> None:
        self.rewards.add(rec)

    def _set_pixmap(self, label: QLabel, image) -> None:
        if image is None or image.isNull():
            return
        pix = QPixmap.fromImage(image)
        label.setPixmap(pix.scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_physics_frame(self, image) -> None:
        self._set_pixmap(self.physics_img, image)

    def _on_preview_frame(self, image) -> None:
        self._set_pixmap(self.preview_img, image)

    def _on_reference_saved(self, path: str) -> None:
        self._log(f"参考轨道已写入：{path}")

    def _on_weights_saved(self, path: str) -> None:
        self._weights = path
        self._weights_kind = "npz"
        self._log(f"权重已写入：{path}")
        self.export_btn.setEnabled(True)

    def _on_finished(self) -> None:
        self._stop_pending = False
        self._sync_buttons()
        if self._weights and Path(self._weights).exists():
            self.export_btn.setEnabled(True)

    # ------------------------------------------ Jumper PPO backend ----
    def _on_jumper_frame(self, path: str) -> None:
        """Show the newest offscreen training frame in the physics pane."""
        image = QImage(path)
        if not image.isNull():
            self._set_pixmap(self.physics_img, image)

    def _on_jumper_trained(self, run_dir: str, ckpt: str) -> None:
        self._weights = ckpt
        self._weights_kind = "pt"
        self._run_dir = run_dir
        if Path(ckpt).exists():
            self._continue_ckpt = ckpt
            self.export_btn.setEnabled(True)
        self._log(f"训练检查点：{ckpt}")
        self._sync_buttons()

    def _on_jumper_failed(self, message: str) -> None:
        self._log(f"训练失败：{message}")
        self.status_label.setText("PPO 训练失败 · 见日志")
        self._sync_buttons()

    def _on_jumper_finished(self) -> None:
        worker = self._jumper
        stopped = bool(worker is not None and getattr(worker, "_stopping", False))
        self._stop_pending = False
        if self._weights and Path(self._weights).exists():
            self.export_btn.setEnabled(True)
        self._sync_buttons()
        replay = self._dance is not None or self._policy_cap is not None
        self.status_label.setText(
            "训练已停止 · 点回放录制" if stopped
            else ("训练完成 · 回放已就绪" if replay else "训练完成"))
        if self._auto_preview_after_stop:
            self._auto_preview_after_stop = False
            if self._run_dir and self._table is not None:
                self._log("停止 · 录制完整回放")
                self.status_label.setText("训练已停止 · 正在录制回放")
                QTimer.singleShot(250, self._record_preview_now)
        elif stopped:
            self._log("训练已停止")

    # -------------------------------- policy replay (trained GPU policy) ----
    def _release_policy_video(self) -> None:
        if self._policy_cap is not None:
            self._policy_cap.release()
            self._policy_cap = None
        self._policy_path = None
        self._policy_idx = -1

    def _on_policy_video_ready(self, path: str) -> None:
        self._release_policy_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            self._log(f"无法打开策略回放视频：{path}")
            return
        self._policy_cap = cap
        self._policy_path = path
        self._policy_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._policy_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        times = np.asarray(self._table.times) if self._table is not None else []
        self._policy_offset = float(times[0]) if len(times) else 0.0
        self.replay_btn.setEnabled(True)
        stopped = bool(self._jumper is not None
                       and getattr(self._jumper, "_stopping", False))
        self.status_label.setText(
            ("训练已停止" if stopped else "训练完成") + " · 回放已就绪")
        self._show_right_page(1)
        self._start_dance_playback()

    def _on_policy_states_ready(self, path: str) -> None:
        """Load the replayed root/joint states and play them in our own model.

        The MP4 from play.py has a fixed camera; this replay can be orbited
        and paused like the local physics preview.
        """
        try:
            data = np.load(path)
            pose = np.asarray(data["pose"], dtype=np.float64)
            joint = np.asarray(data["joint"], dtype=np.float64)
            names = [str(n) for n in np.asarray(data["joint_names"]).ravel()]
            dt = float(np.asarray(data["dt"]).ravel()[0])
        except (OSError, ValueError, KeyError) as exc:
            self._log(f"回放状态读取失败：{exc}")
            return
        if len(pose) < 2 or dt <= 0:
            return
        if self._dance_crab is None:
            self._dance_crab = CrabModel()
        crab = self._dance_crab
        qpos = np.zeros((crab.model.nq, len(pose)))
        qpos[crab.free_adr:crab.free_adr + 7] = pose.T
        used = 0
        for col, name in enumerate(names):
            adr = crab.joint_adr.get(name)
            if adr is not None:
                qpos[adr] = joint[:, col]
                used += 1
        if used == 0:
            self._log("回放状态没有匹配的关节名")
            return
        self._release_policy_video()
        self._dance = {"times": np.arange(len(pose)) * dt, "qpos": qpos}
        stopped = bool(self._jumper is not None
                       and getattr(self._jumper, "_stopping", False))
        self.status_label.setText(
            ("训练已停止" if stopped else "训练完成") + " · 回放已就绪")
        self.replay_btn.setEnabled(True)
        self._show_right_page(1)
        self._start_dance_playback()

    def _tick_policy_video(self) -> None:
        """Play the recorded policy MP4 beside the source video, in sync."""
        cap = self._policy_cap
        if cap is None:
            self._dance_timer.stop()
            return
        elapsed = (self._dance_elapsed if self._dance_paused
                   else self._current_elapsed())
        idx = int(elapsed * self._policy_fps)
        if self._policy_count > 0:
            idx = min(idx, self._policy_count - 1)
        if idx != self._policy_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            self._policy_idx = idx
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                img = QImage(rgb.data, w, h, 3 * w,
                             QImage.Format.Format_RGB888).copy()
                self._set_pixmap(self.preview_img, img)
        self._update_video_frame(self._policy_offset + elapsed)

    # --------------------------------------- full-dance physics playback ----
    def _orbit_physics(self, dx: float, dy: float) -> None:
        self._phys_cam[0] -= 0.4 * dx
        self._phys_cam[1] = float(np.clip(self._phys_cam[1] - 0.4 * dy,
                                          -80.0, 10.0))
        if self._worker is not None:
            self._worker.set_camera(*self._phys_cam)

    def _zoom_physics(self, steps: float) -> None:
        self._phys_cam[2] = float(np.clip(self._phys_cam[2] * (1.0 - 0.1 * steps),
                                          0.25, 1.2))
        if self._worker is not None:
            self._worker.set_camera(*self._phys_cam)

    def _orbit_policy(self, dx: float, dy: float) -> None:
        self._policy_cam[0] -= 0.4 * dx
        self._policy_cam[1] = float(np.clip(self._policy_cam[1] - 0.4 * dy,
                                            -80.0, 10.0))
        if self._dance is not None and self._dance_crab is not None:
            self._tick_dance()

    def _zoom_policy(self, steps: float) -> None:
        self._policy_cam[2] = float(np.clip(self._policy_cam[2] * (1.0 - 0.1 * steps),
                                            0.25, 1.2))
        if self._dance is not None and self._dance_crab is not None:
            self._tick_dance()

    def _on_dance_ready(self, data) -> None:
        self._dance = data
        n = len(data.get("times", [])) if data else 0
        self._log(f"训练后完整舞蹈已就绪：{n} 帧 · 物理仿真回放")
        self.status_label.setText("训练完成 · 回放已就绪")
        self.replay_btn.setEnabled(n > 1)
        self._show_right_page(1)
        self._start_dance_playback()

    def _stop_dance_playback(self) -> None:
        self._dance_timer.stop()

    def _start_dance_playback(self) -> None:
        if self._policy_cap is not None:
            self._begin_playback()
            self._policy_idx = -1
            self._show_right_page(1)
            self._dance_timer.start()
            return
        if self._dance and len(self._dance.get("times", [])) >= 2:
            if self._dance_crab is None:
                self._dance_crab = CrabModel()
            self._begin_playback()
            self._show_right_page(1)
            self._dance_timer.start()
            return
        # A GPU run has a checkpoint but no recorded replay yet (e.g. it was
        # stopped): record one now and play it when it lands.
        if not self._busy() and self._table is not None:
            self._record_preview_now()

    def _begin_playback(self) -> None:
        self._dance_t0 = time.monotonic()
        self._dance_elapsed = 0.0
        self._dance_paused = False
        self.preview_pause_btn.blockSignals(True)
        self.preview_pause_btn.setChecked(False)
        self.preview_pause_btn.setText("暂停")
        self.preview_pause_btn.setEnabled(True)
        self.preview_pause_btn.blockSignals(False)

    def _current_elapsed(self) -> float:
        if self._policy_cap is not None:
            dur = (self._policy_count / self._policy_fps
                   if self._policy_count > 0 else 10.0)
        elif self._dance and len(self._dance.get("times", [])) >= 2:
            times = np.asarray(self._dance["times"])
            dur = max(float(times[-1] - times[0]), 1e-3)
        else:
            return 0.0
        return float((time.monotonic() - self._dance_t0) % max(dur, 1e-3))

    def _toggle_preview_pause(self, checked: bool) -> None:
        if checked:
            if self._policy_cap is None and self._dance is None:
                self.preview_pause_btn.blockSignals(True)
                self.preview_pause_btn.setChecked(False)
                self.preview_pause_btn.blockSignals(False)
                return
            self._dance_elapsed = self._current_elapsed()
            self._dance_paused = True
            self._dance_timer.stop()
            self.preview_pause_btn.setText("播放")
        else:
            self._dance_paused = False
            self._dance_t0 = time.monotonic() - self._dance_elapsed
            if self._policy_cap is not None or self._dance is not None:
                self._dance_timer.start()
            self.preview_pause_btn.setText("暂停")

    def _has_checkpoint(self) -> bool:
        return bool(self._run_dir and Path(self._run_dir).is_dir()
                    and newest_checkpoint(Path(self._run_dir)) is not None)

    def _sync_buttons(self) -> None:
        """One place for button availability; avoids conflicting states."""
        busy = self._busy()
        has_table = self._table is not None
        ckpt = self._has_checkpoint()
        self.start_btn.setEnabled(has_table and not busy and not self._stop_pending)
        self.stop_btn.setEnabled(busy)
        self.params_btn.setEnabled(not busy)
        self.backend_combo.setEnabled(not busy)
        self.quick_preview_btn.setEnabled(ckpt and not busy)
        has_replay = (ckpt or self._dance is not None
                      or self._policy_cap is not None)
        self.replay_btn.setEnabled(has_replay and not busy)
        self.history_btn.setEnabled(True)

    def _quick_preview(self) -> None:
        """Use the newest saved checkpoint for an ~8 s physics replay."""
        if self._busy():
            self._log("训练或回放仍在运行，请先停止再快速预览")
            return
        self._record_preview_now(seconds=8.0)

    def _record_preview_now(self, seconds: float | None = None) -> None:
        if self._busy():
            self._log("训练或回放仍在运行，请先停止")
            return
        run_dir = self._run_dir
        if not run_dir:
            found = latest_run_dir()
            if found is not None:
                run_dir = str(found)
                self._run_dir = run_dir
        if not run_dir:
            self._log("没有已保存的检查点，请先训练一次")
            return
        times = np.asarray(self._table.times) if self._table is not None \
            else np.zeros(0)
        duration = float(times[-1] - times[0]) if len(times) > 1 else 10.0
        quick = seconds is not None
        seconds_eff = min(float(seconds), duration) if quick else duration
        self._log(
            f"录制{'快速 ' if quick else '完整'}回放 · {seconds_eff:.0f}s")
        worker = JumperPreviewWorker(run_dir, duration, seconds=seconds,
                                     parent=self)
        worker.log.connect(self._log)
        worker.preview_ready.connect(self._on_policy_video_ready)
        worker.states_ready.connect(self._on_policy_states_ready)
        worker.failed.connect(self._on_jumper_failed)
        worker.finished.connect(self._on_preview_finished)
        self._preview_worker = worker
        worker.start()
        self._sync_buttons()

    def _on_preview_finished(self) -> None:
        self._stop_pending = False
        self._sync_buttons()

    def _tick_dance(self) -> None:
        """Play the trained policy: recorded MP4 (GPU) or qpos states (ES)."""
        if self._policy_cap is not None:
            self._tick_policy_video()
            return
        data = self._dance
        if not data or self._dance_crab is None:
            self._dance_timer.stop()
            return
        times = np.asarray(data["times"])
        qpos = np.asarray(data["qpos"])
        if len(times) < 2 or qpos.ndim != 2:
            self._dance_timer.stop()
            return
        elapsed = (self._dance_elapsed if self._dance_paused
                   else self._current_elapsed())
        idx = int(np.searchsorted(times, times[0] + elapsed))
        idx = min(max(idx, 0), len(times) - 1)
        self._dance_crab.apply_qpos(qpos[:, idx])
        az, el, dist = self._policy_cam
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [float(qpos[0, idx]), float(qpos[1, idx]), 0.10]
        cam.distance = float(dist)
        cam.azimuth = float(az)
        cam.elevation = float(el)
        rgb = self._dance_crab.render(camera=cam, width=430, height=300)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                     QImage.Format.Format_RGB888).copy()
        self._set_pixmap(self.preview_img, img)
        self._update_video_frame(times[0] + elapsed)

    # ---------------------------------------------------- source video pane ----
    def _open_video_source(self) -> None:
        """Open the original dance video shown next to the policy preview."""
        path = str(self._meta.get("video") or "")
        if self._cap is not None and self._cap_path == path:
            self._cap_idx = -1
            self._update_video_frame(0.0)
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cap_path = path or None
        self._cap_idx = -1
        if not path or not Path(path).exists():
            self.video_img.setText("未加载原视频")
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            self.video_img.setText("无法打开原视频")
            return
        self._cap = cap
        self._cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._cap_fps <= 1:
            self._cap_fps = 30.0
        self._update_video_frame(0.0)

    def _update_video_frame(self, t: float) -> None:
        cap = self._cap
        if cap is None:
            return
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = int(max(0.0, t) * self._cap_fps)
        if count > 0:
            idx = min(idx, count - 1)
        if idx == self._cap_idx:
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        self._cap_idx = idx
        if not ok or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w,
                     QImage.Format.Format_RGB888).copy()
        self._set_pixmap(self.video_img, img)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._stop_dance_playback()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if (self._dance or self._policy_cap is not None) \
                and not self._dance_timer.isActive():
            self._start_dance_playback()

    def shutdown(self) -> None:
        self.stop_training()
        self._stop_dance_playback()
        self._release_policy_video()
        if self._dance_crab is not None:
            self._dance_crab.close()
            self._dance_crab = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _on_render_toggled(self, on: bool) -> None:
        self._params.render = on
        self.physics_img.setPixmap(QPixmap())
        self.preview_img.setPixmap(QPixmap())
        if self._backend == "jumper":
            if on:
                self.physics_img.setText("训练渲染")
                self.preview_img.setText("训练完成后显示策略回放")
            else:
                self.physics_img.setText("实时画面已关闭")
                self.preview_img.setText("训练完成后显示策略回放")
            return
        if on:
            self.physics_img.setText("训练中显示物理仿真画面")
            self.preview_img.setText("训练中显示当前策略姿态")
        else:
            self.physics_img.setText("渲染已关闭 · 训练继续")
            self.preview_img.setText("渲染已关闭 · 训练继续")

    # ------------------------------------------------------------- export ----
    def _export_weights(self) -> None:
        if not self._weights or not Path(self._weights).exists():
            return
        default = str(Path.home() / "Desktop" / Path(self._weights).name)
        if self._weights_kind == "pt":
            path, _ = QFileDialog.getSaveFileName(
                self, "导出训练策略权重", default,
                "PyTorch 权重 (*.pt)")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "导出训练策略权重", default, "NumPy 权重 (*.npz)")
        if not path:
            return
        try:
            shutil.copyfile(self._weights, path)
            self._log(f"权重已导出：{path}")
        except OSError as exc:
            self._log(f"导出失败：{exc}")
