"""Main window: drag-drop import, analysis worker, synced playback."""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QCoreApplication, QPoint, QSettings, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QCursor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget,
)

from app.constants import (
    CURVE_UPDATE_EVERY, MAX_CRABS, PLAYBACK_FPS, STAGE_SPREAD,
    SUPPORTED_VIDEO_EXT,
)
from app.core.crab_model import CrabModel
from app.core.retarget import build_crab_commands, compute_human_channels
from app.core.s2d_pipeline import process_video
from app.ui.crab_view import CrabView
from app.ui.curves_panel import CurvesPanel
from app.ui.joint_probe import JointProbeDialog
from app.ui.environment_dialog import EnvironmentDialog
from app.ui.settings_dialog import SettingsDialog, load_settings
from app.ui.train_view import TrainPage
from app.ui.transport import TransportBar
from app.ui.video_panel import VideoPanel
from app.core.s2d_pipeline import video_hash, CACHE_DIR

STYLE = """
QMainWindow, QWidget { background: #0d1117; color: #e6edf3; font-size: 13px; }
QLabel#panelTitle { font-size: 14px; font-weight: 600; color: #e6edf3; }
QLabel#panelTip  { color: #8b949e; }
QLabel#headerTitle { font-size: 19px; font-weight: 700; color: #f0f6fc; }
QLabel#headerSub   { color: #8b949e; }
QLabel#videoSurface {
    background: #10141a; border: 1px solid #21262d; border-radius: 8px;
}
QLabel#crabSurface {
    background: #0f1319; border: 1px solid #21262d; border-radius: 8px;
}
QPushButton {
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
    border-radius: 6px; padding: 5px 14px;
}
QPushButton:hover { background: #30363d; }
QPushButton:disabled { color: #484f58; background: #161b22; }
QPushButton#accentBtn {
    background: #1f6feb; border: none; font-weight: 600; padding: 7px 18px;
}
QPushButton#accentBtn:hover { background: #388bfd; }
QPushButton#modeBtn { padding: 4px 10px; font-size: 12px; }
QPushButton#modeBtn:checked { background: #1f6feb; border-color: #1f6feb; }
QSlider::groove:horizontal {
    height: 6px; background: #21262d; border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
    background: #e6b450;
}
QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 3px; }
QStatusBar { background: #010409; color: #8b949e; }
QProgressBar {
    background: #21262d; border: none; border-radius: 4px;
    text-visible: false; max-height: 8px;
}
QProgressBar::chunk { background: #1f6feb; border-radius: 4px; }
QSplitter::handle { background: #010409; width: 3px; }
QDialog { background: #0d1117; }
QGroupBox {
    border: 1px solid #21262d; border-radius: 8px;
    margin-top: 10px; padding: 6px 6px 6px 6px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8b949e;
}
QCheckBox { spacing: 6px; }
QComboBox {
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
    border-radius: 6px; padding: 5px 10px;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #161b22; color: #e6edf3;
    selection-background-color: #1f6feb; selection-color: #f0f6fc;
}
"""


class AnalysisWorker(QThread):
    """Runs Sports2D + retargeting off the GUI thread."""

    progress = Signal(float)
    progress_text = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, video: Path, claw_ik: bool = True, parent=None):
        super().__init__(parent)
        self.video = video
        self.claw_ik = claw_ik

    def run(self) -> None:
        try:
            self.progress_text.emit("开始 Sports2D 姿态识别…")
            res = process_video(self.video, progress=self.progress.emit,
                                log=self.progress_text.emit)
            self.progress.emit(1.0)
            self.progress_text.emit("正在构建螃蟹动作…")
            # One model calibrates every person's table (the probed signs
            # and limits are identical across instances).
            crab = CrabModel()  # separate instance: worker-side calibration
            persons = []
            live_pct = (res.repair or {}).get("live_pct", [])
            for i, kps in enumerate((res.persons or [res.kps])[:MAX_CRABS]):
                channels = compute_human_channels(kps)
                persons.append({
                    "kps": kps,
                    "channels": channels,
                    "table": build_crab_commands(channels, crab,
                                                 claw_ik=self.claw_ik),
                    "valid_pct": live_pct[i] if i < len(live_pct) else None,
                })
            self.progress_text.emit(f"完成 · {len(persons)} 人")
            self.done.emit({
                "video": self.video,
                "persons": persons,
                "repair": res.repair or {},
            })
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


class _HistoryMenu(QMenu):
    """History popup whose right-click never activates the hovered action.

    A plain QMenu treats the right button as an activation in this context, so
    right-clicking a history item loaded that video before the delete menu
    appeared. Consume the right button here and forward the position to the
    custom context-menu handler instead.
    """

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self.customContextMenuRequested.emit(event.position().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if not QCoreApplication.organizationName():
            QCoreApplication.setOrganizationName("dangz")
            QCoreApplication.setApplicationName("crab-dance")
        QApplication.setApplicationDisplayName("Jumper Studio")
        self.setWindowTitle("Jumper Studio")
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)

        # --- widgets
        self.video_panel = VideoPanel()
        self.crab = CrabModel()
        self.crab_view = CrabView(self.crab)
        self.crab_view._label.setObjectName("crabSurface")
        self.curves = CurvesPanel()
        self.transport = TransportBar()

        self.import_btn = QPushButton("导入视频")
        self.import_btn.setObjectName("accentBtn")
        self.import_btn.clicked.connect(self._pick_file)
        self.history_btn = QPushButton("历史")
        self.history_btn.setToolTip("最近导入的视频 · 右键删除")
        self.history_btn.clicked.connect(self._open_history)
        self.reanalyze_btn = QPushButton("重新识别")
        self.reanalyze_btn.setToolTip("删除缓存并重新识别")
        self.reanalyze_btn.setEnabled(False)
        self.reanalyze_btn.clicked.connect(self._reanalyze)
        self.settings_btn = QPushButton("设置")
        self.settings_btn.clicked.connect(self._open_settings)
        self.transport.add_leading(self.import_btn)
        self.transport.add_leading(self.history_btn)
        self.transport.add_leading(self.reanalyze_btn)
        self.transport.add_leading(self.settings_btn)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._card("导入视频", self.video_panel))
        splitter.addWidget(self._card("螃蟹预览", self.crab_view,
                                     tip="鼠标拖动旋转 · 滚轮缩放"))
        splitter.setSizes([640, 640])

        self.curves_card = self._card("动作曲线", self.curves,
                                      tip="躯干欧拉角 / 钳关节指令")

        # Top-right entry into the secondary training page.
        self.train_btn = QPushButton("进入训练 →")
        self.train_btn.setToolTip("把当前识别结果导入动作训练页")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._open_train_page)
        top_row = QHBoxLayout()
        top_row.addStretch(1)
        top_row.addWidget(self.train_btn)

        body = QVBoxLayout()
        body.setContentsMargins(12, 8, 12, 8)
        body.setSpacing(8)
        body.addLayout(top_row)
        body.addWidget(splitter, stretch=1)
        body.addWidget(self.curves_card)
        body.addWidget(self.transport)

        central = QWidget()
        central.setLayout(body)
        # Page 0 = the dance view, page 1 = the training page (lazy).
        self.stack = QStackedWidget()
        self.stack.addWidget(central)
        self._train_page: TrainPage | None = None
        self.setCentralWidget(self.stack)

        # --- state
        self.worker: AnalysisWorker | None = None
        self._person_payloads: list[dict] = []
        self._current_video: Path | None = None
        self.playing = False
        self.t = 0.0
        self.duration = 0.0
        self._tick = 0
        self._last_wall: float | None = None   # wall-clock of the last tick
        self._idle_start = time.monotonic()
        self._settings_state = load_settings()
        self._settings_dialog: SettingsDialog | None = None

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / PLAYBACK_FPS))
        self.timer.timeout.connect(self._on_tick)

        # Idle animation drives the crab before any video is loaded.
        self.idle_timer = QTimer(self)
        self.idle_timer.setInterval(120)
        self.idle_timer.timeout.connect(self._on_idle)
        self.idle_timer.start()

        # --- wiring
        self.transport.play_toggled.connect(self.set_playing)
        self.transport.seek_requested.connect(self.seek)
        self.curves.seek_requested.connect(self.seek)

        # Apply persisted settings (panels / overlays / backdrop).
        self.set_curves_visible(self._settings_state["curves"])
        self._set_ghost_visible(self._settings_state["ghost"])
        self._set_detect_visible(self._settings_state["detect"])
        self._set_skeleton_visible(self._settings_state["skeleton"])
        self.crab_view.set_mapping_visible(False)   # mapping overlay retired
        self._set_height_factor(self._settings_state["height"])
        self._set_background(self._settings_state["background"])
        if (self._settings_state["background"] == "custom"
                and (self._settings_state["bg_image"]
                     or self._settings_state["floor_image"])):
            self.crab_view.set_custom_background(
                self._settings_state["bg_image"] or None,
                self._settings_state["floor_image"] or None)
        self._probe_dialog: JointProbeDialog | None = None
        self._env_dialog: EnvironmentDialog | None = None

        self.status_progress = QProgressBar()
        self.status_progress.setMaximumWidth(260)
        self.status_progress.setVisible(False)
        # The built-in percentage text clips vertically inside the status
        # bar (user: only the top half shows). The live percentage is
        # already in the status message ("姿态识别中… N%"), so the bar
        # carries no text of its own.
        self.status_progress.setTextVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)
        self.statusBar().showMessage("拖入视频开始")

        self._place_window()

    # ------------------------------------------------------------- window ----
    def _place_window(self, screen=None) -> None:
        """Open at a modest size, centred on the screen, fully on-screen.

        The old fixed 1320x860 is huge on a scaled secondary monitor (logical
        pixels are multiplied by the 1.75 device ratio), and if the frame is
        taller than the available area Windows centres it with the title bar
        above the screen -- the close button becomes unreachable. Size against
        the available geometry instead of a fixed number.
        """
        screen = screen or QApplication.screenAt(QCursor.pos())
        screen = screen or QApplication.primaryScreen()
        if screen is None:
            self.resize(1100, 720)
            return
        avail = screen.availableGeometry()
        min_w, min_h = self.minimumSizeHint().width(), self.minimumSizeHint().height()
        w = max(min_w, min(1100, avail.width() - 80))
        h = max(min_h, min(720, avail.height() - 96))
        self.resize(w, h)
        self.move(avail.x() + max(0, (avail.width() - w) // 2),
                  avail.y() + max(0, (avail.height() - h) // 2))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Safety net: if the frame still lands partly outside the screen
        # (Windows placement quirks), nudge it back so the title bar -- and
        # its close button -- stays clickable.
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        fg = self.frameGeometry()
        x = min(max(fg.x(), avail.x()),
                max(avail.x(), avail.right() + 1 - fg.width()))
        y = min(max(fg.y(), avail.y()),
                max(avail.y(), avail.bottom() + 1 - fg.height()))
        if (x, y) != (fg.x(), fg.y()):
            self.move(x, y)

    # ------------------------------------------------------------ settings ----
    def _open_settings(self) -> None:
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(self._settings_state, self)
            self._settings_dialog.curves_toggled.connect(self.set_curves_visible)
            self._settings_dialog.ghost_toggled.connect(self._set_ghost_visible)
            self._settings_dialog.detect_toggled.connect(self._set_detect_visible)
            self._settings_dialog.skeleton_toggled.connect(self._set_skeleton_visible)
            self._settings_dialog.height_changed.connect(self._set_height_factor)
            self._settings_dialog.multi_toggled.connect(self._set_multi_enabled)
            self._settings_dialog.person_changed.connect(self._set_follow_person)
            self._settings_dialog.background_changed.connect(self._set_background)
            self._settings_dialog.custom_bg_changed.connect(self._set_custom_bg)
            self._settings_dialog.probe_requested.connect(self._open_probe)
            self._settings_dialog.env_requested.connect(self._open_environment)
            if getattr(self, "_person_payloads", None):
                self._settings_dialog.set_person_count(
                    len(self._person_payloads),
                    [p.get("valid_pct") for p in self._person_payloads],
                    self._settings_state["person"])
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def set_curves_visible(self, visible: bool) -> None:
        self._settings_state["curves"] = visible
        self.curves_card.setVisible(visible)

    def _set_mapping_visible(self, visible: bool) -> None:
        self._settings_state["mapping"] = visible
        self.crab_view.set_mapping_visible(visible)

    def _set_ghost_visible(self, visible: bool) -> None:
        self._settings_state["ghost"] = visible
        self.crab_view.set_ghost_visible(visible)

    def _set_detect_visible(self, visible: bool) -> None:
        self._settings_state["detect"] = visible
        self.video_panel.set_detect_visible(visible)

    def _set_skeleton_visible(self, visible: bool) -> None:
        self._settings_state["skeleton"] = visible
        self.video_panel.set_skeleton_visible(visible)


    def _set_claw_ik(self, enabled: bool) -> None:
        self._settings_state["claw_ik"] = True
        self._rebuild_tables("钳部映射已更新")

    def _rebuild_tables(self, message: str = "") -> None:
        """Rebuild every person's command table from cached channels.

        Used after a live mapping setting or a keyframe-calibration change;
        no video re-detection. IK solves per frame, hence the wait cursor.
        """
        persons = getattr(self, "_person_payloads", None)
        if not persons:
            return
        was_playing = self.playing
        self._pause()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            crab = self.crab_view.crab
            for p in persons:
                p["table"] = build_crab_commands(
                    p["channels"], crab, claw_ik=True)
            self._apply_person_mode()
        finally:
            QApplication.restoreOverrideCursor()
        if message:
            self.statusBar().showMessage(message, 5000)
        self._update_all(force=True)
        if was_playing:
            self.set_playing(True)







    # -------------------------------------------------------- pill memory ----



    def _set_height_factor(self, factor: float) -> None:
        self._settings_state["height"] = factor
        self.crab_view.set_height_factor(factor)

    def _set_background(self, bg_id: str) -> None:
        self._settings_state["background"] = bg_id
        self.crab_view.set_background(bg_id)

    def _set_custom_bg(self, image_path: str, floor_path: str) -> None:
        self._settings_state["background"] = "custom"
        self._settings_state["bg_image"] = image_path
        self._settings_state["floor_image"] = floor_path
        self.statusBar().showMessage("应用自定义背景 / 地面…", 3000)
        self.crab_view.set_custom_background(image_path or None,
                                             floor_path or None)

    def _open_environment(self, first_run: bool = False) -> None:
        if self._env_dialog is None:
            self._env_dialog = EnvironmentDialog(self, first_run=first_run)
        else:
            self._env_dialog.first_run = first_run
            self._env_dialog._diagnose()
        self._env_dialog.show()
        self._env_dialog.raise_()
        self._env_dialog.activateWindow()

    def _open_probe(self) -> None:
        if self._probe_dialog is None:
            self._probe_dialog = JointProbeDialog(self.crab_view, self)
            self._probe_dialog.calib_changed.connect(
                lambda: self._rebuild_tables("映射校准已应用"))
        self._probe_dialog.show()
        self._probe_dialog.raise_()
        self._probe_dialog.activateWindow()

    # ------------------------------------------------------------- history ----
    KEY_HISTORY = "history/recent_v2"
    KEY_HISTORY_OLD = "history/recent"
    HISTORY_MAX = 8

    def _recent_entries(self) -> list[dict]:
        """[{path, ts}] newest first; migrates the old path-only list."""
        raw = QSettings().value(self.KEY_HISTORY, "", str) or ""
        entries: list[dict] = []
        if raw:
            try:
                data = json.loads(raw)
                for item in data:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        entries.append({"path": item["path"],
                                        "ts": float(item.get("ts") or 0.0)})
            except (ValueError, TypeError):
                entries = []
        if not entries:
            old = QSettings().value(self.KEY_HISTORY_OLD, [], list) or []
            for p in old:
                if isinstance(p, str):
                    try:
                        ts = Path(p).stat().st_mtime
                    except OSError:
                        ts = 0.0
                    entries.append({"path": p, "ts": ts})
        return entries[:self.HISTORY_MAX]

    def _save_history(self, entries: list[dict]) -> None:
        QSettings().setValue(self.KEY_HISTORY,
                             json.dumps(entries[:self.HISTORY_MAX]))

    def _push_history(self, video: Path) -> None:
        entries = [e for e in self._recent_entries()
                   if e.get("path") != str(video)]
        entries.insert(0, {"path": str(video), "ts": time.time()})
        self._save_history(entries)

    def _remove_history(self, path: str) -> None:
        self._save_history([e for e in self._recent_entries()
                            if e.get("path") != path])
        self.statusBar().showMessage("已从历史中删除", 2500)

    def _pose_cache_ready(self, video: Path) -> bool:
        """True only when Sports2D finished and left a TRC for this video."""
        cache = CACHE_DIR / video_hash(video)
        return ((cache / "_done").is_file()
                and any(cache.rglob("*.trc")))

    def load_last_video(self) -> None:
        """Reopen the most recent dance on startup, but only if cached.

        A missing or interrupted recognition used to start a 10-minute
        Sports2D run as soon as the app opened. Now the last history entry is
        inspected first: if it would need re-analysis, open nothing and let
        the user choose a video explicitly.
        """
        for entry in self._recent_entries():
            path = Path(entry["path"])
            if not path.exists():
                continue
            if not self._pose_cache_ready(path):
                self.statusBar().showMessage(
                    "上次视频需要重新识别 · 未自动打开，请手动导入", 8000)
                return
            self.load_video(path)
            return

    @staticmethod
    def _history_label(entry: dict) -> str:
        path = Path(entry["path"])
        name = path.stem
        ts = float(entry.get("ts") or 0.0)
        when = (datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
                if ts > 0 else "")
        return (name if path.exists() else "✕ " + name) + (
            f" · {when}" if when else "")

    def _open_history(self) -> None:
        menu = _HistoryMenu(self)
        entries = self._recent_entries()
        if not entries:
            act = menu.addAction("暂无历史")
            act.setEnabled(False)
        for entry in entries:
            path = Path(entry["path"])
            act = menu.addAction(self._history_label(entry))
            act.setToolTip(entry["path"])
            act.setData(entry["path"])
            act.setEnabled(path.exists())
            act.triggered.connect(lambda _=False, v=path: self.load_video(v))
        menu.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def context(pos) -> None:
            act = menu.actionAt(pos)
            if act is None or not act.data():
                return
            path = str(act.data())
            sub = QMenu(menu)
            sub.setStyleSheet(menu.styleSheet())
            delete = sub.addAction("从历史删除")
            delete.triggered.connect(lambda: self._remove_history(path))
            sub.exec(menu.mapToGlobal(pos))
            menu.close()

        menu.customContextMenuRequested.connect(context)
        menu.exec(self.history_btn.mapToGlobal(
            QPoint(0, self.history_btn.height())))

    # ------------------------------------------------------------- layout ----
    @staticmethod
    def _card(title_text: str, inner: QWidget, tip: str = "") -> QWidget:
        card = QWidget()
        card.setObjectName("cardBox")
        card.setStyleSheet(
            "QWidget#cardBox { background: #010409; border-radius: 10px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)
        head = QHBoxLayout()
        title = QLabel(title_text)
        title.setObjectName("panelTitle")
        head.addWidget(title)
        head.addStretch(1)
        if tip:
            tip_lab = QLabel(tip)
            tip_lab.setObjectName("panelTip")
            head.addWidget(tip_lab)
        lay.addLayout(head)
        lay.addWidget(inner, stretch=1)
        return card

    # ------------------------------------------------------------- import ----
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.load_video(Path(path))
                return

    def _pick_file(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_VIDEO_EXT))
        path, _ = QFileDialog.getOpenFileName(
            self, "选择舞蹈视频", "", f"视频文件 {exts}")
        if path:
            self.load_video(Path(path))

    def _cancel_analysis(self) -> bool:
        """Force-stop the in-process Sports2D call after user confirmation.

        Sports2D has no cooperative cancel hook; this is the only way to let
        the user replace a running recognition. The partial cache is removed
        so the next attempt starts cleanly.
        """
        worker = self.worker
        if worker is None or not worker.isRunning():
            return True
        video = getattr(worker, "video", None)
        for sig in (worker.progress, worker.progress_text,
                    worker.done, worker.failed):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        worker.requestInterruption()
        worker.terminate()
        worker.wait(4000)
        if worker.isRunning():
            self.statusBar().showMessage("无法停止当前识别，请重启软件", 8000)
            return False
        if video is not None:
            shutil.rmtree(CACHE_DIR / video_hash(video), ignore_errors=True)
        self.worker = None
        self.status_progress.setVisible(False)
        self.video_panel.end_progress()
        self.statusBar().showMessage("已停止当前识别")
        return True

    def load_video(self, path: Path) -> None:
        if self.worker is not None and self.worker.isRunning():
            if QMessageBox.question(
                    self, "正在识别",
                    "当前视频还在识别。停止它并导入新视频？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No) !=                     QMessageBox.StandardButton.Yes:
                return
            if not self._cancel_analysis():
                return
        if not path.exists() or path.suffix.lower() not in SUPPORTED_VIDEO_EXT:
            self.statusBar().showMessage(f"不支持的文件: {path.name}", 5000)
            return

        self._pause()
        # Show the idle wave while Sports2D runs; the analysed dance replaces
        # it when _analysis_done arrives. Stopping the timer here left a
        # frozen crab on startup / during analysis.
        self.duration = 0.0
        self.crab_view.set_actors([])
        self.idle_timer.start()
        if self._train_page is not None:
            self._train_page.stop_training()
        if getattr(self, "stack", None) is not None:
            self.stack.setCurrentIndex(0)
        self.train_btn.setEnabled(False)
        self.statusBar().showMessage(
            f"正在分析 {path.name} … · 首次运行需下载姿态模型，请耐心等待")
        self.status_progress.setRange(0, 0)  # indeterminate until tqdm ticks
        self.status_progress.setVisible(True)

        self.video_panel.begin_progress()
        self.worker = AnalysisWorker(
            path, claw_ik=True)
        self.worker.progress.connect(self._on_progress)
        self.worker.progress_text.connect(self.video_panel.append_progress)
        self.worker.done.connect(self._analysis_done)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.start()

    def _on_progress(self, fraction: float) -> None:
        """Determinate pose-progress bar (fed by Sports2D's tqdm output)."""
        frac = max(0.0, min(fraction, 1.0))
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(int(frac * 100))
        if frac >= 1.0:
            self.statusBar().showMessage("姿态识别完成，正在构建螃蟹动作…")
        else:
            self.statusBar().showMessage(f"姿态识别中… {int(frac * 100)}%")

    def _analysis_failed(self, message: str) -> None:
        self.status_progress.setVisible(False)
        self.video_panel.end_progress()
        self.statusBar().showMessage("分析失败", 8000)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("分析失败")
        box.setText(message.splitlines()[0])
        box.setDetailedText(message)
        box.exec()

    def _analysis_done(self, payload: dict) -> None:
        self.status_progress.setVisible(False)
        self.video_panel.end_progress()
        video: Path = payload["video"]

        if not self.video_panel.open_video(str(video)):
            self.statusBar().showMessage("无法打开视频", 5000)
            return
        self._current_video = video
        self.reanalyze_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self._person_payloads: list[dict] = payload["persons"]
        repair: dict = payload.get("repair") or {}
        if self._settings_dialog is not None:
            self._settings_dialog.set_person_count(
                len(self._person_payloads),
                [p.get("valid_pct") for p in self._person_payloads],
                self._settings_state["person"])
        self._apply_person_mode()

        self.idle_timer.stop()
        self.duration = self.video_panel.duration
        self.transport.set_duration(self.duration)
        self._push_history(video)
        n = len(self._person_payloads)
        notes = []
        if n > 1:
            notes.append(f"检测到 {n} 人 · 设置里切换跟随")
        if repair.get("merged"):
            notes.append(f"已缝合 {repair['merged']} 条断续轨迹")
        if repair.get("dropped"):
            notes.append(f"剔除 {repair['dropped']} 条无效轨迹")
        self.statusBar().showMessage(
            f"就绪 · {video.name}"
            f" · 腿长比例 {self._person_payloads[0]['channels'].leg_ratio:.2f}"
            + (" · " + " · ".join(notes) if notes else ""))
        self.seek(0.0)
        self.set_playing(True)

    # ------------------------------------------------------------ re-run ----
    def _reanalyze(self) -> None:
        """Drop the cached pose result and re-run detection on this video
        (e.g. after the person count changed)."""
        if self._current_video is None or not self._current_video.exists():
            return
        if self.worker is not None and self.worker.isRunning():
            self.statusBar().showMessage("正在分析，请稍候…", 3000)
            return
        cache = CACHE_DIR / video_hash(self._current_video)
        if cache.exists():
            import shutil
            shutil.rmtree(cache, ignore_errors=True)
        self.statusBar().showMessage("已清除缓存，重新识别中…", 4000)
        self.load_video(self._current_video)

    # ------------------------------------------------------ person mode ----
    def _stage_positions(self) -> list[tuple[float, float]]:
        """World xy per person: video left-to-right mapped onto the stage
        axis across the camera view (min-max over the detected people)."""
        med_x = []
        for p in self._person_payloads:
            hips = [(p["kps"].xy("hip_l")[:, 0] + p["kps"].xy("hip_r")[:, 0]) / 2]
            med_x.append(float(np.nanmedian(hips)))
        lo, hi = min(med_x), max(med_x)
        ts = [(x - lo) / (hi - lo) * 2.0 - 1.0 if hi > lo else 0.0
              for x in med_x]
        az = np.radians(self.crab_view.camera.azimuth - 90.0)
        axis = np.array([np.cos(az), np.sin(az)])
        return [tuple((t * STAGE_SPREAD / 2.0) * axis) for t in ts]

    def _apply_person_mode(self) -> None:
        """Cast the crabs per the follow/multi settings (no re-analysis)."""
        persons = getattr(self, "_person_payloads", None)
        if not persons:
            return
        multi = self._settings_state["multi"] and len(persons) > 1
        if multi:
            cast = persons
            xys = self._stage_positions()
            follow = None  # every person is somebody's crab
        else:
            # The settings combo is 1-based with 0 = auto (largest person).
            person = int(self._settings_state["person"])
            idx = min(max(person - 1, 0), len(persons) - 1)
            cast = [persons[idx]]
            xys = [(0.0, 0.0)]
            follow = idx
        self.video_panel.set_persons(
            [p["kps"] for p in persons], follow=follow)

        self.crab_view.set_actors([
            {"table": p["table"], "channels": p["channels"], "xy": xy}
            for p, xy in zip(cast, xys)
        ])
        # Curves + pills follow the primary crab (followed person, or the
        # largest one in multi mode).
        primary = cast[0]
        self.curves.set_channels(primary["channels"])
        self.curves.set_table(
            primary["table"],
            ranges={j: tuple(np.degrees(self.crab_view.crab.joint_range[f"LF_J{j}_joint"]))
                    for j in range(5)})
        # Repaint immediately: a mode switch rebuilds the cast and clears
        # the crab view's last frame time, so with playback paused the
        # preview used to stay frozen until the user pressed play.
        if self.duration > 0:
            t = min(self.t, max(self.duration - 1e-3, 0.0))
            self.crab_view.show_time(t, force=True)
            self.video_panel.show_time(t)




    def _set_multi_enabled(self, enabled: bool) -> None:
        self._settings_state["multi"] = enabled
        self._apply_person_mode()

    def _set_follow_person(self, person_1based: int) -> None:
        self._settings_state["person"] = person_1based
        if not self._settings_state["multi"]:
            self._apply_person_mode()

    # ------------------------------------------------------- training page ----
    def _primary_payload(self) -> tuple[dict | None, int]:
        """The payload the training page should import (same target as the
        followed crab: the followed person, or person 1 in multi mode)."""
        persons = getattr(self, "_person_payloads", None)
        if not persons:
            return None, 0
        if self._settings_state["multi"] and len(persons) > 1:
            return persons[0], 0
        person = int(self._settings_state["person"])
        idx = min(max(person - 1, 0), len(persons) - 1)
        return persons[idx], idx

    def _open_train_page(self) -> None:
        payload, idx = self._primary_payload()
        if payload is None:
            self.statusBar().showMessage("先导入并识别一个舞蹈视频", 4000)
            return
        # First arrival at training: diagnose once and offer to install.
        if (not os.environ.get("CRAB_SKIP_ENV_PROMPT")
                and not QSettings().value("setup/first_train_diagnosed",
                                          False, bool)):
            QSettings().setValue("setup/first_train_diagnosed", True)
            self._open_environment(first_run=True)
        self._pause()
        video = self._current_video
        stem = video.stem if video else "motion"
        meta = {
            "video": str(video or ""),
            "video_name": stem,
            "video_stem": stem,
            "tag": video_hash(video)[:16] if video else "ref",
            "person": f"人物 {idx + 1}",
        }
        if self._train_page is None:
            self._train_page = TrainPage(self)
            self._train_page.back_requested.connect(self._close_train_page)
            self.stack.addWidget(self._train_page)
        self._train_page.set_reference(payload["table"], meta)
        self.stack.setCurrentIndex(1)

    def _close_train_page(self) -> None:
        self.stack.setCurrentIndex(0)
        self.seek(self.t)

    # ----------------------------------------------------------- playback ----
    def set_playing(self, playing: bool) -> None:
        if self.duration <= 0:
            return
        self.playing = playing
        self.transport.set_playing(playing)
        if playing:
            self._last_wall = None   # fresh wall-clock reference
            self.timer.start()
        else:
            self.timer.stop()

    def _pause(self) -> None:
        self.set_playing(False)

    def seek(self, t: float) -> None:
        self.t = max(0.0, min(t, max(self.duration - 1e-3, 0.0)))
        self._tick = 0
        self._last_wall = None
        self._update_all(force=True)

    def _on_idle(self) -> None:
        # The wave is the no-dance placeholder. Once a dance is analysed and
        # running, _analysis_done stops this timer.
        if self.duration > 0:
            return
        self.crab_view.idle_tick(time.monotonic() - self._idle_start)

    def _on_tick(self) -> None:
        # Wall-clock pacing: heavy renders (multi-crab compositing) delay
        # timer ticks, but the video time follows REAL elapsed time, so
        # playback drops frames instead of sliding into slow motion.
        now = time.monotonic()
        dt = (now - self._last_wall) if self._last_wall is not None \
            else self.timer.interval() / 1000.0
        self._last_wall = now
        dt = min(dt, 3.0 * self.timer.interval() / 1000.0)
        self.t += dt
        if self.t >= self.duration:
            self.t = max(self.duration - 1e-3, 0.0)
            self._update_all(force=True)
            self.set_playing(False)
            return
        self._update_all()

    def _update_all(self, force: bool = False) -> None:
        if self.duration <= 0:
            return
        self.video_panel.show_time(self.t)
        self.crab_view.show_time(self.t, force=force)
        self._tick += 1
        if self.curves_card.isVisible() and (
                force or self._tick % CURVE_UPDATE_EVERY == 0):
            self.curves.set_cursor(self.t)
        self.transport.update_time(self.t, self.playing)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._pause()
        self.idle_timer.stop()
        self.timer.stop()
        if self._probe_dialog is not None:
            self._probe_dialog.close()
        if self._env_dialog is not None:
            self._env_dialog.close()
        if self._train_page is not None:
            self._train_page.shutdown()
        worker = self.worker
        if worker is not None and worker.isRunning():
            # A running Sports2D analysis cannot be interrupted mid-call.
            # Disconnect its signals FIRST so a late done/failed can never
            # touch the dead UI -- that path popped a fresh dialog/window
            # AFTER the user closed the app ("关了会自己重新启动"). If the
            # thread will not stop in 2 s, force-exit: every setting is
            # already persisted (QSettings writes synchronously).
            for sig in (worker.progress, worker.done, worker.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            if not worker.wait(2000):
                import os
                os._exit(0)
        self.video_panel.close_video()
        self.crab_view.close()
        super().closeEvent(event)
