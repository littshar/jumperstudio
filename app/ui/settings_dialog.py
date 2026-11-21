"""Settings dialog: overlay/display toggles + crab background presets.

Changes apply live (the main window reacts to the signals) and persist to
QSettings, so they survive restarts.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from app.constants import (
    BG_DEFAULT, CRAB_BACKGROUNDS, HEIGHT_FACTOR_MAX, HEIGHT_FACTOR_MIN,
)

KEY_CURVES = "ui/show_curves"
KEY_MAPPING = "ui/show_mapping"
KEY_GHOST = "ui/show_ghost"
KEY_DETECT = "ui/show_detect"
KEY_SKELETON = "ui/show_skeleton"
# The 转身/跳跃 pill strips were REMOVED (round-18): the auto front/back
# facing tracks turns directly and detected hops play on their own, so the
# spin-default key is retired with the UI.
KEY_CLAW_IK = "retarget/claw_ik"   # GMR-style IK claw tracking (vs channels)
KEY_HEIGHT = "retarget/height_factor"
KEY_MULTI = "retarget/multi_crab"
KEY_PERSON = "retarget/follow_person"   # 0 = auto (largest person)
KEY_BG = "crab/background"
KEY_BG_IMG = "crab/custom_bg_image"
KEY_FLOOR_IMG = "crab/custom_floor_image"

HEIGHT_MIN, HEIGHT_MAX = HEIGHT_FACTOR_MIN, HEIGHT_FACTOR_MAX


def load_settings() -> dict:
    """Read persisted settings, falling back to the defaults."""
    s = QSettings()
    ids = {p["id"] for p in CRAB_BACKGROUNDS} | {"custom"}
    bg = s.value(KEY_BG, BG_DEFAULT, str)
    try:
        height = float(s.value(KEY_HEIGHT, 1.0))
    except (TypeError, ValueError):
        height = 1.0
    try:
        person = int(s.value(KEY_PERSON, 0))
    except (TypeError, ValueError):
        person = 0
    return {
        "curves": s.value(KEY_CURVES, True, bool),
        "mapping": False,
        "ghost": s.value(KEY_GHOST, False, bool),
        "detect": s.value(KEY_DETECT, False, bool),
        "skeleton": s.value(KEY_SKELETON, True, bool),
        "claw_ik": True,
        "height": min(max(height, HEIGHT_MIN), HEIGHT_MAX),
        "multi": s.value(KEY_MULTI, False, bool),
        "person": max(person, 0),
        "background": bg if bg in ids else BG_DEFAULT,
        "bg_image": s.value(KEY_BG_IMG, "", str),
        "floor_image": s.value(KEY_FLOOR_IMG, "", str),
    }


class SettingsDialog(QDialog):
    """Non-modal settings; every change takes effect immediately."""

    curves_toggled = Signal(bool)
    ghost_toggled = Signal(bool)
    detect_toggled = Signal(bool)
    skeleton_toggled = Signal(bool)
    height_changed = Signal(float)
    multi_toggled = Signal(bool)
    person_changed = Signal(int)
    background_changed = Signal(str)
    custom_bg_changed = Signal(str, str)   # image path, floor image path
    probe_requested = Signal()             # open the joint probe tool
    env_requested = Signal()               # open the environment dialog

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(False)
        self.setMinimumWidth(360)

        def checkbox(label: str, key: str, checked: bool, signal: Signal,
                     tip: str = ""):
            box = QCheckBox(label)
            box.setObjectName(f"cb_{key}")
            box.setChecked(checked)
            if tip:
                box.setToolTip(tip)
            box.toggled.connect(signal.emit)
            box.toggled.connect(lambda v, k=key: QSettings().setValue(k, v))
            return box

        ui_box = QGroupBox("界面")
        ui_lay = QVBoxLayout(ui_box)
        ui_lay.addWidget(checkbox(
            "动作曲线", KEY_CURVES, state["curves"], self.curves_toggled,
            "显示底部曲线"))
        ui_lay.addWidget(checkbox(
            "显示手部动作识别", KEY_GHOST, state["ghost"], self.ghost_toggled,
            "显示识别出的舞者手臂"))

        retarget_box = QGroupBox("重定向")
        retarget_lay = QVBoxLayout(retarget_box)

        height_row = QHBoxLayout()
        height_label = QLabel("螃蟹站高")
        self.height_value = QLabel(f"{state['height'] * 100:.0f}%")
        self.height_value.setFixedWidth(48)
        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(int(HEIGHT_MIN * 100), int(HEIGHT_MAX * 100))
        self.height_slider.setValue(int(state["height"] * 100))
        self.height_slider.valueChanged.connect(self._on_height)
        height_row.addWidget(height_label)
        height_row.addWidget(self.height_slider, stretch=1)
        height_row.addWidget(self.height_value)
        retarget_lay.addLayout(height_row)

        # Multi-person staging: which dancer the crab follows, or one crab
        # per person lined up by on-screen position.
        self.multi_box = QCheckBox("多人模式", objectName=f"cb_{KEY_MULTI}")
        self.multi_box.setToolTip("每人一只螃蟹")
        self.multi_box.setChecked(state["multi"])
        self.multi_box.toggled.connect(self.multi_toggled.emit)
        self.multi_box.toggled.connect(
            lambda v: QSettings().setValue(KEY_MULTI, v))
        retarget_lay.addWidget(self.multi_box)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("单人跟随"))
        self.person_combo = QComboBox()
        self.person_combo.addItem("自动", 0)
        self.person_combo.setCurrentIndex(0)
        self.person_combo.currentIndexChanged.connect(self._on_person)
        person_row.addWidget(self.person_combo, stretch=1)
        self.person_row_widget = QWidget()
        self.person_row_widget.setLayout(person_row)
        retarget_lay.addWidget(self.person_row_widget)
        self.multi_box.toggled.connect(self.person_row_widget.setEnabled)
        self.person_row_widget.setEnabled(not state["multi"])

        ov_box = QGroupBox("视频叠加")
        ov_lay = QVBoxLayout(ov_box)
        ov_lay.addWidget(checkbox(
            "人物识别框", KEY_DETECT, state["detect"], self.detect_toggled,
            "显示人物框"))
        ov_lay.addWidget(checkbox(
            "骨架", KEY_SKELETON, state["skeleton"], self.skeleton_toggled,
            "显示骨架"))

        bg_box = QGroupBox("螃蟹背景")
        self.bg_combo = QComboBox()
        for preset in CRAB_BACKGROUNDS:
            self.bg_combo.addItem(self._swatch(preset), preset["name"],
                                  preset["id"])
        self.bg_combo.addItem(self._swatch(
            {"top": (90, 90, 96), "bottom": (150, 150, 156)}), "自定义", "custom")
        idx = self.bg_combo.findData(state["background"])
        self.bg_combo.setCurrentIndex(max(idx, 0))
        self.bg_combo.currentIndexChanged.connect(self._on_bg)
        bg_lay = QVBoxLayout(bg_box)
        bg_lay.addWidget(self.bg_combo)

        # Custom backdrop + floor textures, imported by the user.
        import_row = QHBoxLayout()
        btn_bg = QPushButton("背景图")
        btn_bg.clicked.connect(self._pick_bg_image)
        btn_floor = QPushButton("地面贴图")
        btn_floor.clicked.connect(self._pick_floor_image)
        btn_clear = QPushButton("清除")
        btn_clear.clicked.connect(self._clear_custom)
        import_row.addWidget(btn_bg)
        import_row.addWidget(btn_floor)
        import_row.addWidget(btn_clear)
        btn_bg.setToolTip("背景图")
        btn_floor.setToolTip("地面贴图")
        btn_clear.setToolTip("清除自定义")
        bg_lay.addLayout(import_row)
        self._bg_image_path = state["bg_image"]
        self._floor_image_path = state["floor_image"]

        probe_btn = QPushButton("关节映射")
        probe_btn.setToolTip("")
        probe_btn.clicked.connect(self.probe_requested.emit)
        bg_lay.addWidget(probe_btn)

        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self._reset)
        close_btn = QPushButton("完成")
        close_btn.setObjectName("accentBtn")
        close_btn.clicked.connect(self.close)
        env_btn = QPushButton("环境诊断")
        env_btn.setToolTip("检查并安装训练环境")
        env_btn.clicked.connect(self.env_requested.emit)
        btns = QHBoxLayout()
        btns.addWidget(reset_btn)
        btns.addWidget(env_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(ui_box)
        lay.addWidget(retarget_box)
        lay.addWidget(ov_box)
        lay.addWidget(bg_box)
        lay.addLayout(btns)

    # ------------------------------------------------------------ events ----
    def set_person_count(self, n: int, valid_pcts: list | None = None,
                         current: int = 0) -> None:
        """List the detected people in the follow combo (1-based labels).

        valid_pcts carries each person's tracked-frame percentage (after
        track repair), shown so the user can tell a well-tracked dancer
        from a fragmentary one before switching 跟随. `current` is the
        saved 1-based selection (0 = auto); it is restored after the list
        is rebuilt so reopening the dialog does not silently reset the
        followed person.
        """
        n = max(1, n)
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        self.person_combo.addItem("自动", 0)
        for i in range(n):
            pct = valid_pcts[i] if valid_pcts and i < len(valid_pcts) else None
            label = f"人物 {i + 1}"
            if pct is not None:
                label += f" · {pct}% 有效"
            self.person_combo.addItem(label, i + 1)
        idx = self.person_combo.findData(max(0, min(int(current), n)))
        self.person_combo.setCurrentIndex(max(idx, 0))
        self.person_combo.blockSignals(False)

    def select_person(self, index_1based: int) -> None:
        idx = self.person_combo.findData(max(0, index_1based))
        if idx >= 0:
            self.person_combo.setCurrentIndex(idx)

    def _on_person(self, index: int) -> None:
        value = int(self.person_combo.itemData(index) or 0)
        QSettings().setValue(KEY_PERSON, value)
        self.person_changed.emit(value)

    def _on_height(self, value: int) -> None:
        factor = value / 100.0
        self.height_value.setText(f"{factor * 100:.0f}%")
        QSettings().setValue(KEY_HEIGHT, factor)
        self.height_changed.emit(factor)

    def _on_bg(self, index: int) -> None:
        bg_id = self.bg_combo.itemData(index)
        QSettings().setValue(KEY_BG, bg_id)
        if bg_id == "custom":
            self.custom_bg_changed.emit(self._bg_image_path,
                                        self._floor_image_path)
        else:
            self.background_changed.emit(bg_id)

    # ------------------------------------------------- custom background ----
    def _pick_bg_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        self._bg_image_path = path
        QSettings().setValue(KEY_BG_IMG, path)
        QSettings().setValue(KEY_BG, "custom")
        self._select_bg("custom")
        self.custom_bg_changed.emit(path, self._floor_image_path)

    def _pick_floor_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择地面贴图", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        self._floor_image_path = path
        QSettings().setValue(KEY_FLOOR_IMG, path)
        QSettings().setValue(KEY_BG, "custom")
        self._select_bg("custom")
        self.custom_bg_changed.emit(self._bg_image_path, path)

    def _clear_custom(self) -> None:
        self._bg_image_path = ""
        self._floor_image_path = ""
        QSettings().setValue(KEY_BG_IMG, "")
        QSettings().setValue(KEY_FLOOR_IMG, "")
        QSettings().setValue(KEY_BG, BG_DEFAULT)
        self._select_bg(BG_DEFAULT)
        # Empty the texture first (this rebuilds the model); a bare
        # background_changed() only re-tints the floor and left the user's
        # imported image on it, which read as "clear does nothing".
        self.custom_bg_changed.emit("", "")
        self.background_changed.emit(BG_DEFAULT)

    def _select_bg(self, bg_id: str) -> None:
        idx = self.bg_combo.findData(bg_id)
        if idx >= 0:
            self.bg_combo.blockSignals(True)
            self.bg_combo.setCurrentIndex(idx)
            self.bg_combo.blockSignals(False)
        QSettings().setValue(KEY_BG, bg_id)

    def _reset(self) -> None:
        for box, key, default in self._boxes():
            if box.isChecked() != default:
                box.setChecked(default)
        if self.height_slider.value() != 100:
            self.height_slider.setValue(100)
        if self.person_combo.currentIndex() != 0:
            self.person_combo.setCurrentIndex(0)

    def _boxes(self):
        return [
            (self.findChild(QCheckBox, f"cb_{key}"), key, default)
            for key, default in (
                (KEY_CURVES, True), (KEY_GHOST, False),
                (KEY_DETECT, False), (KEY_SKELETON, True),
                (KEY_MULTI, False),
            )
        ]

    @staticmethod
    def _swatch(preset: dict) -> QPixmap:
        """Tiny gradient preview used as the combo entry icon."""
        pm = QPixmap(34, 20)
        pm.fill(QColor(0x2b, 0x31, 0x38))
        p = QPainter(pm)
        grad = QLinearGradient(0, 0, 0, 20)
        grad.setColorAt(0.0, QColor(*preset.get("top", (60, 63, 68))))
        grad.setColorAt(1.0, QColor(*preset.get("bottom", (90, 95, 102))))
        p.fillRect(pm.rect().adjusted(1, 1, -1, -1), grad)
        p.end()
        return pm
