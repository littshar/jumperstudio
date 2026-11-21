"""Transport bar: play/pause, draggable progress slider, time label.

The 转身/跳跃 segment strips were REMOVED (user, round-18): the auto
front/back facing tracks turns directly and detected hops play on their
own, so no per-segment pills are needed.
"""


from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)


def _fmt(sec: float) -> str:
    sec = max(0.0, sec)
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}"


class JumpSlider(QSlider):
    """Slider with click-to-jump AND drag-to-scrub on the groove.

    Vanilla QSlider only steps by page-step on click, and its built-in
    handle-drag ignores presses that land on the groove. Both interactions
    are handled here: a press seeks straight to the click point and starts
    a scrub (paused playback, live preview); moves track the mouse; release
    ends the scrub. Only the standard pressed / moved / released signals
    are emitted, so the pause-while-scrubbing logic is unchanged.
    """

    def _value_at(self, x: float) -> int:
        handle = 16
        span = max(self.width() - handle, 1)
        x = min(max(x - handle / 2, 0.0), float(span))
        return self.minimum() + round(
            x / span * (self.maximum() - self.minimum()))

    def _scrubbing(self, event) -> bool:
        return (event.buttons() & Qt.MouseButton.LeftButton
                and self.isEnabled() and self.maximum() > self.minimum())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (event.button() == Qt.MouseButton.LeftButton and self.isEnabled()
                and self.maximum() > self.minimum()):
            value = self._value_at(event.position().x())
            self.setSliderPosition(value)
            self.sliderPressed.emit()
            self.sliderMoved.emit(value)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._scrubbing(event):
            value = self._value_at(event.position().x())
            if value != self.value():
                self.setSliderPosition(value)
                self.sliderMoved.emit(value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if (event.button() == Qt.MouseButton.LeftButton and self.isEnabled()
                and self.maximum() > self.minimum()):
            self.sliderReleased.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TransportBar(QWidget):
    """Signals: play_toggled(bool playing), seek_requested(float seconds)."""

    play_toggled = Signal(bool)
    seek_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._was_playing = False
        self._syncing = False

        self.btn = QPushButton("▶ 播放")
        self.btn.setFixedWidth(88)
        self.btn.setEnabled(False)
        self.btn.clicked.connect(self._on_btn)

        self.slider = JumpSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._on_pressed)
        self.slider.sliderMoved.connect(self._on_moved)
        self.slider.sliderReleased.connect(self._on_released)

        slider_col = QVBoxLayout()
        slider_col.setContentsMargins(0, 0, 0, 0)
        slider_col.setSpacing(1)
        slider_col.addWidget(self.slider)

        self.label = QLabel("00:00.0 / 00:00.0")

        self._leading = 0  # insert index for buttons added via add_leading()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.btn)
        layout.addLayout(slider_col, stretch=1)
        layout.addWidget(self.label)

    def add_leading(self, widget) -> None:
        """Place a widget at the far left of the bar (import/settings)."""
        lay = self.layout()
        lay.insertWidget(self._leading, widget)
        lay.insertSpacing(self._leading + 1, 8)
        self._leading += 2

    # ------------------------------------------------------------- state ----
    def set_duration(self, seconds: float) -> None:
        self._duration = max(seconds, 0.0)
        self.slider.setRange(0, int(self._duration * 1000))
        self.slider.setEnabled(self._duration > 0)
        self.btn.setEnabled(self._duration > 0)
        self.label.setText(f"00:00.0 / {_fmt(self._duration)}")

    def set_playing(self, playing: bool) -> None:
        self.btn.setText("⏸ 暂停" if playing else "▶ 播放")

    def update_time(self, t: float, playing: bool) -> None:
        """Called by the playback loop -- reflects, never triggers."""
        self._syncing = True
        self.slider.setValue(int(t * 1000))
        self._syncing = False
        self.label.setText(f"{_fmt(t)} / {_fmt(self._duration)}")
        self.set_playing(playing)

    # ------------------------------------------------------------ events ----
    def _on_btn(self) -> None:
        self.play_toggled.emit(self.btn.text().startswith("▶"))

    def _on_pressed(self) -> None:
        self._was_playing = self.btn.text().startswith("⏸")
        if self._was_playing:
            self.play_toggled.emit(False)  # pause while scrubbing

    def _on_moved(self, value: int) -> None:
        if self._syncing:
            return
        self.seek_requested.emit(value / 1000.0)

    def _on_released(self) -> None:
        if self._was_playing:
            self.play_toggled.emit(True)
