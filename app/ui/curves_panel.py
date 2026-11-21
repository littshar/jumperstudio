"""Bottom curve panel: torso Euler angles + both hands' motion over time."""

from __future__ import annotations

import matplotlib
import numpy as np
from PySide6.QtCore import Signal

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# CJK-capable font stack (Windows first) + dark theme defaults.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

from app.core.retarget import HumanChannels


class CurvesPanel(FigureCanvasQTAgg):
    """Two stacked plots with a shared time cursor.

    Top: torso pitch / roll / yaw (deg) -- yaw is a weak 2D approximation.
    Bottom: the remapped claw joint commands J0..J4 (deg) -- LF solid,
    RF dashed -- i.e. what the claws are actually told to do.
    Click anywhere to seek to that time.
    """

    seek_requested = Signal(float)

    _BG = "#0d1117"
    _FG = "#8b949e"

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 3), facecolor=self._BG,
                             constrained_layout=True)
        super().__init__(self.figure)
        self.ax_torso = self.figure.add_subplot(2, 1, 1)
        self.ax_hands = self.figure.add_subplot(2, 1, 2, sharex=self.ax_torso)

        for ax in (self.ax_torso, self.ax_hands):
            ax.set_facecolor(self._BG)
            ax.tick_params(colors=self._FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#21262d")
            ax.grid(alpha=0.2, color=self._FG)

        self.ax_torso.set_ylabel("躯干欧拉角 °", color=self._FG)
        self.ax_hands.set_ylabel("手部动作 °", color=self._FG)
        self.ax_hands.set_xlabel("时间 s", color=self._FG)

        self._cursor_t = self.ax_torso.axvline(0, color="#e6b450", lw=1)
        self._cursor_b = self.ax_hands.axvline(0, color="#e6b450", lw=1)
        self._t_max = 1.0

        self._placeholder()
        self.mpl_connect("button_press_event", self._on_click)
        self.setMinimumHeight(180)

    def _placeholder(self) -> None:
        for ax, text in ((self.ax_torso, "躯干曲线"),
                         (self.ax_hands, "钳关节曲线")):
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                    va="center", fontsize=11, color="#586e88")
            ax.set_xticks([])
            ax.set_yticks([])

    def _style_axes(self) -> None:
        for ax in (self.ax_torso, self.ax_hands):
            ax.set_facecolor(self._BG)
            ax.tick_params(colors=self._FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#21262d")
            ax.grid(alpha=0.2, color=self._FG)

    # -------------------------------------------------------------- data ----
    def set_channels(self, ch: HumanChannels) -> None:
        t = ch.times
        self._t_max = float(t[-1]) if len(t) else 1.0

        ax = self.ax_torso
        ax.clear()
        ax.plot(t, ch.torso_pitch, color="#e06c75", lw=1, label="pitch 前后倾")
        ax.plot(t, ch.torso_roll, color="#61afef", lw=1, label="roll 左右倾")
        ax.plot(t, ch.torso_yaw, color="#c678dd", lw=1, ls="--",
                label="yaw 转体")
        # Air gestures, 0-100 %. Jump needs both feet up; a raise is one
        # ankle high while grounded.
        ax.plot(t, ch.jump * 100.0, color="#98c379", lw=1.4, label="跳跃 %")
        ax.plot(t, ch.raise_l * 100.0, color="#56b6c2", lw=1, ls=":",
                label="抬左腿 %")
        ax.plot(t, ch.raise_r * 100.0, color="#d19a66", lw=1, ls=":",
                label="抬右腿 %")
        ax.set_ylabel("躯干欧拉角 ° · 跳跃/抬腿 (%)", color=self._FG)
        leg = ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.3)
        for text in leg.get_texts():
            text.set_color("#c9d1d9")
        self._style_axes()
        self.draw_idle()

    def set_table(self, table, ranges=None) -> None:
        """Bottom plot: the remapped claw joint commands (deg).

        ranges: optional {joint_index: (lo_deg, hi_deg)} usable joint
        limits, drawn as dotted lines so saturation is visible.
        """
        t = table.times
        self._t_max = float(t[-1]) if len(t) else 1.0

        ax = self.ax_hands
        ax.clear()
        joint_colors = {0: "#61afef", 1: "#98c379", 2: "#e5c07b",
                        3: "#e06c75", 4: "#c678dd"}
        for j in range(5):
            col = joint_colors[j]
            ax.plot(t, np.degrees(table.claw[f"LF_J{j}_joint"]),
                    color=col, lw=1, label=f"J{j}")
            ax.plot(t, np.degrees(table.claw[f"RF_J{j}_joint"]),
                    color=col, lw=1, ls="--")
            if ranges and j in ranges:
                lo, hi = ranges[j]
                for lim in (lo, hi):
                    ax.axhline(lim, color=col, lw=0.7, ls=":",
                               alpha=0.45)
        ax.set_ylabel("钳关节指令 °", color=self._FG)
        ax.set_xlabel("时间 s · 实线 LF 左钳 / 虚线 RF 右钳 · 点线=关节限位",
                      color=self._FG)
        leg = ax.legend(loc="upper right", fontsize=7, ncol=5, framealpha=0.3)
        for text in leg.get_texts():
            text.set_color("#c9d1d9")
        ax.set_xlim(0, self._t_max)
        ax.margins(y=0.10)
        self._style_axes()

        # Re-create cursors after clear().
        self._cursor_t = self.ax_torso.axvline(0, color="#e6b450", lw=1)
        self._cursor_b = self.ax_hands.axvline(0, color="#e6b450", lw=1)
        self.draw_idle()

    # ------------------------------------------------------------ cursor ----
    def set_cursor(self, t: float) -> None:
        if self._cursor_t is None:
            return
        self._cursor_t.set_xdata([t])
        self._cursor_b.set_xdata([t])
        self.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes not in (self.ax_torso, self.ax_hands):
            return
        if event.xdata is None:
            return
        self.seek_requested.emit(float(max(0.0, event.xdata)))
