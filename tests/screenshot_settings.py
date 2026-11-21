"""Offscreen GUI test: settings-driven look (background + overlay + curves).

Shows the idle crab with a preset background and the mapping overlay on,
curves panel off, then screenshots. No video needed.

Run:  .venv/Scripts/python -m tests.screenshot_settings [preset_id] [out.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    preset = sys.argv[1] if len(sys.argv) > 1 else "sunset"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("gui_settings.png")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()

    def apply() -> None:
        win.crab_view.set_mapping_visible(True)
        win.crab_view.set_background(preset)
        win.set_curves_visible(False)
        win.statusBar().showMessage(f"预览设置：背景 {preset} · 映射标注开 · 曲线关")

    def grab() -> None:
        ok = win.grab().save(str(out))
        print(f"saved {out} ok={ok}")
        app.quit()

    QTimer.singleShot(600, apply)   # let a few idle frames render first
    QTimer.singleShot(2200, grab)

    QTimer.singleShot(30_000, app.quit)
    app.exec()

    if out.exists() and out.stat().st_size > 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
