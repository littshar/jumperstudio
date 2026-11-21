"""Offscreen GUI smoke test: launch, screenshot the idle state, save PNG.

Run:  .venv/Scripts/python -m tests.screenshot_gui [out.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gui_idle.png")

    # Let the idle animation tick a few times so the crab is on screen.
    from PySide6.QtCore import QTimer
    grab_timer = QTimer()
    grab_timer.setSingleShot(True)
    grab_timer.timeout.connect(app.quit)
    QTimer.singleShot(1500, lambda: win.grab().save(str(out)))
    QTimer.singleShot(2200, app.quit)
    app.exec()

    if out.exists() and out.stat().st_size > 0:
        print(f"saved {out}")
        return 0
    print("screenshot failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
