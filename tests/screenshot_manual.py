"""Offscreen screenshot: manual ghost-arm mode with the angle sliders.

Loads the default (cached) video, enters 手动幽灵 mode, poses the left
arm with sliders, and grabs the whole window + dialog.

Run:  .venv/Scripts/python -m tests.screenshot_manual [out.png]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    video = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        r"C:\Users\dangz\Desktop\小螃蟹视频\只因你太美(鸡你太美)原版.90717632.mp4")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.ui.joint_probe import JointProbeDialog
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()
    dlg = JointProbeDialog(win.crab_view)
    dlg.show()

    state = {"phase": 0}

    def poll() -> None:
        if state["phase"] == 0:
            if win.duration == 0:
                win.load_video(video)
                return
            state["phase"] = 1
            win.seek(5.0)
            dlg._set_mode(1)          # manual ghost
            QApplication.processEvents()
            # Pose: right arm raised sideways, elbows bent.
            dlg._angle_sliders[("r", "roll")].setValue(-80)
            dlg._angle_sliders[("r", "bend")].setValue(70)
            dlg._angle_sliders[("l", "pitch")].setValue(60)
            dlg._angle_sliders[("l", "bend")].setValue(120)
            QApplication.processEvents()
            out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gui_manual.png")
            dlg.move(win.x() + win.width() - dlg.width() - 40,
                     win.y() + 120)
            win.grab().save(str(out))
            print(f"saved {out}")
            app.quit()

    t = QTimer()
    t.setInterval(120)
    t.timeout.connect(poll)
    t.start()
    QTimer.singleShot(120_000, app.quit)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
