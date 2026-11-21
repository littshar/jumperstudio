"""Offscreen GUI test with a cached video: play, then screenshot.

Requires the video to already be analysed (cache hit) so the worker returns
instantly -- run the pipeline on the video first.

Run:  .venv/Scripts/python -m tests.screenshot_gui_playing <video> [out.png]
                                          [--bg PRESET] [--mapping] [--no-curves]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    bg = None
    if "--bg" in sys.argv:
        bg = sys.argv[sys.argv.index("--bg") + 1]
    video = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else Path("gui_playing.png")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()

    state = {"shots": 0, "saved": False}

    def try_load() -> None:
        if win.duration > 0:
            return  # analysis done (cache hit), playing
        if state["shots"] == 0:
            win.load_video(video)
            state["shots"] += 1

    def apply_look() -> None:
        if bg:
            win.crab_view.set_background(bg)
        if "--mapping" in flags:
            win.crab_view.set_mapping_visible(True)
        if "--ghost" in flags:
            win.crab_view.set_ghost_visible(True)
        if "--detect" in flags:
            win.video_panel.set_detect_visible(True)
        if "--no-curves" in flags:
            win.set_curves_visible(False)

    # Poll until the video is playing, then screenshot after a few seconds.
    def poll() -> None:
        try_load()
        if win.duration > 0:
            if state["shots"] == 1:
                state["shots"] = 2
                apply_look()
            if win.t > 4.0 and not state["saved"]:
                ok = win.grab().save(str(out))
                state["saved"] = True
                print(f"saved {out} ok={ok} t={win.t:.2f}s")
                app.quit()

    timer = QTimer()
    timer.setInterval(200)
    timer.timeout.connect(poll)
    timer.start()

    # Hard timeout: 5 minutes.
    QTimer.singleShot(300_000, app.quit)
    app.exec()

    if out.exists() and out.stat().st_size > 0:
        print("done")
        return 0
    print("screenshot failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
