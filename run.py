"""crab-dance entry point.

Usage:
    python run.py                # launch the GUI
    python run.py --fetch        # download the jumper robot model (first run)
    python run.py --selftest     # model + retarget offline checks
    python run.py --video PATH   # launch the GUI with a video preloaded
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    args = sys.argv[1:]

    if "--fetch" in args:
        from app.core.fetch_assets import fetch_jumper_assets
        return 0 if fetch_jumper_assets(force="--force" in args) else 1

    if "--selftest" in args:
        from app.core.crab_model import CrabModel
        errors = CrabModel().selftest()
        if errors:
            print("SELFTEST FAILURES:")
            for e in errors:
                print(" -", e)
            return 1
        print("crab model selftest passed")
        from tests.test_retarget import main as retarget_main
        return retarget_main()

    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    from PySide6.QtCore import QTimer

    for i, a in enumerate(args):
        if a == "--video" and i + 1 < len(args):
            video = Path(args[i + 1])
            QTimer.singleShot(0, lambda v=video: win.load_video(v))
            break
    else:
        # No explicit video: reopen the most recent dance by default.
        QTimer.singleShot(200, win.load_last_video)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
