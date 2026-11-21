"""Offscreen GUI test: follow-person switching + mode-switch refresh.

Covers the user's complaints:
  * in a multi-person video the 单人跟随 selector must switch the crab;
  * switching single <-> multi while PAUSED must repaint immediately
    (no "press play first"), and the combo must keep the saved person;
  * no re-analysis is involved (cached payloads only).

Run:  .venv/Scripts/python -m tests.test_person_follow
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.core.crab_model import CrabModel
    from app.core.retarget import build_crab_commands, compute_human_channels
    from app.ui.main_window import MainWindow
    from app.ui.settings_dialog import SettingsDialog
    from tests.test_retarget import kps_from_frames, make_kp

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    failures: list[str] = []

    crab = CrabModel()
    payloads = []
    for i, dx in enumerate((0.0, 120.0)):
        frames = []
        for k in range(20):
            f = make_kp(0.0, 0.9, 60.0 + dx, 110.0)
            for name in f:
                f[name] = f[name].copy()
                f[name][0] += dx if name != "nose" else 0.0
            frames.append(f)
        kps = kps_from_frames(frames)
        ch = compute_human_channels(kps)
        payloads.append({
            "kps": kps, "channels": ch,
            "table": build_crab_commands(ch, crab, claw_ik=False),
            "valid_pct": 90 - 10 * i,
        })

    win._person_payloads = payloads
    win.duration = 20.0
    win.t = 5.0

    def cast_labels() -> list:
        return [a.table for a in win.crab_view.actors]

    def check(label: str, ok: bool) -> None:
        if not ok:
            failures.append(label)

    # --- single mode, follow person 1 / 2 ---------------------------------
    win._settings_state["multi"] = False
    win._settings_state["person"] = 1
    win.crab_view._last_t = None
    win._apply_person_mode()
    check("follow person 1 casts person 1",
          len(win.crab_view.actors) == 1
          and cast_labels()[0] is payloads[0]["table"])
    check("follow person 1 highlights video person 1",
          win.video_panel._follow == 0)
    check("paused mode switch repainted the crab view",
          abs((win.crab_view._last_t or -1) - win.t) < 1e-6)

    win._settings_state["person"] = 2
    win._apply_person_mode()
    check("follow person 2 casts person 2",
          len(win.crab_view.actors) == 1
          and cast_labels()[0] is payloads[1]["table"])
    check("follow person 2 highlights video person 2",
          win.video_panel._follow == 1)

    # --- auto (0) -> best/first ------------------------------------------
    win._settings_state["person"] = 0
    win._apply_person_mode()
    check("auto follows the first/largest person",
          cast_labels()[0] is payloads[0]["table"])

    # --- multi mode: everyone, and back ----------------------------------
    win._settings_state["multi"] = True
    win._apply_person_mode()
    check("multi casts every person", len(win.crab_view.actors) == 2)
    check("multi highlights everyone", win.video_panel._follow == -1)
    check("multi mode repainted", win.crab_view._last_t is not None)

    win._settings_state["multi"] = False
    win._settings_state["person"] = 2
    win._apply_person_mode()
    check("back to single follows the saved person",
          len(win.crab_view.actors) == 1
          and cast_labels()[0] is payloads[1]["table"])

    # --- settings combo keeps the saved selection + emits 1-based --------
    from PySide6.QtCore import QSettings
    key = "retarget/follow_person"
    saved = QSettings().value(key, None)
    dlg = SettingsDialog(win._settings_state, win)
    dlg.set_person_count(2, [90, 80], current=2)
    combo = dlg.person_combo
    check("combo restores the saved person", combo.currentData() == 2)

    emitted: list[int] = []
    dlg.person_changed.connect(emitted.append)
    combo.setCurrentIndex(combo.findData(1))
    check("combo emits the 1-based person index", emitted == [1])
    app.processEvents()

    # Restore the user's persisted value (the dialog writes on change).
    if saved is None:
        QSettings().remove(key)
    else:
        QSettings().setValue(key, saved)

    dlg.close()
    win.close()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("person follow / mode-switch checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
