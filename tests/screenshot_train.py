"""Offscreen screenshot of the SuperMimic training page after a short run.

Run:  .venv/Scripts/python -m tests.screenshot_train [out.png]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CRAB_SKIP_ENV_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.core.crab_model import CrabModel
    from app.core.retarget import build_crab_commands, compute_human_channels
    from app.core.train_worker import TrainParams
    from app.ui.main_window import MainWindow
    from tests.test_retarget import kps_from_frames, make_kp

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()

    crab = CrabModel()
    frames = [make_kp(0.0, 0.9, 30 + 60 * (1 - np.cos(2 * np.pi * k / 60)),
                      125 + 55 * np.cos(2 * np.pi * k / 30))
              for k in range(180)]
    kps = kps_from_frames(frames)
    ch = compute_human_channels(kps)
    table = build_crab_commands(ch, crab, claw_ik=True)
    n = len(table.times)
    jump = np.zeros(n)
    jump[n // 3: n // 2] = 0.8
    mask = np.zeros(n, dtype=bool)
    mask[n // 3: n // 2] = True
    table.jump = jump
    table.jump_mask = mask

    win._person_payloads = [{"kps": kps, "channels": ch, "table": table,
                             "valid_pct": 95}]
    win._current_video = Path(
        r"C:\Users\dangz\Desktop\小螃蟹视频\只因你太美(鸡你太美)原版.90717632.mp4")
    win.duration = 20.0
    win.t = 5.0
    win._settings_state["multi"] = False
    win._settings_state["person"] = 1
    win.train_btn.setEnabled(True)
    win.train_btn.click()
    page = win._train_page
    # Screenshot the local CPU trial (the GPU backend would spend minutes
    # preparing and start a real run in the Jumper workspace).
    page.backend_combo.setCurrentIndex(page.backend_combo.findData("es"))
    page._params_override = TrainParams(
        iterations=8, population=5, elites=2, stride=4, max_frames=60,
        render=True, seed=2)
    page.start_training()
    t0 = time.time()
    while time.time() - t0 < 120:
        app.processEvents()
        if page._weights:
            break
        time.sleep(0.02)
    for _ in range(30):
        app.processEvents()
        time.sleep(0.02)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "tests/gui_train.png")
    win.grab().save(str(out))
    print(f"saved {out}")
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
