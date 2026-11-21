"""Write the Jumper dance task for curriculum stage A or B.

Loads the app's cached Sports2D analysis for the given video and calls the
same prepare_dance_task the training page uses, with the stage weights.

    python tools/make_curriculum_task.py --video <mp4> --stage a
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CRAB_SKIP_ENV_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--stage", choices=("a", "b", "jump"), default="a")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication

    from app.core.jumper_train import (
        CURRICULUM_PPO_WEIGHTS, INTEGRATION_PPO_WEIGHTS, JUMP_PPO_WEIGHTS,
        prepare_dance_task,
    )
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.load_video(args.video)
    t0 = time.time()
    while time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.05)
        if getattr(win, "_person_payloads", None):
            break
    people = getattr(win, "_person_payloads", None) or []
    if not people:
        print("no analysed person found", file=sys.stderr)
        return 1
    weights = (CURRICULUM_PPO_WEIGHTS if args.stage == "a"
               else JUMP_PPO_WEIGHTS if args.stage == "jump"
               else INTEGRATION_PPO_WEIGHTS)
    info = prepare_dance_task(people[0]["table"], win.crab_view.crab,
                              log=print, reward_weights=weights)
    print(f"stage {args.stage} task ready: {info['duration_s']:.1f} s")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
