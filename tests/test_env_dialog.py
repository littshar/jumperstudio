"""Offscreen smoke test for the environment dialog.

Run:  .venv/Scripts/python -m tests.test_env_dialog
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.ui.environment_dialog import EnvironmentDialog

    app = QApplication(sys.argv)
    dlg = EnvironmentDialog(first_run=False)
    dlg.show()
    t0 = time.time()
    while time.time() - t0 < 30:
        app.processEvents()
        if (dlg._diag is not None and not dlg._diag.isRunning()
                and dlg.table.rowCount() > 0):
            break
        time.sleep(0.03)
    failures = []
    if dlg.table.rowCount() < 5:
        failures.append(f"diagnosis produced {dlg.table.rowCount()} rows")
    if "训练环境" not in dlg.summary.text():
        failures.append("summary has no environment status")
    if not dlg.close_btn.isEnabled():
        failures.append("close button stayed disabled after diagnosis")
    dlg.close()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("environment dialog checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
