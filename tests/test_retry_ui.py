"""Offscreen test: the 3D-render failure retry surface.

Regression: the dead branch used to overwrite the clickable rich-text
links with plain text (nothing to click), and the retry surface did not
track the label geometry. It is now a real button panel that also keeps
the view size stable.

Run:  .venv/Scripts/python -m tests.test_retry_ui
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.core.crab_model import CrabModel
    from app.ui.crab_view import CrabView

    app = QApplication(sys.argv)
    view = CrabView(CrabModel())
    view.resize(632, 520)
    view.show()
    app.processEvents()
    failures: list[str] = []
    size_before = view.size()
    label_size = view._label.size()

    orig = CrabModel.render

    def boom(*_a, **_k):
        raise RuntimeError("forced GPU failure")

    CrabModel.render = boom
    try:
        view._render_error = True
        for _ in range(8):
            view._paint_frame()
            app.processEvents()
    finally:
        CrabModel.render = orig

    if not view._render_dead:
        failures.append("renderer never entered the dead state")
    if not view._retry_link.isVisible():
        failures.append("retry panel not visible in the dead state")
    if view._retry_link.geometry() != view._label.geometry():
        failures.append(f"retry panel geometry {view._retry_link.geometry()} "
                        f"!= label {view._label.geometry()}")
    if view.size() != size_before:
        failures.append(f"view size changed {size_before} -> {view.size()}")
    if view._label.size() != label_size:
        failures.append(f"label size changed {label_size} -> "
                        f"{view._label.size()}")
    if not view.retry_btn.isVisible():
        failures.append("retry button not visible")
    if not view.diag_btn.isVisible():
        failures.append("diagnose button not visible")

    QTest.mouseClick(view.retry_btn, Qt.MouseButton.LeftButton)
    app.processEvents()
    if view._render_dead or view._retry_link.isVisible():
        failures.append("retry button did not reset the render state")

    view.close()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("retry UI checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
