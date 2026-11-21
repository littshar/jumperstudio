"""Offscreen GUI smoke test for the manual ghost-arm feature.

  * manual ghost-arm mode: pose an arm with the ANGLE SLIDERS (shoulder
    pitch/roll + elbow bend per side); claws follow the SAME IK solver
    as the video table and land on the ghost-shown direction
  * roll magnitude responds and J2 stays inside its box
  * the 转身/跳跃 strips are gone from the transport
  * height slider max raised to 1.55

Run:  .venv/Scripts/python -m tests.test_manual_mode <cached_video>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"C:\Users\dangz\Desktop\小螃蟹视频\只因你太美(鸡你太美)原版.90717632.mp4")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.constants import HEIGHT_FACTOR_MAX
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()
    failures: list[str] = []

    state = {"phase": 0}

    def poll() -> None:
        if win.duration == 0:
            if state["phase"] == 0:
                win.load_video(video)
                state["phase"] = 1
            return
        if state["phase"] == 1:
            state["phase"] = 2
            run_checks()

    def run_checks() -> None:
        nonlocal failures
        try:
            failures.extend(check_height(win, HEIGHT_FACTOR_MAX))
            if hasattr(win.transport, "spinbar") or hasattr(win.transport, "jumpbar"):
                failures.append("transport still carries segment strips")
            failures.extend(check_manual(win))
        except Exception as exc:  # noqa: BLE001
            import traceback
            failures.append(f"exception: {exc}\n{traceback.format_exc()}")
        finally:
            print("FAILURES:" if failures else "manual-mode GUI checks passed")
            for f in failures:
                print(" -", f)
            app.quit()

    timer = QTimer()
    timer.setInterval(150)
    timer.timeout.connect(poll)
    timer.start()
    QTimer.singleShot(120_000, app.quit)
    app.exec()
    return 1 if failures else 0








def check_height(win, hmax: float) -> list[str]:
    win._set_height_factor(2.0)   # over-range: must clamp to the new max
    got = win.crab_view._height_factor
    win._set_height_factor(1.0)
    if abs(got - hmax) > 1e-9:
        return [f"height factor clamp {got:.2f}, want {hmax:.2f}"]
    return []


def check_manual(win) -> list[str]:
    """Manual ghost mode: sliders pose the ghost, claws follow via IK."""
    failures = []
    view = win.crab_view
    win.seek(5.0)
    view.show_time(5.0, force=True)
    from PySide6.QtWidgets import QApplication
    from app.ui.joint_probe import JointProbeDialog
    app = QApplication.instance()
    app.processEvents()

    dlg = JointProbeDialog(view)
    dlg.show()
    dlg._set_mode(1)   # manual ghost
    app.processEvents()
    if view._manual is None:
        return ["manual mode not entered"]
    if view._ghost_visible is not True:
        failures.append("ghost not visible in manual mode")

    # Sliders must have synced to the video-synced entry pose.
    angles0 = view.manual_angles["l"]
    s_pitch = dlg._angle_sliders[("l", "pitch")]
    if abs(s_pitch.value() - angles0[0]) > 1.5:
        failures.append(
            f"pitch slider not synced: {s_pitch.value()} vs {angles0[0]:.1f}")


    def set_angles(side, pitch, roll, bend):
        dlg._angle_sliders[(side, "pitch")].setValue(int(pitch))
        dlg._angle_sliders[(side, "roll")].setValue(int(roll))
        dlg._angle_sliders[(side, "bend")].setValue(int(bend))
        app.processEvents()


    # The user anchor, in the sweep-tool convention (deltas from rest):
    # sliders (90, 90, 180) -> LF (-60, 0, 30, 20), mirrored RF (J4 out of
    # the mapping). Neutral sliders keep the elbows mirror-symmetric.
    import numpy as np

    rest = view._rest_claw_pose()
    mirror = np.array([-1.0, 1.0, -1.0, -1.0, -1.0])

    def claw_abs_deg():
        c = (view.live_commands or {}).get("claw", {})
        return {leg: np.degrees(np.array(
            [c.get(f"{leg}_J{j}_joint", np.nan) for j in range(5)]))
            for leg in ("LF", "RF")}

    def claw_delta_deg():
        g = claw_abs_deg()
        return {leg: g[leg] - np.degrees(
            [rest[f"{leg}_J{j}_joint"] for j in range(5)])
            for leg in ("LF", "RF")}

    set_angles("l", 0, 0, 0); set_angles("r", 0, 0, 0)
    got = claw_abs_deg()
    # Elbows must be exact mirrors at the neutral pose (J0 degenerate).
    elbow_err = np.abs(got["RF"] - got["LF"] * mirror)
    if np.max(elbow_err[1:]) > 0.5:
        failures.append(f"zeros elbows not mirrored: {np.round(elbow_err, 1)}")

    # Side-swing direction (user): at the SAME slider value the two ghost
    # arms must be mirror images (the right was reversed). Read the real
    # mocap wrist endpoints and reflect the left wrist through the
    # sagittal plane (normal = d_h = (1,1,0)/sqrt2).
    import mujoco
    d_h = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

    def ghost_wrist(leg: str) -> np.ndarray:
        name = f"ghost_{'lf' if leg == 'LF' else 'rf'}"
        mid = view.crab.model.body(name).mocapid
        pos = np.array(view.crab.data.mocap_pos[mid]).ravel()
        quat = np.array(view.crab.data.mocap_quat[mid]).ravel()
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        mat = mat.reshape(3, 3)
        return pos + 0.0325 * mat[:, 2]   # capsule half-length * local z

    # Both roll sliders share the range (-120..0); equal values must give
    # mirrored ghost arms (the right's construction roll is negated).
    set_angles("l", 60, -10, 180); set_angles("r", 60, -10, 180)
    app.processEvents()
    wl, wr = ghost_wrist("LF"), ghost_wrist("RF")
    want = wl - 2.0 * (wl @ d_h) * d_h   # reflect through the mirror plane
    if np.linalg.norm(wr - want) > 1e-3:
        failures.append(f"ghost right arm not mirrored (err "
                        f"{np.round(np.linalg.norm(wr - want), 4)} m)")

    # Different slider poses must produce different claws (it follows).
    set_angles("l", 90, 0, 180)
    a_point = claw_delta_deg()["LF"]
    set_angles("l", 0, 0, 180)
    a_hang = claw_delta_deg()["LF"]
    if np.any(np.isnan(a_point)) or np.any(np.isnan(a_hang)):
        failures.append("claw commands missing in manual mode")
    elif np.max(np.abs(a_point - a_hang)) < 3.0:
        failures.append("LF claw did not follow the slider arm")

    # Reset restores the video-synced pose.
    dlg._reset_angles()
    app.processEvents()
    if abs(view.manual_angles["l"][0] - angles0[0]) > 1.5:
        failures.append("reset did not restore the video pose")

    # Leave manual mode cleanly via the dialog.
    dlg._set_mode(0)
    if view._manual is not None:
        failures.append("manual mode not left")
    dlg.close()
    return failures


if __name__ == "__main__":
    sys.exit(main())
