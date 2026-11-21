"""Offscreen GUI test: the SuperMimic training secondary page.

Covers: top-right button -> whole-page switch; reference track + jump
segments imported and logged; the parameter dialog (settable fields +
memory/time estimates + recommendations); a real (tiny) PHYSICS training
run in the worker thread; the separate physics/preview render panes; the
live log; reward curves; the reference NPZ and the weights-export button
enabling only after training.

Run:  .venv/Scripts/python -m tests.test_train_page
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
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.core.crab_model import CrabModel
    from app.core.jumper_train import pretrained_checkpoint
    from app.core.retarget import build_crab_commands, compute_human_channels
    from app.core.train_worker import TrainParams
    from app.ui.main_window import MainWindow
    from app.ui.train_params_dialog import TrainParamsDialog
    from tests.test_retarget import kps_from_frames, make_kp

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1320, 860)
    win.show()
    failures: list[str] = []
    if win.windowTitle() != "Jumper Studio":
        failures.append("app title is not Jumper Studio")
    if pretrained_checkpoint() is None:
        failures.append("bundled pretrain checkpoint is missing")

    crab = CrabModel()
    frames = [make_kp(0.0, 0.9, 60.0, 110.0) for _ in range(60)]
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
    payload = {"kps": kps, "channels": ch, "table": table, "valid_pct": 95}

    win._person_payloads = [payload]
    win.duration = 20.0
    win.t = 5.0
    win._settings_state["multi"] = False
    win._settings_state["person"] = 1
    win.train_btn.setEnabled(True)
    win.train_btn.click()
    app.processEvents()

    if win.stack.currentIndex() != 1 or win._train_page is None:
        failures.append("training button did not switch to the secondary page")
        print("FAILURES:", failures)
        return 1
    page = win._train_page
    # This test runs the local CPU trial, not the GPU backend (which would
    # export a clip into the Jumper workspace and start a real training).
    page.backend_combo.setCurrentIndex(page.backend_combo.findData("es"))
    app.processEvents()
    if page._table is not table:
        failures.append("reference table not imported into the page")
    log_text = page.log.toPlainText()
    if "跳跃段 1 个" not in log_text:
        failures.append("jump segment not logged")
    if page.export_btn.isEnabled():
        failures.append("export button must start disabled")
    # Button state machine: a stopped run has a checkpoint -> quick preview
    # and replay enabled; while stopping Start stays disabled.
    import tempfile
    from app.core.jumper_train import JumperTrainWorker
    tmp_run = Path(tempfile.mkdtemp())
    fake_ckpt = tmp_run / "model_1.pt"
    fake_ckpt.write_bytes(b"x")
    page._run_dir = str(tmp_run)
    page._on_jumper_trained(str(tmp_run), str(fake_ckpt))
    page._sync_buttons()
    if not page.quick_preview_btn.isEnabled():
        failures.append("quick preview stayed disabled after a checkpoint")
    if page._continue_ckpt != str(fake_ckpt):
        failures.append("stopped run did not become the continue checkpoint")
    page._stop_pending = True
    page._sync_buttons()
    if page.start_btn.isEnabled():
        failures.append("start enabled while stop is still pending")
    page._stop_pending = False
    w = JumperTrainWorker(None, None, checkpoint=fake_ckpt)
    if w.checkpoint != fake_ckpt:
        failures.append("worker did not accept the resume checkpoint")
    import shutil
    shutil.rmtree(tmp_run, ignore_errors=True)
    # Undo the synthetic Jumper run so the ES-training wait below still sees
    # the real weights signal rather than this fake .pt path.
    page._weights = None
    page._weights_kind = "npz"
    page._run_dir = None
    page._continue_ckpt = None
    page.export_btn.setEnabled(False)
    page._sync_buttons()
    # Stopping must not leave the page reading "training".
    from types import SimpleNamespace
    saved = page._jumper
    page._jumper = SimpleNamespace(_stopping=True)
    page._auto_preview_after_stop = False
    page._on_jumper_finished()
    if "训练已停止" not in page.status_label.text():
        failures.append("stopped page still shows a training status")
    page._jumper = saved
    page.status_label.setText("动作轨道已导入")

    # Parameter dialog: settable fields + resource estimates.
    dlg = TrainParamsDialog(page._params, page._frame_count, True, page)
    info = dlg.info.text()
    rec = dlg.recommend.text()
    for needle in ("系统内存", "预估占用", "剩余内存", "预估时间"):
        if needle not in info:
            failures.append(f"resource info missing: {needle}")
    if "推荐：训练数量" not in rec:
        failures.append("recommendation line missing")
    if (dlg.sm_envs.value(), dlg.sm_iters.value()) != (2048, 300):
        failures.append("PPO fine-tune defaults are not envs=2048/iters=300")
    if "dance_base.pt" not in dlg.pretrain_label.text():
        failures.append("fixed pretrain checkpoint not shown")
    if "49152" not in dlg.ppo_batch.text():
        failures.append("PPO batch (num_envs x 24) not shown")
    # The page was switched to the local trial above, so the dialog opens on
    # that method; the GPU method is selectable from the same combo.
    if dlg.method.currentData() != "es":
        failures.append("training method did not follow the page's backend")
    if dlg.trial_box.isHidden() or not dlg.sm_box.isHidden():
        failures.append("method-specific parameter groups are not exclusive")
    if dlg.method.findData("jumper_ppo") < 0 or \
            dlg.method.findData("supermimic") < 0:
        failures.append("training method options incomplete")
    else:
        dlg.method.setCurrentIndex(dlg.method.findData("jumper_ppo"))
        if dlg.params(page._params).method != "jumper_ppo":
            failures.append("dialog did not return the selected GPU method")
    gpu_info = dlg.info.text()
    for needle in ("rsl_rl PPO", "GPU 微调", "批量", "显卡"):
        if needle not in gpu_info:
            failures.append(f"GPU config info missing: {needle}")
    if "MuJoCo 物理仍为 CPU" in gpu_info:
        failures.append("stale 'physics stays on CPU' note")
    if not dlg.trial_box.isHidden() or dlg.sm_box.isHidden():
        failures.append("PPO parameter group did not replace the trial group")
    if dlg.method.findData("supermimic") < 0:
        failures.append("SuperMimic method option missing")
    else:
        sm_idx = dlg.method.findData("supermimic")
        if dlg.method.model().item(sm_idx).isEnabled():
            failures.append("SuperMimic method must be disabled until the "
                            "crab port exists")
    sm_idx = dlg.method.findData("supermimic")
    tip = str(dlg.method.itemData(sm_idx, Qt.ItemDataRole.ToolTipRole) or "")
    if "G1" not in tip and "移植" not in tip:
        failures.append("SuperMimic port status not in the item tooltip")
    sm = dlg.params(page._params)
    if (sm.sm_num_envs, sm.sm_batch_size, sm.sm_iterations) != (2048, 256, 300):
        failures.append("dialog did not return the GPU fine-tune config")
    if sm.method != "jumper_ppo":
        failures.append("dialog did not return the training method")
    dlg.precision.setValue(6.0)
    dlg.population.setValue(5)
    dlg.iterations.setValue(9)
    dlg.hand_safe.setValue(40)
    p = dlg.params(page._params)
    if (p.population, p.iterations, round(p.precision_deg, 1),
            round(p.hand_safe_m * 1000)) != (5, 9, 6.0, 40):
        failures.append(f"dialog did not return edited params: {p}")
    dlg.close()

    # Tiny physics training run with rendering on (separate panes).
    page._params_override = TrainParams(
        iterations=2, population=2, elites=1, stride=8, max_frames=12,
        render=True, seed=3, substeps=6)
    page.start_training()
    t0 = time.time()
    while time.time() - t0 < 120:
        app.processEvents()
        if page._weights:
            break
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.02)

    if not page._weights or not Path(page._weights).exists():
        failures.append("weights were not produced")
    if ".npz" not in page.log.toPlainText():
        failures.append("reference NPZ path not logged")
    if not page.export_btn.isEnabled():
        failures.append("export button not enabled after training")
    if page.progress.value() != 2:
        failures.append(f"progress stayed at {page.progress.value()}/2")
    if len(page.rewards._hist) < 2:
        failures.append("reward curves got no data")
    elif not all(k in page.rewards._hist[0]
                 for k in ("return_best", "track", "stab", "hands")):
        failures.append("reward metrics incomplete")
    if page.physics_img.pixmap().isNull():
        failures.append("physics render pane got no frame")
    if page.preview_img.pixmap().isNull():
        failures.append("policy preview pane got no frame")
    if "训练完成" not in page.log.toPlainText():
        failures.append("log has no completion line")
    if page._dance is None or len(page._dance.get("times", [])) < 2:
        failures.append("full-dance physics playback was not recorded")
    from app.ui.reward_dialog import PPO_EXTRA_WEIGHTS, RewardDialog
    from app.core.jumper_train import _ENV_PY
    rd = RewardDialog(None)
    rd.ppo_table.add_row("joint_vel_l2", PPO_EXTRA_WEIGHTS["joint_vel_l2"])
    if "joint_vel_l2" not in rd.weights()["ppo"]:
        failures.append("added reward term did not reach the weights")
    rd.close()
    if "joint_vel_l2" not in _ENV_PY or "EXTRA_REWARDS" not in _ENV_PY:
        failures.append("added reward term is not generated into the task")
    if not page.replay_btn.isEnabled():
        failures.append("replay button not enabled after training")
    if "训练后完整舞蹈" not in page.log.toPlainText():
        failures.append("dance playback line missing from the log")
    # Mouse orbit on the policy preview.
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    az0 = page._policy_cam[0]
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(20, 20),
                     QPointF(20, 20), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(page.preview_img, ev)
    ev2 = QMouseEvent(QEvent.Type.MouseMove, QPointF(50, 20),
                      QPointF(50, 20), Qt.MouseButton.NoButton,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(page.preview_img, ev2)
    ev3 = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(50, 20),
                      QPointF(50, 20), Qt.MouseButton.LeftButton,
                      Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(page.preview_img, ev3)
    if abs(page._policy_cam[0] - az0) < 1.0:
        failures.append("policy preview mouse drag did not orbit the camera")
    if page._dance_crab is None:
        failures.append("dance playback model not created")
    if not page._dance_timer.isActive():
        failures.append("dance playback timer not running")
    if not page.preview_pause_btn.isEnabled():
        failures.append("policy preview pause button stayed disabled")
    page._toggle_preview_pause(True)
    if page._dance_timer.isActive() or page.preview_pause_btn.text() != "播放":
        failures.append("policy preview did not pause")
    page._toggle_preview_pause(False)
    if not page._dance_timer.isActive():
        failures.append("policy preview did not resume")

    # Shared right window: log while training, source video for the result;
    # the log sticks to its newest line without a manual drag.
    if page.right_stack.currentIndex() != 1:
        failures.append("result view did not switch to the source video")
    page.log_tab.click()
    app.processEvents()
    if page.right_stack.currentIndex() != 0 or not page.log_tab.isChecked():
        failures.append("log tab did not switch the shared window back")
    page.video_tab.click()
    app.processEvents()
    if page.right_stack.currentIndex() != 1 or not page.video_tab.isChecked():
        failures.append("video tab did not switch the shared window")
    for i in range(80):
        page.log.appendPlainText(f"scroll probe {i}")
    page._log("scroll probe last")
    app.processEvents()
    bar = page.log.verticalScrollBar()
    if bar.value() < bar.maximum():
        failures.append(f"log did not stick to the bottom "
                        f"({bar.value()}/{bar.maximum()})")

    page.back_btn.click()
    app.processEvents()
    if win.stack.currentIndex() != 0:
        failures.append("back button did not return to the main page")

    win.close()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("training page checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
