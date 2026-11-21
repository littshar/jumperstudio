"""Training parameter dialog: settable knobs + resource estimation.

Shows the machine's memory / GPU, the estimated memory footprint, the
remaining memory and the estimated training time for the current settings,
and computes a recommended population / epoch count from the measured
physics-step speed and a ~75 s budget.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import psutil
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QVBoxLayout,
)

from app.core.jumper_train import (
    cached_iter_seconds, pretrained_checkpoint, pretrained_meta,
)
from app.ui.reward_dialog import RewardDialog, show_weight_versions
from app.core.train_worker import (
    TrainParams, benchmark_frame_seconds, cached_frame_seconds,
)

BUDGET_SECONDS = 75.0


_GPU_CACHE: str | None = None
_BACKEND_CACHE: tuple[bool, str] | None = None

#: The community SuperMimic re-implementation (world model + imagined
#: rollouts). G1-only as installed; the dialogue says exactly what is missing
#: for the crab rather than hiding the option.
_SUPERMIMIC_DIR = Path(r"D:\supermimic_win\allmimic")


def _supermimic_status() -> str:
    if _SUPERMIMIC_DIR.is_dir():
        return ("allmimic 已安装 · 仅 Unitree G1 · 29 DoF。"
                "22-DoF 螃蟹的机器人配置、世界模型维度、动作库格式"
                "与关节权重尚未移植，SuperMimic 还不能训练螃蟹。")
    return "未找到 allmimic。SuperMimic 不可用，当前只有 PPO 能训练螃蟹。"


def _gpu_info() -> str:
    """Cached nvidia-smi query -- never re-run it on every keystroke."""
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                           if os.name == "nt" else 0))
        line = (out.stdout or "").strip().splitlines()
        if line:
            _GPU_CACHE = line[0].strip()
            return _GPU_CACHE
    except Exception:  # noqa: BLE001 - GPU info is best effort
        pass
    _GPU_CACHE = "未检测到 NVIDIA 显卡"
    return _GPU_CACHE


def _training_backend() -> tuple[bool, str]:
    """(usable, description) for the GPU training backend the start button runs.

    This is the **Jumper/mjlab workspace**, not this Python's torch: the app
    launches the training in `D:\\jumper_ws\\.venv`, so the app's own venv may
    have no torch at all (e.g. when started with a system Python) while GPU
    training works perfectly. Checking torch here was reporting "no GPU" on a
    machine that was training on one.
    """
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE
    try:
        from app.core.jumper_train import jumper_status
        ok, why = jumper_status()
        if ok:
            _BACKEND_CACHE = (
                True,
                f"Jumper/mjlab · warp:cuda · {why}")
        else:
            _BACKEND_CACHE = (False, why)
    except Exception as exc:  # noqa: BLE001 - the notice is best effort
        _BACKEND_CACHE = (False, f"无法检查 Jumper 训练环境：{exc}")
    return _BACKEND_CACHE


class _EstimateWorker(QThread):
    """nvidia-smi + the physics benchmark, off the GUI thread."""

    done = Signal(object)

    def __init__(self, substeps: int, parent=None):
        super().__init__(parent)
        self.substeps = int(substeps)

    def run(self) -> None:
        try:
            gpu = _gpu_info()
            _ok, desc = _training_backend()
            per_frame = benchmark_frame_seconds(self.substeps)
            self.done.emit((gpu, desc, per_frame))
        except Exception as exc:  # noqa: BLE001
            self.done.emit((_gpu_info(), f"检测失败：{exc}", None))


class TrainParamsDialog(QDialog):
    def __init__(self, params: TrainParams, frame_count: int,
                 render: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练参数")
        self.setModal(True)
        self.setMinimumWidth(520)
        self._frame_count = max(frame_count, 2)
        self._render = render
        self._estimate_worker: _EstimateWorker | None = None
        self._measured = None

        # ---- training method (what the start button actually runs)
        self.method = QComboBox()
        self.method.addItem("PPO", "jumper_ppo")
        self.method.addItem("SuperMimic 世界模型", "supermimic")
        sm_idx = self.method.findData("supermimic")
        self.method.model().item(sm_idx).setEnabled(False)
        self.method.setItemData(sm_idx, _supermimic_status(),
                                Qt.ItemDataRole.ToolTipRole)
        self.method.addItem("Smoke", "es")
        want = params.method if params.method in ("jumper_ppo", "es") \
            else "jumper_ppo"
        self.method.setCurrentIndex(max(0, self.method.findData(want)))
        self.method.currentIndexChanged.connect(self._on_method_changed)
        method_form = QFormLayout()
        method_form.addRow("训练方法", self.method)
        method_box = QGroupBox("训练方法")
        method_box.setLayout(method_form)

        # ---- Jumper PPO training config
        self.sm_envs = QSpinBox()
        self.sm_envs.setRange(1, 8192)
        self.sm_envs.setValue(params.sm_num_envs)
        self.sm_envs.valueChanged.connect(self._update_ppo_batch)
        self.sm_iters = QSpinBox()
        self.sm_iters.setRange(1, 1_000_000)
        self.sm_iters.setValue(params.sm_iterations)
        self.ppo_batch = QLabel()
        self.ppo_batch.setStyleSheet("color:#8b949e;")
        self.ppo_batch.setToolTip(
            "rsl_rl 的 PPO 每轮先用 num_envs 个环境各走 "
            "num_steps_per_env=24 步采样，再分成 4 个 mini-batch 更新；"
            "Jumper 没有独立的 batch_size 参数。")
        # One bulk dial: left = fast/rough, right = fine/slow. It only
        # touches the numeric fields; no benchmark or GPU query runs here.
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(0, 100)
        # Start the dials at the values the page already has, so reopening the
        # dialog does not silently reset a custom iteration count.
        iters0 = max(1, int(params.sm_iterations))
        self.quality_slider.setValue(round(
            (min(max(iters0, 20), 600) - 20) / 580.0 * 100))
        self.quality_slider.setToolTip(
            "训练精细度：最左更快、更粗；最右更精细、更久。中间线性过渡")
        self.vram_slider = QSlider(Qt.Orientation.Horizontal)
        self.vram_slider.setRange(10, 90)
        envs0 = max(1, int(params.sm_num_envs))
        self.vram_slider.setValue(round(min(max(
            (0.0019 * envs0 + 0.9) / 8.2 * 100.0, 10.0), 90.0)))
        self.vram_slider.setToolTip(
            "显存占用上限越高，并行环境越多，训练时间越短")
        self.slider_time = QLabel()
        self.slider_time.setStyleSheet("color:#8b949e;")
        self.slider_time.setWordWrap(True)
        self.quality_slider.valueChanged.connect(self._on_bulk_sliders)
        self.vram_slider.valueChanged.connect(self._on_bulk_sliders)

        base = pretrained_checkpoint()
        meta = pretrained_meta()
        where = (f"第 {meta['iteration']} 轮" if meta.get("iteration")
                 else "已训练权重")
        self._pretrain_meta = meta
        self._pretrain_where = where
        self.pretrain_label = QLabel(
            f"{meta.get('name', 'dance_base.pt')} · {where}"
            if base else "缺失")
        self.pretrain_label.setStyleSheet(
            "color:#8b949e;" if base else "color:#e6b450;")
        self.pretrain_label.setToolTip(
            "基础 PPO 权重已固定；训练只在此基础上微调当前舞蹈。")

        sm_form = QFormLayout()
        sm_form.addRow("预训练权重", self.pretrain_label)
        sm_form.addRow("训练精细度", self.quality_slider)
        sm_form.addRow("显存占用上限", self.vram_slider)
        sm_form.addRow("", self.slider_time)
        self.sm_envs.setToolTip("并行环境数，也是 PPO 的采样批量来源")
        sm_form.addRow("并行环境", self.sm_envs)
        sm_form.addRow("采样批量", self.ppo_batch)
        sm_form.addRow("迭代轮次", self.sm_iters)
        self.curriculum_box = QCheckBox("强化转身跳跃")
        self.curriculum_box.setChecked(bool(getattr(params, "curriculum", False)))
        self.curriculum_box.setToolTip("只加重转身、高度和身体姿态奖励，不改变训练方法")
        sm_form.addRow("", self.curriculum_box)
        self.rewards_btn = QPushButton("奖励权重…")
        self.rewards_btn.clicked.connect(self._open_rewards)
        sm_form.addRow("", self.rewards_btn)
        self._reward_weights = getattr(params, "reward_weights", {}) or {}
        self._reward_profile = getattr(params, "reward_profile", "")
        self._show_versions = getattr(params, "show_weight_versions",
                                      show_weight_versions())
        if self._reward_profile:
            self.rewards_btn.setText(f"奖励权重 · {self._reward_profile}")
        self.sm_box = QGroupBox("PPO 微调")
        self.sm_box.setLayout(sm_form)

        form = QFormLayout()
        self.precision = QDoubleSpinBox()
        self.precision.setRange(1.0, 30.0)
        self.precision.setSingleStep(0.5)
        self.precision.setDecimals(1)
        self.precision.setSuffix(" °")
        self.precision.setValue(params.precision_deg)
        self.precision.setToolTip("允许的关节跟踪误差；越大越容易、越粗糙")
        form.addRow("动作精度", self.precision)

        self.population = QSpinBox()
        self.population.setRange(2, 64)
        self.population.setValue(params.population)
        self.population.setToolTip("每个训练轮次评估的候选策略数量")
        form.addRow("候选数量", self.population)

        self.iterations = QSpinBox()
        self.iterations.setRange(2, 500)
        self.iterations.setValue(params.iterations)
        form.addRow("训练轮数", self.iterations)

        self.hand_safe = QSpinBox()
        self.hand_safe.setRange(5, 150)
        self.hand_safe.setValue(int(round(params.hand_safe_m * 1000)))
        self.hand_safe.setSuffix(" mm")
        form.addRow("双手距离", self.hand_safe)

        self.height_term = QSpinBox()
        self.height_term.setRange(10, 300)
        self.height_term.setValue(int(round(params.height_term_m * 1000)))
        self.height_term.setSuffix(" mm")
        form.addRow("高度终止", self.height_term)

        self.tilt_term = QSpinBox()
        self.tilt_term.setRange(5, 90)
        self.tilt_term.setValue(int(round(params.tilt_term_deg)))
        self.tilt_term.setSuffix(" °")
        form.addRow("倾角终止", self.tilt_term)

        self.substeps = QSpinBox()
        self.substeps.setRange(2, 40)
        self.substeps.setValue(params.substeps)
        self.substeps.setToolTip("每采样帧的物理子步，越高越精细也越慢")
        form.addRow("物理子步", self.substeps)

        self.render_box = QCheckBox("实时画面")
        self.render_box.setChecked(render)
        self.render_box.setToolTip(
            "GPU 训练在应用内离屏渲染，约 +0.2 秒/轮；Smoke 渲染物理画面。")
        form.addRow("", self.render_box)

        self.trial_box = QGroupBox("Smoke")
        self.trial_box.setLayout(form)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#c9d1d9;")
        self.recommend = QLabel()
        self.recommend.setWordWrap(True)
        self.recommend.setStyleSheet("color:#e6b450;")
        self.rec_pop = 8
        self.rec_iter = 24
        self.rec_sm_iter = 5000

        self.calc_btn = QPushButton("计算预估")
        self.calc_btn.clicked.connect(self._refresh_async)
        self.apply_rec_btn = QPushButton("应用推荐")
        self.apply_rec_btn.clicked.connect(self._apply_recommended)
        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self._reset_defaults)
        ok_btn = QPushButton("确定")
        ok_btn.setObjectName("accentBtn")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        self.hint = QLabel("点「计算预估」实测本机速度并检测 GPU 后端")
        self.hint.setStyleSheet("color:#e6b450;")
        btns = QHBoxLayout()
        btns.addWidget(self.calc_btn)
        btns.addWidget(self.apply_rec_btn)
        btns.addWidget(reset_btn)
        btns.addStretch(1)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(method_box)
        lay.addWidget(self.sm_box)
        lay.addWidget(self.trial_box)
        lay.addWidget(self.info, stretch=1)
        lay.addWidget(self.recommend)
        lay.addWidget(self.hint)
        lay.addLayout(btns)

        # Editing must stay instant: no benchmark / nvidia-smi on every
        # digit. The estimates refresh only via 计算预估 (or once on open).
        for w in (self.precision, self.population, self.iterations,
                  self.hand_safe, self.height_term, self.tilt_term,
                  self.substeps, self.sm_envs, self.sm_iters):
            w.valueChanged.connect(self._mark_stale)
        self.render_box.toggled.connect(self._on_render)
        self._update_ppo_batch()
        self._on_method_changed()
        self._on_bulk_sliders()
        self._refresh(measure=False)
        QTimer.singleShot(80, self._refresh_async)

    # ------------------------------------------------------------ actions ----
    def _on_bulk_sliders(self) -> None:
        """Bulk dial -> numbers + a rough time estimate (no I/O)."""
        q = self.quality_slider.value() / 100.0
        v = self.vram_slider.value() / 100.0
        # VRAM: 8.2 GB total, ~0.9 GB for the process/context, ~1.9 MB/env.
        envs = int(round(((8.2 * v - 0.9) / 0.0019) / 128.0)) * 128
        envs = max(256, min(4096, envs))
        # The quality dial is a total-work budget at 2048 envs. More VRAM
        # means more parallel envs, so the same work needs fewer iterations
        # and the wall-clock time goes DOWN as the VRAM slider goes right.
        low, high = 20.0, 600.0
        base_iters = int(round((low + (high - low) * q) / 10.0)) * 10
        base_iters = max(int(low), min(int(high), base_iters))
        ref_envs = 2048.0
        iters = int(round(base_iters * ref_envs / max(envs, 1) / 10.0)) * 10
        iters = max(20, min(4000, iters))
        pop = int(round(4 + 12 * q))
        es_iters = int(round(8 + 112 * q))
        substeps = int(round(4 + 16 * q))
        for widget, value in ((self.sm_envs, envs), (self.sm_iters, iters),
                              (self.population, pop), (self.iterations, es_iters),
                              (self.substeps, substeps)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        per_iter = 0.9 + 0.0005 * envs

        def secs(iter_count: float, env_count: float) -> float:
            return iter_count * (0.9 + 0.0005 * env_count)

        cur = secs(iters, envs)
        # Quality ends at the current VRAM.
        lo_q = secs(max(20, int(round(low * ref_envs / envs / 10.0)) * 10), envs)
        hi_q = secs(max(20, int(round(high * ref_envs / envs / 10.0)) * 10), envs)
        # VRAM ends at the current quality; higher VRAM is faster.
        v_lo, v_hi = 256, 3328
        lo_v = secs(base_iters * ref_envs / v_lo, v_lo)
        hi_v = secs(base_iters * ref_envs / v_hi, v_hi)

        def fmt(sec):
            if sec < 90:
                return f"{sec:.0f} 秒"
            if sec < 3600:
                return f"{sec / 60:.0f} 分钟"
            return f"{sec / 3600:.1f} 小时"

        self.slider_time.setText(
            f"质量 最左 ≈ {fmt(lo_q)} · 最右 ≈ {fmt(hi_q)} · 当前 ≈ {fmt(cur)}"
            f"\n显存 低 ≈ {fmt(lo_v)} · 高 ≈ {fmt(hi_v)}"
            f" · {envs} 环境 / {iters} 轮")
        self._mark_stale()

    def _refresh_async(self) -> None:
        """Default / button-triggered measure, off the GUI thread."""
        if self._estimate_worker is not None and self._estimate_worker.isRunning():
            return
        self.calc_btn.setEnabled(False)
        self.calc_btn.setText("计算中…")
        self.hint.setText("正在检测显卡并实测物理速度…")
        self._estimate_worker = _EstimateWorker(self.substeps.value(), self)
        self._estimate_worker.done.connect(self._on_estimate_done)
        self._estimate_worker.finished.connect(self._on_estimate_finished)
        self._estimate_worker.start()

    def _on_estimate_done(self, data) -> None:
        self._measured = data
        self._refresh(measure=False)
        self.hint.setText("预估已更新")

    def _on_estimate_finished(self) -> None:
        self.calc_btn.setEnabled(True)
        self.calc_btn.setText("计算预估")

    def done(self, result: int) -> None:
        # Never destroy the dialog while nvidia-smi / the benchmark is still
        # running in its worker thread.
        worker = self._estimate_worker
        if worker is not None and worker.isRunning():
            worker.wait(8000)
        self._estimate_worker = None
        super().done(result)

    def _open_rewards(self) -> None:
        dlg = RewardDialog(self._reward_weights, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._reward_weights = dlg.weights()
        self._reward_profile = dlg.profile_name()
        self._show_versions = dlg.show_versions()
        self.rewards_btn.setText(
            f"奖励权重 · {self._reward_profile}" if self._reward_profile
            else "奖励权重")
        self.hint.setText("奖励权重已更新 · 点击「计算预估」刷新预估")

    def _on_method_changed(self) -> None:
        method = self.method.currentData()
        # Only the selected method's knobs are shown; the other group returns
        # when its method is chosen.
        if hasattr(self, "sm_box"):
            self.sm_box.setVisible(method == "jumper_ppo")
        if hasattr(self, "trial_box"):
            self.trial_box.setVisible(method == "es")
        if hasattr(self, "info"):
            self._refresh(measure=False)

    def _update_ppo_batch(self) -> None:
        """Show PPO's real batch: num_envs x 24, split into 4 mini-batches."""
        n = max(self.sm_envs.value(), 1)
        self.ppo_batch.setText(f"{n * 24}")

    def _mark_stale(self) -> None:
        self.hint.setText("参数已修改 · 点击「计算预估」刷新预估与推荐")

    def _on_render(self, on: bool) -> None:
        self._render = on
        self._mark_stale()

    def _reset_defaults(self) -> None:
        d = TrainParams()
        self.method.setCurrentIndex(self.method.findData("jumper_ppo"))
        self.quality_slider.setValue(48)
        self.vram_slider.setValue(58)
        self.curriculum_box.setChecked(d.curriculum)
        self.sm_envs.setValue(d.sm_num_envs)
        self.sm_iters.setValue(d.sm_iterations)
        self.precision.setValue(d.precision_deg)
        self.population.setValue(d.population)
        self.iterations.setValue(d.iterations)
        self.hand_safe.setValue(int(round(d.hand_safe_m * 1000)))
        self.height_term.setValue(int(round(d.height_term_m * 1000)))
        self.tilt_term.setValue(int(round(d.tilt_term_deg)))
        self.substeps.setValue(d.substeps)

    def _apply_recommended(self) -> None:
        self.population.setValue(self.rec_pop)
        self.iterations.setValue(self.rec_iter)
        self.sm_iters.setValue(self.rec_sm_iter)
        self._refresh(measure=False)

    # ------------------------------------------------------------ estimate ----
    def _refresh(self, measure: bool = True) -> None:
        global _GPU_CACHE, _BACKEND_CACHE
        self.hint.setText("" if measure else
                          "点「计算预估」实测本机速度并检测 GPU 训练环境")
        vm = psutil.virtual_memory()
        gb = 1024 ** 3
        cores = os.cpu_count() or 8
        if self._measured is not None:
            gpu, backend_desc, per_frame = self._measured
            if per_frame is None:
                per_frame = cached_frame_seconds(self.substeps.value()) or 5.0e-4
        else:
            # Opening the dialog must be instant and quiet: use what the
            # last run cached; the single default measure fills the rest in
            # a worker after the window is visible.
            gpu = _GPU_CACHE or "检测中…"
            backend_desc = (_BACKEND_CACHE[1] if _BACKEND_CACHE is not None
                            else "检测中…")
            per_frame = cached_frame_seconds(self.substeps.value()) or 5.0e-4
        note = ("" if self._measured is not None
                else "\n速度为默认估算 · 点「计算预估」实测")
        eff = max(3, math.ceil(self._frame_count / 300))
        frames = int(math.ceil(self._frame_count / eff))
        pop = self.population.value()
        iters = self.iterations.value()
        train_s = frames * per_frame * pop * iters * 1.15
        render_s = 0.0
        if self._render:
            render_s = (iters / 2) * (frames * (2 / 3) * per_frame + 0.08)
        total_s = train_s + render_s

        # ---- GPU training (Jumper/mjlab, rsl_rl PPO): real measured speed,
        # never a made-up formula when this machine has already trained.
        envs = max(self.sm_envs.value(), 1)
        sm_iters = self.sm_iters.value()
        speed = cached_iter_seconds()
        if speed:
            measured = float(speed["seconds"])
            m_envs = int(speed.get("envs") or envs)
            if m_envs == envs:
                per_iter = measured
                speed_basis = f"约 {measured:.2f} 秒/轮 · num_envs={m_envs}"
            else:
                per_iter = measured * (0.75 + 0.25 * envs / max(m_envs, 1))
                speed_basis = (
                    f"约 {measured:.2f} 秒/轮 · 按环境数换算到 {envs}")
        else:
            per_iter = 0.9 + 0.0005 * envs
            speed_basis = f"约 {per_iter:.2f} 秒/轮 · 按环境数粗估"
        if self._render:
            per_iter += 0.2
            speed_basis += " · 含应用内实时画面约 +0.2 秒/轮"
        full_s = sm_iters * per_iter

        # Memory: model/context + reference arrays + renderer scratch (local
        # trial). The GPU run's VRAM is measured separately.
        ref_mb = self._frame_count * 29 * 8 * 3 / (1024 ** 2)
        est_mb = 140.0 + ref_mb
        remaining_gb = max(0.0, (vm.available - est_mb * 1024 ** 2) / gb)

        # Recommendations: ~BUDGET seconds at the measured speed; wider
        # populations use the 32-core machine but the ES itself is serial.
        pop_rec = max(8, min(24, int(round(2.2 * math.sqrt(cores)))))
        it_rec = int(BUDGET_SECONDS / max(
            frames * per_frame * pop_rec * 1.15, 1e-6))
        self.rec_pop = pop_rec
        self.rec_iter = max(8, min(120, it_rec))
        self.rec_sm_iter = max(20, int(round(300 * 2048 / max(envs, 1)
                                             / 10.0)) * 10)
        vram_gb = min(8.2, 0.9 + 0.0022 * envs)
        if full_s < 90:
            time_text = f"{full_s:.0f} 秒"
        elif full_s < 3600:
            time_text = f"{full_s / 60:.0f} 分钟"
        else:
            time_text = f"{full_s / 3600:.1f} 小时"

        lines = [
            f"系统内存：总 {vm.total / gb:.1f} GB · 已用 {vm.used / gb:.1f} GB"
            f" · 可用 {vm.available / gb:.1f} GB",
        ]
        if self.method.currentData() == "es":
            lines += [
                "",
                "【Smoke】CPU 物理 + ES",
                f"  设备：CPU {cores} 核 · 采样帧 {frames} · "
                f"物理 {per_frame * 1000:.2f} ms/帧",
                f"  预估时间 ≈ {total_s:.1f} 秒"
                + (f"，其中渲染 ≈ {render_s:.1f} 秒" if self._render else "")
                + f" · 预估占用 ≈ {est_mb:.0f} MB · "
                f"剩余内存 ≈ {remaining_gb:.1f} GB",
            ]
        else:
            lines += [
                "",
                "【GPU 微调】Jumper/mjlab · rsl_rl PPO",
                f"  显卡：{gpu}",
                f"  num_envs={envs} · 批量 {envs * 24} · "
                f"新增 {sm_iters} 轮",
                f"  速度：{speed_basis}",
                f"  预计时间 ≈ {time_text}"
                f" · {sm_iters} 轮 × {per_iter:.2f} 秒/轮",
                f"  显存 ≈ {vram_gb:.1f} GB",
            ]
        self.info.setText("\n".join(lines) + note)
        if self.method.currentData() == "es":
            self.recommend.setText(
                f"Smoke 推荐：训练数量 {self.rec_pop} · 训练轮数 "
                f"{self.rec_iter} · {BUDGET_SECONDS:.0f} 秒预算")
        else:
            self.recommend.setText(
                f"GPU 微调推荐：{self.rec_sm_iter} 轮 ≈ "
                f"{self.rec_sm_iter * per_iter / 60:.0f} 分钟"
                f" · 从预训练权重继续，可随时停止")

    # ------------------------------------------------------------- result ----
    def params(self, base: TrainParams) -> TrainParams:
        return TrainParams(
            method=self.method.currentData() or "jumper_ppo",
            sm_num_envs=self.sm_envs.value(),
            # rsl_rl's PPO has no batch_size setting; kept on the dataclass
            # only as the inherited SuperMimic default.
            sm_batch_size=getattr(base, "sm_batch_size", 256),
            sm_iterations=self.sm_iters.value(),
            iterations=self.iterations.value(),
            population=self.population.value(),
            elites=min(base.elites, max(1, self.population.value() - 1)),
            sigma=base.sigma,
            precision_deg=self.precision.value(),
            stride=base.stride,
            max_frames=base.max_frames,
            render=self.render_box.isChecked(),
            seed=base.seed,
            hand_safe_m=self.hand_safe.value() / 1000.0,
            height_term_m=self.height_term.value() / 1000.0,
            tilt_term_deg=float(self.tilt_term.value()),
            substeps=self.substeps.value(),
            device="auto",
            reward_weights=self._reward_weights,
            reward_profile=self._reward_profile,
            show_weight_versions=self._show_versions,
            curriculum=self.curriculum_box.isChecked(),
        )
