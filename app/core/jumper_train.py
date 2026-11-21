"""Run Jumper's GPU reference-tracking (rsl_rl PPO) training, in-app.

The app's own trainer (`train_worker.py`) is a CPU evolution strategy over PD
gains -- a faithful but small-scale trial. This module drives the real thing:
the Jumper repository's mjlab/PyTorch stack (Windows + warp:cuda), whose dance
tasks are already BeyondMimic-style reference-guided tracking on this exact
22-DoF crab.

What happens when the app starts a GPU run:

1. `prepare_dance_task` exports the recognised dance with
   `app.core.jumper_clip.build_dance_clip` into
   `<workspace>/tasks/jumper/dance_crab/media/crab.npz`, and writes that task's
   config files with the clip's *measured* statistics (joint-position and
   velocity scales for the reward terms, root-height termination band).
2. `scripts/train.py --task jumper.dance_crab --backend warp --device cuda:0`
   is launched as a child process; its output is parsed for the iteration
   count and the metrics shown on the curves.
3. When training ends, `scripts/play.py` records the trained policy dancing the
   whole clip under physics to an MP4, which the training page plays beside the
   source video.

The workspace is `D:\\jumper_ws` by default; set `CRAB_JUMPER_WS` to move it.
Everything is a subprocess, so the app itself never imports torch and a failed
run cannot take the GUI down with it.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.constants import ASSETS_DIR, CACHE_DIR, JUMP_HOP_GAIN
from app.core.crab_model import CrabModel
from app.core.jumper_clip import _JUMPER_LEGS, build_dance_clip

TASK_ID = "jumper.dance_crab"
DEFAULT_WORKSPACE = Path(r"D:\jumper_ws")
SPEED_PATH = CACHE_DIR / "jumper" / "speed.json"
#: The slow, offline-trained PPO base. Every app run fine-tunes the current
#: dance from this checkpoint instead of learning from scratch, which is what
#: keeps the in-app step short.
PRETRAIN_PATH = ASSETS_DIR / "pretrained" / "dance_base.pt"
PRETRAIN_META = PRETRAIN_PATH.with_suffix(".json")

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_ITER_TIME = re.compile(r"Iteration time:\s+([0-9.]+)\s*s")
_LOG_DIR = re.compile(r"\[mjrl\] log directory\s+(.+?)\s*$")
_RESUME = re.compile(r"resumed at iteration\s+(\d+); running\s+(\d+)\s+more")
_KV = re.compile(r"^\s*(.+?):\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


def cached_iter_seconds() -> dict | None:
    """The last run's measured PPO iteration time, or None.

    Written by `JumperTrainWorker` from the "Iteration time" line, so the
    parameters dialog can estimate with this machine's real speed instead of a
    guess made up from a two-iteration smoke test.
    """
    try:
        data = json.loads(SPEED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        seconds = float(data.get("seconds"))
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    data["seconds"] = seconds
    return data


def workspace_path() -> Path:
    return Path(os.environ.get("CRAB_JUMPER_WS", str(DEFAULT_WORKSPACE)))


def venv_python(workspace: Path) -> Path:
    exe = "python.exe" if os.name == "nt" else "python"
    return workspace / ".venv" / ("Scripts" if os.name == "nt" else "bin") / exe


def jumper_status(workspace: Path | None = None) -> tuple[bool, str]:
    """(usable, reason) for the external GPU training stack."""
    ws = Path(workspace or workspace_path())
    py = venv_python(ws)
    if not ws.is_dir():
        return False, f"未找到 Jumper 训练环境：{ws}"
    if not py.is_file():
        return False, f"未找到训练用 Python：{py}"
    if not (ws / "scripts" / "train.py").is_file():
        return False, f"不是 Jumper 仓库 · 缺 scripts/train.py · {ws}"
    return True, f"{ws} · warp/cuda"


def pretrained_checkpoint() -> Path | None:
    """The fixed PPO base every app dance run starts from."""
    return PRETRAIN_PATH if PRETRAIN_PATH.is_file() else None


def pretrained_meta() -> dict:
    """Name/source/iteration of the bundled base, for labels and manifests."""
    data = {"name": PRETRAIN_PATH.name, "iteration": None, "source": ""}
    try:
        loaded = json.loads(PRETRAIN_META.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update(loaded)
    except (OSError, ValueError):
        pass
    return data


# ------------------------------------------------------------------ stats ----
def clip_stats(clip_path: Path) -> dict:
    """Measured statistics of an exported clip, for the task's reward scales."""
    with np.load(clip_path) as data:
        dt = float(data["dt"])
        cols, vs = [], []
        for leg, k in _JUMPER_LEGS:
            for j in range(k):
                q = np.asarray(data[f"meas_L{_leg_index(leg)}_j{j}"],
                               dtype=np.float64)
                cols.append(q)
                vs.append(np.diff(q) / dt)
        z = np.asarray(data["meas_body_z"], dtype=np.float64)
        tilt = np.degrees(np.max(np.abs(np.stack(
            [data["meas_body_roll"], data["meas_body_pitch"]], axis=1))))
    q = np.stack(cols, axis=1)
    v = np.stack(vs, axis=1)
    sd = q.std(axis=0)
    rms_pos = float(np.sqrt(np.mean(sd ** 2)))
    rms_vel = float(np.sqrt(np.mean(v.std(axis=0) ** 2)))
    return {
        "frames": int(len(q)),
        "dt": dt,
        "duration_s": float(len(q) * dt),
        "pos_std_median": float(np.median(sd)),
        "pos_std_max": float(sd.max()),
        "rms_pos": rms_pos,
        "vel_std_median": float(np.median(v.std(axis=0))),
        "vel_std_max": float(v.std(axis=0).max()),
        "rms_vel": rms_vel,
        "peak_speed": float(np.abs(v).max()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "tilt_max_deg": float(tilt),
    }


def _leg_index(leg: str) -> int:
    return ("LF", "LM", "LR", "RR", "RM", "RF").index(leg)


# ------------------------------------------------------- task preparation ----
_INIT_PY = '''"""jumper.dance_crab -- the crab-dance app's own choreography, by imitation.

Generated by the crab-dance app (`app/core/jumper_train.py`); the clip is
`media/crab.npz`, exported from a Sports2D pose estimate retargeted onto this
robot. Regenerated on every app training run, so edits here are overwritten.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="{task_id}",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating the crab-dance app's Sports2D dance; "
                "the clip is committed in tasks/jumper/dance_crab/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
'''

_RL_PY = '''"""Hyper-parameters for {task_id}, from jumper.dance.

The network, PPO and runner settings were chosen for imitating a dance on this
robot; regenerated by the crab-dance app, so edit the app template instead.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="{task_id}",
        # A choreography has no left-right symmetry; mirroring its samples
        # would train the average of the dance and its mirror image.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
'''

_ENV_PY = '''"""Environment config for {task_id}: the app's dance, `media/crab.npz`.

Generated by the crab-dance app from this clip's measurements, following the
rules written in `tasks/jumper/dance/env_cfg.py`. Measured at the source rate:
per-joint position std {pos_med:.3f} rad at the median and {pos_max:.3f} at the
most active, RMS amplitude {rms_pos:.3f} rad; joint velocity RMS {rms_vel:.3f}
rad/s, peaking at {peak:.1f} rad/s; the base between {zmin:.1f} and {zmax:.1f}
mm; tilt at most {tilt:.1f} degrees. Reward weights are starting points.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp

from ..common.dance.env import dance_env_cfg

MEDIA = Path(__file__).resolve().parent / "media"

EPISODE_S = 10.0


#: Reward weights written into the generated task (editable in the app).
REWARD_WEIGHTS = {reward_weights}


#: New reward terms the app may add on top of the task's own set. The first
#: group exists in mjlab already; keeping them here means "add reward" in the
#: dialog writes a term the backend really evaluates.
EXTRA_REWARDS = {{
    "joint_vel_l2": lambda w: RewardTermCfg(
        func=mdp.joint_vel_l2, weight=w,
        params={{"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}}),
    "joint_acc_l2": lambda w: RewardTermCfg(
        func=mdp.joint_acc_l2, weight=w,
        params={{"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}}),
    "joint_torques_l2": lambda w: RewardTermCfg(
        func=mdp.joint_torques_l2, weight=w,
        params={{"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))}}),
    "flat_orientation_l2": lambda w: RewardTermCfg(
        func=mdp.flat_orientation_l2, weight=w,
        params={{"asset_cfg": SceneEntityCfg("robot")}}),
    "is_alive": lambda w: RewardTermCfg(func=mdp.is_alive, weight=w),
    "is_terminated": lambda w: RewardTermCfg(func=mdp.is_terminated, weight=w),
}}


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        rsi_pose_range={{
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        }},
        rsi_velocity_range={{
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        }},
        rsi_joint_range=(-0.05, 0.05),
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # 0.44 x the clip's RMS amplitude; standing still scores ~0.006.
        joint_pos_std={pos_std:.2f},
        joint_pos_weight=1.0,
        # Two thirds of the clip's velocity RMS; standing still ~0.10.
        joint_vel_std={vel_std:.2f},
        joint_vel_weight=0.5,
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        power_weight=-0.02,
        torque_headroom_weight=-2.0,
        joint_limit_weight=-10.0,
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # Relative to the reference; sized above the clip's own vertical band
        # ({band_mm:.0f} mm), so a missed hop does not terminate every jump.
        anchor_height_error={height_term:.2f},
        anchor_tilt_error=0.3,
        # This clip lifts support feet: a hop raises the whole body
        # ({hop_mm:.0f} mm) and leg-raise gestures tuck one foot
        # ({lift_mm:.0f} mm). The stock 50 mm fires during every jump before a
        # policy has learned to leave the ground, so it is sized above the
        # reference's own foot lift with a margin.
        support_foot_error={support_foot:.2f},
        push_interval_s=(3.0, 8.0),
        push_velocity_range={{
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        }},
    )
    for _name, _weight in REWARD_WEIGHTS.items():
        _w = float(_weight)
        if _name in cfg.rewards:
            cfg.rewards[_name].weight = _w
        elif _name in EXTRA_REWARDS:
            cfg.rewards[_name] = EXTRA_REWARDS[_name](_w)
    return cfg
'''


DEFAULT_PPO_WEIGHTS = {
    "motion_global_root_pos": 0.5, "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0, "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0, "motion_body_ang_vel": 1.0,
    "motion_joint_pos": 1.0, "motion_joint_vel": 0.5,
    "action_rate_l2": -0.1, "action_acc_l2": -0.05,
    "power": -0.02, "torque_headroom": -2.0,
    "joint_limit": -10.0, "self_collisions": -1.0,
    "anchor_height": 1.0,
}
#: Curriculum fine-tune for turns/jumps: the body root/twist terms get more
#: weight and the height termination is relaxed in prepare_dance_task. The
#: fixed base is still the starting point; this only shapes the extra rounds.
CURRICULUM_PPO_WEIGHTS = {
    "motion_global_root_pos": 3.0, "motion_global_root_ori": 8.0,
    "motion_body_pos": 1.5, "motion_body_ori": 3.0,
    "motion_body_lin_vel": 1.5, "motion_body_ang_vel": 3.0,
    "motion_joint_pos": 0.8, "motion_joint_vel": 0.5,
    "action_rate_l2": -0.05, "action_acc_l2": -0.02,
    "power": -0.02, "torque_headroom": -2.0,
    "joint_limit": -10.0, "self_collisions": -1.0,
    "anchor_height": 6.0,
}

#: Stage 2 after the body curriculum: bring the hands and the body together.
#: Root tracking stays high enough to keep turns/jumps, but the joint tracking
#: and the smoothness terms return to a normal dance balance.
INTEGRATION_PPO_WEIGHTS = {
    "motion_global_root_pos": 1.5, "motion_global_root_ori": 2.0,
    "motion_body_pos": 1.0, "motion_body_ori": 1.5,
    "motion_body_lin_vel": 1.0, "motion_body_ang_vel": 1.5,
    "motion_joint_pos": 1.0, "motion_joint_vel": 0.5,
    "action_rate_l2": -0.05, "action_acc_l2": -0.02,
    "power": -0.02, "torque_headroom": -2.0,
    "joint_limit": -10.0, "self_collisions": -1.0,
    "anchor_height": 4.0,
}

#: Jump-focused stage: keep the turning/gait ability from the skill base and
#: push the vertical/root tracking until the reference hop is reproduced.
JUMP_PPO_WEIGHTS = {
    "motion_global_root_pos": 6.0, "motion_global_root_ori": 4.0,
    "motion_body_pos": 1.5, "motion_body_ori": 3.0,
    "motion_body_lin_vel": 1.5, "motion_body_ang_vel": 3.0,
    "motion_joint_pos": 1.0, "motion_joint_vel": 0.5,
    "action_rate_l2": -0.02, "action_acc_l2": -0.01,
    "power": -0.02, "torque_headroom": -2.0,
    "joint_limit": -10.0, "self_collisions": -1.0,
    "anchor_height": 10.0,
}

#: Extra terms the generated env defines; the dialog may add these for real.
SUPPORTED_EXTRA_REWARDS = (
    "joint_vel_l2", "joint_acc_l2", "joint_torques_l2",
    "flat_orientation_l2", "is_alive", "is_terminated", "anchor_height",
)


def prepare_dance_task(table, crab: CrabModel,
                       workspace: Path | None = None,
                       log=lambda _msg: None,
                       reward_weights: dict | None = None,
                       curriculum: bool = False) -> dict:
    """Export the clip and write the Jumper task; returns what it measured."""
    if curriculum:
        reward_weights = {**CURRICULUM_PPO_WEIGHTS,
                          **(reward_weights or {})}
    ws = Path(workspace or workspace_path())
    ok, why = jumper_status(ws)
    if not ok:
        raise RuntimeError(why)
    task = ws / "tasks" / "jumper" / "dance_crab"
    media = task / "media"
    media.mkdir(parents=True, exist_ok=True)
    # More than one clip in media/ is an error in the Jumper converter, not a
    # choice it makes for you.
    for old in media.glob("*.npz"):
        try:
            old.unlink()
        except OSError:
            pass
    out = media / "crab.npz"
    tmp = media / "crab.tmp.npz"
    log("导出参考舞蹈 → Jumper 片段…")
    build_dance_clip(table, crab, tmp)
    tmp.replace(out)

    stats = clip_stats(out)
    band = stats["z_max"] - stats["z_min"]
    # Above the reference's own vertical band: a missed hop must not
    # terminate every jump. The old 0.65*band was BELOW the band
    # (80 mm vs 126 mm on the dance clip), which trained the policy
    # to stay down.
    height_term = max(0.10, round(band + 0.04, 2))
    jump = getattr(table, "jump", None)
    hop_max = (float(np.max(np.asarray(jump, dtype=float)))
               * JUMP_HOP_GAIN * crab.z_stand
               if jump is not None and len(jump) else 0.0)
    lifts = getattr(table, "foot_lift", None) or {}
    lift_max = 0.0
    for arr in lifts.values():
        a = np.asarray(arr, dtype=float)
        if a.size:
            lift_max = max(lift_max, float(np.max(a)))
    # Leave a real margin above the clip's own foot lift: +20 mm made
    # the support-foot termination fire on a slightly late jump.
    support_foot = min(0.20, max(0.08,
                                 round(max(hop_max, lift_max) + 0.06, 2)))
    task.joinpath("__init__.py").write_text(
        _INIT_PY.format(task_id=TASK_ID), encoding="utf-8")
    task.joinpath("rl_cfg.py").write_text(
        _RL_PY.format(task_id=TASK_ID), encoding="utf-8")
    task.joinpath("env_cfg.py").write_text(_ENV_PY.format(
        task_id=TASK_ID,
        pos_med=stats["pos_std_median"], pos_max=stats["pos_std_max"],
        rms_pos=stats["rms_pos"], rms_vel=stats["rms_vel"],
        peak=stats["peak_speed"], zmin=stats["z_min"] * 1000,
        zmax=stats["z_max"] * 1000, tilt=stats["tilt_max_deg"],
        pos_std=round(0.44 * stats["rms_pos"], 2),
        vel_std=round(2.0 / 3.0 * stats["rms_vel"], 2),
        band_mm=band * 1000, height_term=height_term,
        hop_mm=hop_max * 1000, lift_mm=lift_max * 1000,
        support_foot=support_foot,
        reward_weights=repr({**DEFAULT_PPO_WEIGHTS,
                             **{k: float(v) for k, v in (reward_weights or {}).items()
                                if k in DEFAULT_PPO_WEIGHTS
                                or k in SUPPORTED_EXTRA_REWARDS}}),
    ), encoding="utf-8")
    _ensure_registered(ws)
    log(f"舞蹈已写入 {out} · {stats['duration_s']:.1f} s · "
        f"{stats['frames']} 帧 · 峰值 {stats['peak_speed']:.1f} rad/s")
    return {"clip": str(out), **stats}


def _ensure_registered(ws: Path) -> None:
    """Make `import tasks` register the generated task (one line, idempotent)."""
    init = ws / "tasks" / "__init__.py"
    try:
        text = init.read_text(encoding="utf-8")
    except OSError:
        return
    if "import dance_crab" in text:
        return
    line = "from .jumper import dance_crab  # noqa: F401,E402\n"
    marker = "from .jumper import dance_brazilian  # noqa: F401,E402\n"
    if marker in text:
        text = text.replace(marker, marker + line, 1)
    else:
        key = "# Adding a task means adding one line here; nothing under scripts/ changes.\n"
        text = text.replace(key, key + line, 1)
    if "import dance_crab" not in text:
        raise RuntimeError(
            f"无法在 {init} 中注册训练任务；请在 tasks/__init__.py 中加入：\n"
            f"    {line.strip()}")
    init.write_text(text, encoding="utf-8")


# ------------------------------------------------------------- subprocess ----
def _popen_kwargs() -> dict:
    """Hide the console of the external training process (app runs pythonw).

    CREATE_NO_WINDOW does not affect the Jumper viewer's own GLFW window; the
    stdout/stderr pipes above are what carry its console output into the app.
    """
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["TORCHDYNAMO_DISABLE"] = "1"     # torch.compile has no Triton here
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _line_text(raw: str) -> str:
    return _ANSI.sub("", raw).rstrip()


def _is_noise(line: str) -> bool:
    """True for Warp/scene/model dumps that would bury the training log."""
    s = line.strip()
    if not s:
        return True
    if s.startswith("Module "):
        return True
    if s.startswith(("warnings.warn", "self._add_entities", "timestep:",
                     "integrator:", "WARNING: Attach", "Kernel cache:",
                     "compiled = spec.compile()")):
        return True
    if re.match(r"^\(\d+\):", s):          # the printed network layers
        return True
    if s.startswith(("+", "|", "\\\\?\\")):
        return True                    # box tables, warp cache path
    if s.startswith(("Actor Model", "Critic Model", "Resolved observation",
                     "Could not find git repository")):
        return True
    noisy = ("Linear(in_features", "MLPModel(", "(obs_normalizer)",
             "(distribution):", "(mlp):", "(0): Linear", "(1): ELU",
             "Layer (", "FutureWarning", "UserWarning")
    if any(tok in s for tok in noisy):
        return True
    if s in (")", "}", "(", "[", "]", ")", ")]", ")]))"):
        return True
    return False


def newest_checkpoint(run_dir: Path) -> Path | None:
    """The newest `model_*.pt` in a run directory, or None."""
    pts = sorted(Path(run_dir).glob("model_*.pt"),
                 key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


def latest_run_dir(workspace: Path | None = None) -> Path | None:
    """The newest run directory that has a checkpoint, or None.

    Used by the "preview the last weights" button -- after the app restarts
    there is no in-session run to remember, but the checkpoints from the last
    (stopped) training are still on disk.
    """
    ws = Path(workspace or workspace_path())
    logs = ws / "logs" / "jumper" / TASK_ID
    if not logs.is_dir():
        return None
    runs = [p for p in logs.iterdir()
            if p.is_dir() and newest_checkpoint(p) is not None]
    if not runs:
        return None
    return max(runs, key=lambda p: newest_checkpoint(p).stat().st_mtime)


class JumperTrainWorker(QThread):
    """Prepares the task and runs one Jumper training, then records a replay."""

    log = Signal(str)
    progress = Signal(int, int)
    metrics = Signal(dict)
    status = Signal(str)
    trained = Signal(str, str)        # run dir, newest checkpoint
    preview_ready = Signal(str)       # path to the policy MP4
    states_ready = Signal(str)        # path to the re-viewable replay NPZ
    live_frame = Signal(str)          # newest training JPEG (in-app preview)
    live_preview_frame = Signal(str)  # second env: live policy preview
    failed = Signal(str)

    def __init__(self, table, params, render: bool = True,
                 record_preview: bool = True, checkpoint: Path | None = None,
                 workspace: Path | None = None, parent=None):
        super().__init__(parent)
        self.table = table
        self.params = params
        self.render = bool(render)
        self.record_preview = bool(record_preview)
        # None = the fixed bundled base; a stopped run passes its newest
        # checkpoint so Start continues that run instead of restarting.
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.workspace = Path(workspace or workspace_path())
        self._proc: subprocess.Popen | None = None
        self._stopping = False
        self._run_dir: Path | None = None
        self._iteration = 0
        self._total = 0
        # Iteration the fixed base checkpoint ended on; progress and the
        # on-screen iteration count are relative to it.
        self._base_iter = 0
        self._last: dict = {}
        self._iter_times: list[float] = []
        self._frame_stop: threading.Event | None = None
        self._preview_frame_stop: threading.Event | None = None

    # ------------------------------------------------------------ control ----
    def request_stop(self) -> None:
        self._stopping = True
        self._stop_frame_poll()
        self._stop_preview_poll()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _start_frame_poll(self, folder: Path) -> None:
        """Emit the newest training JPEG whenever the trainer writes one."""
        self._frame_stop = threading.Event()
        stop = self._frame_stop

        def poll() -> None:
            last: Path | None = None
            while not stop.wait(0.25):
                try:
                    frames = list(folder.glob("live_*.jpg"))
                    if not frames:
                        continue
                    newest = max(frames,
                                 key=lambda p: p.stat().st_mtime_ns)
                except OSError:
                    continue
                if newest != last:
                    last = newest
                    self.live_frame.emit(str(newest))

        threading.Thread(target=poll, name="jumper-live-frame",
                         daemon=True).start()

    def _stop_frame_poll(self) -> None:
        event, self._frame_stop = self._frame_stop, None
        if event is not None:
            event.set()

    def _start_preview_poll(self, folder: Path) -> None:
        """Emit the newest live-policy JPEG from the second environment."""
        self._preview_frame_stop = threading.Event()
        stop = self._preview_frame_stop

        def poll() -> None:
            last: Path | None = None
            while not stop.wait(0.25):
                try:
                    frames = list(folder.glob("live_*.jpg"))
                    if not frames:
                        continue
                    newest = max(frames, key=lambda p: p.stat().st_mtime_ns)
                except OSError:
                    continue
                if newest != last:
                    last = newest
                    self.live_preview_frame.emit(str(newest))

        threading.Thread(target=poll, name="jumper-live-preview",
                         daemon=True).start()

    def _stop_preview_poll(self) -> None:
        event, self._preview_frame_stop = self._preview_frame_stop, None
        if event is not None:
            event.set()

    # ---------------------------------------------------------------- run ----
    def run(self) -> None:
        try:
            ok, why = jumper_status(self.workspace)
            if not ok:
                self.failed.emit(why)
                return
            base = self.checkpoint or pretrained_checkpoint()
            if base is None or not Path(base).is_file():
                self.failed.emit(
                    f"缺少预训练权重：{base or PRETRAIN_PATH}\n"
                    "请把训练好的 dance_base.pt 放到该路径后重试。")
                return
            py = venv_python(self.workspace)
            crab = CrabModel()
            try:
                info = prepare_dance_task(
                    self.table, crab, self.workspace, log=self.log.emit,
                    reward_weights=(getattr(self.params, "reward_weights", {})
                                    or {}).get("ppo", {}),
                    curriculum=bool(getattr(self.params, "curriculum", False)))
            finally:
                crab.close()
            self._clip_duration = float(info["duration_s"])
            if self._stopping:
                self.log.emit("已取消")
                return

            n = int(getattr(self.params, "sm_num_envs", 512))
            iters = int(getattr(self.params, "sm_iterations", 30000))
            # The app's wrapper runs the stock train.py but renders one env
            # offscreen into a JPEG instead of opening Jumper's separate viewer
            # window; the page shows that JPEG in its own physics pane.
            live_script = Path(__file__).resolve().parent / "jumper_live_train.py"
            extra_env: dict = {}
            if self.render:
                frames = CACHE_DIR / "jumper" / "live"
                frames.mkdir(parents=True, exist_ok=True)
                for old in frames.glob("*"):
                    try:
                        old.unlink()
                    except OSError:
                        pass
                extra_env = {
                    "CRAB_JUMPER_FRAME_DIR": str(frames),
                    "CRAB_JUMPER_FRAME_FPS": "5",
                    "CRAB_JUMPER_FRAME_SIZE": "480x360",
                }
                self._start_frame_poll(frames)
            base = Path(base)
            meta = pretrained_meta()
            where = (f"第 {meta['iteration']} 轮"
                     if base == PRETRAIN_PATH and meta.get("iteration")
                     else base.name)
            cmd = [str(py), str(live_script), "--task", TASK_ID,
                   "--backend", "warp", "--device", "cuda:0",
                   "--num_envs", str(n), "--max-iterations", str(iters),
                   "--checkpoint", str(base),
                   "--no-tensorboard", "--headless"]

            self.log.emit(f"PPO · {n} 环境 · {iters} 轮")
            self.status.emit("PPO 训练中…")
            self._run_dir = self._launch(cmd, parse_training=True,
                                         extra_env=extra_env)

            code = self._proc.returncode if self._proc is not None else -1
            if self._stopping:
                self.log.emit("训练已停止 · 检查点已保留")
                self._emit_trained()
                return
            if code != 0:
                self.failed.emit(f"训练进程退出码 {code}，见上方日志")
                self._emit_trained()
                return
            done = max(self._total - self._base_iter, 1)
            self.progress.emit(done, done)
            self.log.emit("训练完成")
            ckpt = self._emit_trained()
            if ckpt and self.record_preview and not self._stopping:
                self._record_preview(py, ckpt)
        except Exception as exc:  # noqa: BLE001 - surfaced in the page log
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._stop_frame_poll()
            self._stop_preview_poll()
            if self._proc is not None:
                try:
                    if self._proc.stdout:
                        self._proc.stdout.close()
                except OSError:
                    pass
                self._proc = None

    def _write_manifest(self, run_dir: Path) -> None:
        """Record which reward profile/weights produced this run (history)."""
        try:
            meta = pretrained_meta()
            payload = {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reward_profile": getattr(self.params, "reward_profile", ""),
                "reward_weights": getattr(self.params, "reward_weights", {}),
                "num_envs": getattr(self.params, "sm_num_envs", None),
                "iterations": getattr(self.params, "sm_iterations", None),
                "pretrain": meta.get("name", ""),
                "pretrain_iteration": meta.get("iteration"),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            tmp = run_dir / "app_run.tmp"
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, run_dir / "app_run.json")
        except OSError:
            pass

    # ----------------------------------------------------------- training ----
    def _launch(self, cmd: list[str], parse_training: bool,
                extra_env: dict | None = None) -> Path | None:
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.workspace), env=_env(extra_env),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            **_popen_kwargs())
        if self._stopping:
            self._proc.terminate()
        run_dir = None
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            if self._stopping:
                break
            line = _line_text(raw)
            if not line:
                continue
            m = _LOG_DIR.search(line)
            if m:
                run_dir = self.workspace / m.group(1)
                self._run_dir = run_dir
                self._write_manifest(run_dir)
                self.log.emit(f"日志目录：{run_dir}")
                continue
            resumed = _RESUME.search(line)
            if resumed:
                self._base_iter = int(resumed.group(1))
                more = int(resumed.group(2))
                continue
            if parse_training and self._consume_training_line(line):
                continue
            if _is_noise(line):
                continue
            self.log.emit(line)
        self._proc.wait()
        return run_dir

    def _consume_training_line(self, line: str) -> bool:
        """Parse one training line; emits progress/metrics and swallows noise."""
        base = self._base_iter
        total = max(self._total - base, 1) if self._total else 0
        it = min(max(self._iteration - base, 0), total) if total else 0
        m = _ITER.search(line)
        if m:
            self._iteration = int(m.group(1))
            self._total = int(m.group(2))
            total = max(self._total - base, 1)
            it = min(max(self._iteration - base, 0), total)
            self.progress.emit(it, total)
            self._last = {}
            return True
        if "Iteration time:" in line:
            tm = _ITER_TIME.search(line)
            if tm:
                self._record_iter_speed(float(tm.group(1)))
            total = max(self._total - base, 1)
            it = min(max(self._iteration - base, 0), total)
            rec = {"iter": it, **self._last}
            self.metrics.emit(rec)
            # One line per iteration, because the raw block is mostly a
            # per-term dump the parser has already summarised. Iterations are
            # relative to the pretrain so the count matches the progress bar.
            self.log.emit(
                f"第 {min(it + 1, total)}/{total} 轮 · 平均回报 "
                f"{rec.get('mean_reward', 0.0):.2f} · 存活 "
                f"{rec.get('ep_len', 0.0):.1f} 步 · 关节误差 "
                f"{rec.get('joint_err', 0.0):.2f} rad · 身体误差 "
                f"{rec.get('body_err', 0.0):.3f} m")
            self._last = {}
            # The iteration that just completed, so the bar does not sit one
            # short of the end for the whole final iteration.
            done = min(it + 1, total)
            self.progress.emit(done, total)
            return True
        kv = _KV.match(line)
        if kv:
            key = kv.group(1).strip()
            try:
                value = float(kv.group(2))
            except ValueError:
                return False
            if key in _METRIC_KEYS:
                self._last[_METRIC_KEYS[key]] = value
                return True
            # Swallow the long per-term reward dump; it is summarised above.
            if key.startswith("Episode_Reward/") or key.startswith("Metrics/"):
                return True
        return False

    def _record_iter_speed(self, seconds: float) -> None:
        """Cache the measured iteration time so the params dialog can use it."""
        if not (0.0 < seconds < 600.0):
            return
        self._iter_times.append(seconds)
        del self._iter_times[:-40]
        try:
            envs = int(getattr(self.params, "sm_num_envs", 0))
        except (TypeError, ValueError):
            envs = 0
        payload = {
            "seconds": round(float(statistics.median(self._iter_times)), 3),
            "envs": envs,
            "iteration": self._iteration,
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            SPEED_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = SPEED_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, SPEED_PATH)
        except OSError:
            pass

    def _emit_trained(self) -> Path | None:
        run_dir = self._run_dir
        if run_dir is None:
            logs = self.workspace / "logs" / "jumper" / TASK_ID
            dirs = sorted((p for p in logs.glob("*") if p.is_dir()),
                          key=lambda p: p.stat().st_mtime) if logs.is_dir() else []
            run_dir = dirs[-1] if dirs else None
        if run_dir is None:
            return None
        ckpt = newest_checkpoint(run_dir)
        if ckpt is None:
            return None
        self.trained.emit(str(run_dir), str(ckpt))
        return ckpt

    # ------------------------------------------------------------ preview ----
    def _record_preview(self, py: Path, ckpt: Path,
                        seconds: float | None = None) -> None:
        """Record the policy replay; `seconds` asks for a quick short preview."""
        if self._stopping:
            return
        quick = seconds is not None
        duration = min(float(seconds), self._clip_duration) if quick \
            else self._clip_duration
        self.status.emit("录制策略回放…")
        self.log.emit(f"录制回放 {duration:.0f}s")
        steps = int(duration * 50) + 35
        out_dir = CACHE_DIR / "jumper"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "preview" if quick else "policy"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        video = out_dir / f"{tag}_{stamp}.mp4"
        states = out_dir / f"{tag}_{stamp}.npz"
        cmd = [str(py), "scripts/play.py", "--task", TASK_ID,
               "--backend", "warp", "--device", "cuda:0", "--headless",
               "--steps", str(steps), "--video", str(video),
               "--states-out", str(states),
               "--checkpoint", str(ckpt.parent)]
        if quick:
            cmd += ["--video-fps", "24", "--video-width", "480",
                    "--video-height", "360"]
        self._launch_play(cmd)
        if self._stopping:
            return
        if video.is_file() and video.stat().st_size > 0:
            self.log.emit(f"回放已保存：{video.name}")
            self.preview_ready.emit(str(video))
            if states.is_file() and states.stat().st_size > 0:
                self.states_ready.emit(str(states))
            self.status.emit(
                "快速预览已就绪" if quick
                else "训练完成 · 策略回放已就绪")
        else:
            self.log.emit("回放录制失败 · 权重仍可导出")
            self.status.emit(
                "快速预览录制失败" if quick
                else "训练完成 · 回放录制失败")

    def _launch_play(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.workspace), env=_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            **_popen_kwargs())
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            if self._stopping:
                break
            line = _line_text(raw)
            if not line:
                continue
            # play.py rewrites a progress line with \r; keep it to one log
            # line per five seconds rather than flooding the pane.
            now = time.monotonic()
            if "[play]" in line:
                continue
            if _is_noise(line):
                continue
            self.log.emit(line)
        self._proc.wait()


class JumperPreviewWorker(JumperTrainWorker):
    """Records the policy replay for an existing checkpoint (e.g. after a stop)."""

    def __init__(self, checkpoint_dir: str | Path, clip_duration: float,
                 seconds: float | None = None,
                 workspace: Path | None = None, parent=None):
        super().__init__(None, None, workspace=workspace,
                         record_preview=True, parent=parent)
        self.checkpoint_dir = Path(checkpoint_dir)
        self._clip_duration = float(clip_duration)
        self._seconds = seconds

    def run(self) -> None:
        try:
            ok, why = jumper_status(self.workspace)
            if not ok:
                self.failed.emit(why)
                return
            ckpt = newest_checkpoint(self.checkpoint_dir)
            if ckpt is None:
                self.failed.emit(f"没有检查点：{self.checkpoint_dir}")
                return
            self._record_preview(venv_python(self.workspace), ckpt,
                                 seconds=self._seconds)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._proc = None


#: Jumper's metric names -> the short keys the curves use.
_METRIC_KEYS = {
    "Mean reward": "mean_reward",
    "Mean episode length": "ep_len",
    "Metrics/motion/error_joint_pos": "joint_err",
    "Metrics/motion/error_body_pos": "body_err",
    "Metrics/motion/error_anchor_pos": "anchor_err",
    "Metrics/motion/error_joint_vel": "joint_vel_err",
    "Metrics/motion/sampling_entropy": "sampling_entropy",
}
