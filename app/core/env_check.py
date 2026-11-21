"""Training-environment inventory, install and uninstall helpers.

The app uses two environments: its own `.venv` here and the Jumper/mjlab
workspace (default `D:\\jumper_ws`) where PPO runs. This module checks both,
lists them as selectable components for the environment dialog, and builds
the PowerShell command lines used to install, repair or remove them.

The PowerShell scripts live at the project root:
    setup_windows.ps1     -Action install|diagnose -Components a,b
    uninstall_windows.ps1 -Components a,b
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.constants import ASSETS_DIR, CACHE_DIR, PROJECT_DIR
from app.core.jumper_train import pretrained_checkpoint, workspace_path

SETUP_SCRIPT = PROJECT_DIR / "setup_windows.ps1"
UNINSTALL_SCRIPT = PROJECT_DIR / "uninstall_windows.ps1"
APP_VENV = PROJECT_DIR / ".venv"
APP_PY = APP_VENV / "Scripts" / "python.exe"
JUMPER_XML = ASSETS_DIR / "jumper" / "jumper.xml"
SHORTCUT = Path.home() / "Desktop" / "Jumper Studio.lnk"


@dataclass
class Component:
    key: str
    title: str
    status: str          # ok | warn | missing | error
    detail: str
    installable: bool = True
    uninstallable: bool = False
    required: bool = True
    selected: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _run(py: Path, code: str, timeout: float = 25.0) -> tuple[bool, str]:
    try:
        out = subprocess.run(
            [str(py), "-c", code], capture_output=True, text=True,
            timeout=timeout, errors="replace",
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                           if os.name == "nt" else 0))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    text = (out.stdout or out.stderr or "").strip().splitlines()
    return out.returncode == 0, text[-1] if text else ""


def _gpu_text() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                           if os.name == "nt" else 0))
        line = (out.stdout or "").strip().splitlines()
        return line[0].strip() if line else "未检测到 NVIDIA 显卡"
    except (OSError, subprocess.SubprocessError):
        return "未检测到 NVIDIA 显卡"


def _cache_mb() -> float:
    try:
        total = 0
        for p in CACHE_DIR.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)
    except OSError:
        return 0.0


def diagnose(workspace: Path | None = None, deep: bool = True
             ) -> tuple[list[Component], dict]:
    """Inventory both environments. `deep` also imports torch/mjrl.

    Deep checks run subprocesses and can take a few seconds; call this from
    a worker thread. `deep=False` is for tests.
    """
    ws = Path(workspace or workspace_path())
    info = {"workspace": str(ws), "gpu": _gpu_text(), "python": sys.version.split()[0]}

    # ---- app runtime + robot assets
    app_parts = []
    app_ok = APP_PY.is_file()
    app_parts.append("应用虚拟环境" if app_ok else "缺少 .venv")
    try:
        import cv2, mujoco, PySide6, Sports2D  # noqa: F401
        app_parts.append("依赖可导入")
    except Exception as exc:  # noqa: BLE001
        app_ok = False
        app_parts.append(f"依赖异常 {type(exc).__name__}")
    assets_ok = JUMPER_XML.is_file()
    if not assets_ok:
        app_ok = False
    app_parts.append("机器人模型" if assets_ok else "缺少机器人模型")
    components = [Component(
        "app", "应用运行环境", "ok" if app_ok else "error",
        " · ".join(app_parts), installable=True, uninstallable=True,
        required=True, selected=not app_ok)]

    # ---- pretrain
    base = pretrained_checkpoint()
    components.append(Component(
        "pretrain", "预训练权重", "ok" if base else "missing",
        str(base) if base else "缺少 assets/pretrained/dance_base.pt",
        installable=False, uninstallable=False, required=True,
        selected=False))

    # ---- Jumper training workspace
    repo = (ws / "scripts" / "train.py").is_file()
    jpy = ws / ".venv" / "Scripts" / "python.exe"
    venv = jpy.is_file()
    torch_ok = mjrl_ok = False
    torch_text = "未检查"
    mjrl_text = "未检查"
    if venv and deep:
        torch_import, out = _run(
            jpy, "import torch; print(torch.cuda.is_available(), "
                 "torch.cuda.get_device_name(0) if torch.cuda.is_available() "
                 "else '')")
        torch_ok = bool(torch_import and out.lower().startswith("true"))
        torch_text = ("CUDA · " + out) if torch_ok else (
            "torch 已装但 CUDA 不可用" if torch_import else "torch 不可用")
        mjrl_ok, mjrl_text = _run(
            jpy, "import mjrl, mjlab, rsl_rl; print('ok')")
        mjrl_text = "mjrl/mjlab 可导入" if mjrl_ok else (mjrl_text or "导入失败")
    elif venv:
        torch_text = "未做深度检查"
    jumper_ok = repo and venv and (not deep or (torch_ok and mjrl_ok))
    detail = []
    detail.append("仓库" if repo else "缺少仓库")
    detail.append("训练虚拟环境" if venv else "缺少训练虚拟环境")
    detail.append(torch_text)
    if deep:
        detail.append(mjrl_text)
    components.append(Component(
        "jumper", "PPO 训练环境", "ok" if jumper_ok else "missing",
        " · ".join(detail), installable=True, uninstallable=True,
        required=True, selected=not jumper_ok))

    # ---- play.py state export patch
    play = ws / "scripts" / "play.py"
    patch_ok = False
    if play.is_file():
        try:
            patch_ok = "--states-out" in play.read_text(encoding="utf-8")
        except OSError:
            pass
    components.append(Component(
        "patch", "策略预览补丁", "ok" if patch_ok else "missing",
        "play.py 支持 --states-out" if patch_ok else "play.py 缺少 --states-out",
        installable=True, uninstallable=False, required=True,
        selected=not patch_ok))

    # ---- desktop shortcut
    components.append(Component(
        "shortcut", "桌面快捷方式", "ok" if SHORTCUT.is_file() else "missing",
        str(SHORTCUT), installable=True, uninstallable=True,
        required=False, selected=False))

    # ---- caches
    components.append(Component(
        "cache", "缓存", "ok" if CACHE_DIR.is_dir() else "missing",
        f"{_cache_mb():.0f} MB · {CACHE_DIR}", installable=False,
        uninstallable=True, required=False, selected=False))

    ready = all(c.ok for c in components if c.required)
    info["ready"] = ready
    return components, info


def _power_args(script: Path, action: str, components: list[str],
                reinstall: bool = False, from_app: bool = True) -> list[str]:
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script), "-Action", action,
            "-Components", ",".join(components),
            "-JumperDir", str(workspace_path())]
    if from_app:
        args.append("-FromApp")
    if reinstall:
        args.append("-Reinstall")
    return args


def install_command(components: list[str], reinstall: bool = False) -> list[str]:
    return _power_args(SETUP_SCRIPT, "install", components, reinstall)


def uninstall_command(components: list[str]) -> list[str]:
    return _power_args(UNINSTALL_SCRIPT, "uninstall", components)


def diagnose_command() -> list[str]:
    return _power_args(SETUP_SCRIPT, "diagnose", [])
