# Windows 环境部署：Jumper Studio + Jumper/mjlab 训练栈

应用里的 PPO 训练不是 pip 里能装到的一个包，而是由两套彼此独立的环境组成：

| 环境 | 位置（默认） | 作用 | 是否必需 |
|---|---|---|---|
| **A. Jumper Studio** | `<项目目录>\.venv` | 视频识别（Sports2D）、PySide6 界面、角色预览、导出片段、调度训练 | 必需 |
| **B. SuperMimic 训练栈（Jumper/mjlab）** | `D:\jumper_ws` + 其 `.venv` | 真实 GPU 物理（MuJoCo Warp）、PPO/rsl_rl 全身跟踪训练、策略回放录制 | GPU 训练必需；只做本机 CPU 试训时可不装 |
| C. allmimic（可选研究栈） | `D:\supermimic_win` | 社区版 SuperMimic（G1 双足人形），**不识别螃蟹** | 不需要；见文末 |

两套环境互相隔离：应用本身从不 import torch，训练时只是用子进程调用 B 环境的
`scripts/train.py` / `scripts/play.py`。

---

## 一键脚本

在项目根目录双击 `setup_windows.bat`，或在 PowerShell 里运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

脚本会按顺序：

1. 检查 Windows / Python 3.11–3.13 / NVIDIA 驱动 / C++ 编译器提示；
2. 建应用 `.venv`，装 `requirements.txt`，按显卡架构装 CUDA 版 torch，缺资源时用镜像下载机器人模型，跑 `run.py --selftest`，创建桌面快捷方式；
3. 克隆 Jumper 到 `D:\jumper_ws`（直连失败自动走 `ghfast.top` 镜像），建 `.venv`，装 CUDA torch，`pip install -e .` 安装 mjlab/rsl_rl 等依赖，并用 `scripts/train.py --list` 验证任务注册；
4. 退出前打印所有路径与下一步。

常用参数：

```powershell
.\setup_windows.ps1 -JumperDir E:\jumper_ws      # 换训练环境位置
.\setup_windows.ps1 -Python "C:\Python311\python.exe"  # 指定解释器
.\setup_windows.ps1 -TorchBuild cu126             # auto / cu126 / cu128 / cpu
.\setup_windows.ps1 -SkipJumper                   # 只配置本软件
.\setup_windows.ps1 -SkipApp                      # 只配置训练环境
.\setup_windows.ps1 -NoShortcut -SkipTorch        # 不建快捷方式 / 不装 torch
.\setup_windows.ps1 -NoMirror                     # GitHub 可直连时
```

脚本是幂等的：已有的 venv、依赖、机器人资源会跳过；**训练进行中脚本会拒绝改环境**
（会列出进程 PID，除非加 `-Force`）。完整输出写入 `setup_windows.log`。

---

## 本机实测版本（2026-11-17 验证可用）

硬件：RTX 4060 Laptop（Ada，compute cap 8.9，8 GB 显存）、15 GB 内存、Windows 11；驱动需支持 CUDA 12.6 及以上（`nvidia-smi` 右上角），**不需要单独安装 CUDA Toolkit**（torch / warp 自带运行时）。

### A. 本软件环境

| 组件 | 版本 |
|---|---|
| Python | 3.11.9（sports2d 0.8.x 要求 ≥3.11） |
| PySide6 | 6.12.0 |
| sports2d | 0.8.34 |
| mujoco | 4.0.1 |
| opencv-python | 5.0.0.93 |
| numpy / pandas / matplotlib | 2.4.6 / 3.1.1 / 3.11.2 |
| psutil | 7.2.2 |
| torch（可选，仅用于显示 CUDA 状态） | 2.14.1+cu126 |

`requirements.txt` 只列通用依赖；torch 因 CUDA 版本随机器而异，由脚本单独安装。

### B. SuperMimic / Jumper 训练环境

| 组件 | 版本 |
|---|---|
| Python | 3.11.9 |
| `mjrl-lab`（Jumper 仓库，editable） | 0.0.1 |
| torch / torchvision | 2.14.1+cu126 / 0.29.2 |
| mujoco / mujoco-warp | 3.11.0 / 3.11.0 |
| warp-lang | 1.19.0 |
| numpy / scipy / trimesh | 2.4.6 / 1.17.1 / 5.2.1 |
| tensordict | 0.15.6 |
| viser / mjviser | 1.1.1 / 0.0.17 |
| onnx / onnxruntime / onnxscript | 1.23.2 / 1.32.0 / 0.7.2 |
| tensorboard / wandb | 2.21.0 / 0.31.0 |
| rsl_rl / mjlab | 仓库内 `rl/` 下的 vendored 源码（**不要** `pip install mjlab` 或 `rsl-rl-lib`） |

必须设置 `TORCHDYNAMO_DISABLE=1`（Windows 没有 Triton，torch.compile 回退到 eager）。应用启动训练时会自动设置；手工跑脚本时要自己设。

---

## 手工安装（等价步骤，便于排查）

### A. 应用

```powershell
cd <解压目录>\crab-dance
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
# 可选（界面里显示 GPU 状态；不影响训练，训练用 B 环境的 torch）
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 机器人模型资源（首次，GitHub 不通时先设镜像）
$env:JUMPER_MIRROR="https://ghfast.top/https://raw.githubusercontent.com"
.\.venv\Scripts\python run.py --fetch
.\.venv\Scripts\python run.py --selftest
```

### B. Jumper / SuperMimic 训练栈

```powershell
git clone https://ghfast.top/https://github.com/KingKongRobotics/jumper.git D:\jumper_ws
cd D:\jumper_ws
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
# 先装 CUDA torch，避免 pip install -e . 拉成 CPU 版
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python scripts\train.py --list      # 应列出 jumper.dance*
```

注意：

- **编辑安装 `-e .` 不能省**：mjlab / rsl_rl 在 `rl/` 下，只有 editable install 建好的
  package 映射能让 `import mjlab` 找到它们。
- Jumper 官方建议 RTX 50 系（compute cap 12.x）用 cu128；40/30 系 cu126 即可。
- 若首次运行报 Warp/MSVC 编译错误，安装 Visual Studio Build Tools 的
  “使用 C++ 的桌面开发” 工作负载；本机已可用，脚本只在缺 `cl.exe` 时提示。
- 部署到实机、导出 `.app`、Rust 控制器（cargo/wasm）**不是训练必需**，本应用不涉及。

### 在应用里诊断 / 修复 / 卸载

第一次进入训练页会自动运行环境诊断；缺失组件会询问是否自动安装。之后从
「设置 → 环境诊断」可以再次检查。对话框中按组件勾选，支持：

- 安装 / 修复：补齐缺失的依赖、仓库、补丁和快捷方式
- 重装：重建训练虚拟环境或强制重装依赖
- 卸载：删除训练虚拟环境、缓存、快捷方式；应用自身虚拟环境需关闭应用后卸载

命令行等价操作：

```bat
setup_windows.bat -Action diagnose
setup_windows.bat -Action install -Components jumper,patch -FromApp
uninstall_windows.bat -Components jumper,shortcut,cache -Yes
```

`uninstall_windows.bat` 不带参数会交互确认；`-DeleteJumperRepo` 会连同
`D:\jumper_ws` 里的训练检查点一起删除，慎用。

### 在应用里开始训练

1. 打开应用（桌面快捷方式 `Jumper Studio`），导入舞蹈视频并识别；
2. 右上角「进入训练 →」；
3. 训练页底部后端选「PPO · 参考跟踪」，点「开始」；
4. 应用会自动：导出片段到 `D:\jumper_ws\tasks\jumper\dance_crab\media\crab.npz` →
   用内置的固定预训练权重 `assets/pretrained/dance_base.pt` 调用 `scripts/train.py --backend warp --device cuda:0 --checkpoint ...` →
   训练结束后用 `scripts/play.py` 录完整段策略回放。

预训练权重已随应用打包（第 999 轮的舞蹈基础模型），**不需要先做长时间基础训练**；
每次点「开始」只是以它为初始权重微调当前视频的舞蹈。要换基础权重，覆盖
`assets/pretrained/dance_base.pt` 并把 `assets/pretrained/dance_base.json` 的
`iteration` 改成对应轮次即可。

训练环境路径不是默认 `D:\jumper_ws` 时，设置环境变量 `CRAB_JUMPER_WS` 再启动应用。

### 微调耗时（RTX 4060 Laptop 实测）

| num_envs | 速度 | 每轮 | 默认 300 轮 | 显存 |
|---|---|---|---|---|
| 512 | ~12k steps/s | ≈ 1.0 s | ≈ 5 分钟 | ≈ 2 GB |
| **2048（默认）** | ~25k steps/s | **≈ 1.9 s** | **≈ 10 分钟** | **≈ 5.4 GB** |
| 4096 | ~45k steps/s | ≈ 3.8 s | ≈ 19 分钟 | ≈ 9 GB（8 GB 卡会紧张） |

参数窗口默认 2048 环境 / 300 轮，滑块可调 20–600 轮；训练可随时「停止」，
检查点保留，可在「训练历史」里查看/删除。第一次运行 Warp 会编译并缓存内核
（约 20–60 秒），之后启动更快。

`batch_size` 只作为 SuperMimic 配置记录：Jumper 的 PPO 批量固定为 `num_envs × 24`。

策略预览由应用自己的 MuJoCo 重放：`D:\jumper_ws\scripts\play.py` 已加
`--states-out`，把逐帧 root/joint 状态写成 NPZ，所以预览可以拖动旋转和暂停。
如果重新安装 Jumper 仓库，需要保留这个补丁，否则会回退成固定镜头 MP4。

---

## C. 可选：allmimic（社区版 SuperMimic，G1 人形）

如果你还想跑论文版 SuperMimic 的研究栈：

- 仓库：`Shimingyi/allmimic`（`https://ghfast.top/https://github.com/...` 克隆）；
- 环境：Python 3.11，`torch 2.14.1+cu126`、`mjlab==1.2.0`、`mujoco==3.5.0`、
  `mujoco-warp==3.5.0`、`warp-lang 1.12`、rsl-rl、viser 等；
- 必须 `TORCHDYNAMO_DISABLE=1`；
- **限制**：它内置 G1（29 DoF）的资产/任务/动作格式，不包含 Jumper 螃蟹的 22-DoF
  配置；要在螃蟹上跑需要完整的任务/机器人移植，所以本软件的「SuperMimic GPU」用的是
  Jumper 官方维护的 mjlab 全身跟踪栈（同一类 reference-guided tracking），而不是
  allmimic。allmimic 的安装不包含在一键脚本里。

---

## 常见问题

- **GitHub 连不上**：克隆和资源下载都支持 `ghfast.top` 镜像（脚本默认）；私有网络可换
  `-GithubMirror`。PyTorch 官方 index 与 PyPI 可直连。
- **`nvidia-smi` 报错 / 没显卡**：GPU 训练不可用；应用仍可用 CPU 试训（ES）。装好 NVIDIA
  驱动后重开终端。
- **`torch.cuda.is_available()` 是 False**：torch 装成了 CPU 版；用脚本重装对应 CUDA 构建。
- **Warp 编译错误**：装 VS Build Tools（“使用 C++ 的桌面开发”）。
- **训练页显示 GPU 后端不可用**：检查 `D:\jumper_ws\scripts\train.py` 是否存在，或设置
  `CRAB_JUMPER_WS`。
- **想换策略权重位置**：训练运行目录在 `D:\jumper_ws\logs\jumper\jumper.dance_crab\<时间戳>\`
  （`model_N.pt` + `.onnx`）；应用「导出权重」另存 `.pt`。
- **策略回放视频**：`%USERPROFILE%\.crab_dance_cache\jumper\policy_*.mp4`。
