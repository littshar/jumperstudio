# Jumper Studio — agent handoff

This file is the operational spec for another AI agent or maintainer. Read it
before changing code. Keep the UI simple, the paths portable, and the training
stack external.

## What the app is

PySide6 desktop app that:

1. runs Sports2D on a dance video,
2. retargets the 2D human motion onto the **Jumper 22-DoF crab**,
3. previews it kinematically,
4. exports the clip to the Jumper/mjlab reference-tracking task and fine-tunes
   PPO from a fixed pretrained base,
5. replays the trained policy in the app.

The app itself never imports torch. All GPU training runs as a child process in
`D:\jumper_ws\.venv`.

## Entry points

| Path | Purpose |
|---|---|
| `run.py` | GUI entry point |
| `setup_windows.bat` / `.ps1` | one-click app + PPO environment installer |
| `uninstall_windows.bat` / `.ps1` | component-selective uninstaller |
| `start_windows.bat` | first-run setup (if needed) + launch, for a clean unzip |
| `package_release.ps1` | builds a clean release folder + ZIP |
| `tests/test_*.py` | offscreen smoke/unit tests, run as modules |
| `tools/` | maintenance tools: play.py patch, pretrain expander, curriculum task writer |

## Layout

```
app/
  constants.py          robot geometry, HOME, mapping gains, cache path
  core/
    crab_model.py       MJCF model, kinematic pose, renders, selftest
    retarget.py         human channels -> crab commands, claw IK, smoothing
    s2d_pipeline.py     Sports2D run, TRC parsing, track repair
    jumper_clip.py      crab table -> Jumper reference NPZ
    jumper_train.py     Jumper PPO worker, task generation, pretrain, replay
    train_worker.py     local CPU/ES trial (Smoke), TrainParams
    claw_calib.py       keyframe calibration (currently disabled)
    env_check.py        environment inventory / install commands
  ui/
    main_window.py      main page, analysis worker, playback, history
    video_panel.py      source video, detection overlay, analysis console
    crab_view.py        3D preview, ghost arms, probe/manual modes
    train_view.py       training page, live log, preview, button state
    train_params_dialog.py / reward_dialog.py / train_history_dialog.py
    joint_probe.py      joint angle / mapping slider tools
    environment_dialog.py
assets/
  jumper/               robot MJCF, URDF, meshes
  pretrained/           fixed PPO base + metadata
tools/                  patch/expand/curriculum helpers
tests/                  offscreen GUI + algorithm tests
```

## Environment

- App: Python 3.11, `.venv` in the project, requirements in `requirements.txt`.
- PPO: Jumper checkout at `D:\jumper_ws` with its own `.venv`, CUDA torch,
  warp, mjlab/rsl_rl; `CRAB_JUMPER_WS` overrides the path.
- GPU: NVIDIA; 2048 envs use about 5.2 GB VRAM.
- `setup_windows.bat` installs both. It refuses to run while training is active.
- `TORCHDYNAMO_DISABLE=1` is set for the training subprocess; Windows has no
  Triton.

## Run and test

```bat
.venv\Scripts\python run.py
.venv\Scripts\python run.py --selftest
.venv\Scripts\python -m tests.test_train_page
.venv\Scripts\python -m tests.test_floor_texture
.venv\Scripts\python -m tests.test_env_check
```

Tests should use `QT_QPA_PLATFORM=offscreen` (the test modules set it) and
`CRAB_SKIP_ENV_PROMPT=1` (set in training-page tests).

## Training contract

- Fixed base: `assets/pretrained/dance_base.pt`, metadata in
  `assets/pretrained/dance_base.json`. Do not replace it with an incompatible
  observation layout.
- The dance task actor/critic include simulation-only `ref_yaw_error` (2) and
  `ref_height_error` (1); observation dims are **212 / 278**. The metadata must
  match. Use `tools/expand_pretrain.py` if appending inputs.
- The app always runs `scripts/train.py --checkpoint <base-or-last>`; a stopped
  run's newest checkpoint is the next Start's checkpoint. Importing a new video
  resets to the fixed base.
- Training is a subprocess through `app/core/jumper_live_train.py`; no viewer
  window. The preview pane is intentionally blank during training and shows the
  recorded replay after it.
- `--headless`; no separate GLFW window. Frame JPEGs are written atomically and
  polled by the app.
- Do not start app training while a background curriculum run is active.

## UI conventions

- Simple, dark, no decorative tutorial text. Use tooltips for explanations.
- Chinese labels; no explanatory parentheses in visible labels.
- All long operations in QThread/subprocess; never block the GUI thread.
- Centralize button enable/disable in `TrainPage._sync_buttons()`; do not
  hand-toggle start/stop/preview buttons in multiple callbacks.
- Keep the training log append-only; move the cursor to the end.
- Policy preview: 1x wall-clock replay, orbit/zoom, pause. Training live render
  only on the left pane.
- Settings persist through QSettings. Keep the internal `dangz/crab-dance` app
  id so existing settings survive renames.

## External Jumper workspace

- `D:\jumper_ws` is patched for preview states:
  `scripts/play.py --states-out` writes per-frame root/joint NPZ.
  Re-apply with `tools/patch_play_states.py` if the checkout is replaced.
- The task `jumper.dance_crab` is generated by `prepare_dance_task`; its
  `media/crab.npz` comes from the current dance and is overwritten each run.
- `tasks/jumper/posture` and `tasks/jumper/jump` provide the open-source
  turning-gait and jump skill checkpoints used for warm starts.

## Packaging / clean unzip

`package_release.ps1` copies only source, assets, tools, tests, scripts and
docs into `dist/JumperStudio`, then zips it. `.venv`, caches, logs, screenshots
and training outputs are excluded.

A recipient needs Python 3.11 installed. They unzip and run
`setup_windows.bat` once (or `start_windows.bat`, which does setup on first
run). This creates the venv, fetches robot assets, installs the Jumper
workspace/PPO stack, and creates the desktop shortcut.

Do not commit or package:

```
.venv/            # rebuilt by setup
__pycache__/      # generated
*.log             # setup logs
*.npz/*.mp4 in root (dance/test scratch)
tests/*.png       # screenshots/artifacts
.cache/           # runtime cache
```

## Known limits

- Training quality depends on the Jumper workspace and GPU; the app only
  orchestrates it.
- SuperMimic/allmimic is disabled (G1-only port not finished).
- Old checkpoints with 209/275 observation dims cannot load into the 212/278
  task; retrain from the fixed base.
- The keyframe calibration UI is disabled (`claw_calib.CALIBRATION_ENABLED=False`);
  the slider-driven mapping remains.
