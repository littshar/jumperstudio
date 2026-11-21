"""Jumper `scripts/train.py` + a live preview frame, for the crab-dance app.

The app runs this file with the Jumper workspace as the working directory and
that workspace's Python. It loads `scripts/train.py` *as a module* -- not by
rewriting it -- and replaces its `maybe_viewer` hook with a context manager that
renders one environment offscreen to JPEGs at a few frames a second. Everything
else is the stock training entry point: task/asset resolution, TensorBoard off,
checkpoints, resume, logging.

Why not Jumper's live window? It is a separate GLFW window, which the app should
not spawn. This keeps the picture inside the app: the app's training worker sets
`CRAB_JUMPER_FRAME_DIR`, polls it, and shows the newest frame in the
"训练渲染" pane. The camera/recorder is Jumper's own (`mjrl.viewer.video`), so it
is the same renderer its MP4s use.

Each frame is written to a **new numbered file** (atomic rename) and only the
newest few are kept, so the training process never replaces a file the app may
be reading -- a plain fixed JPEG path can fail with a sharing violation on
Windows while the app loads it.

Environment variables (all optional):
    CRAB_JUMPER_FRAME_DIR       directory for the JPEG frames. Unset = no frames.
    CRAB_JUMPER_FRAME_FPS       frames per second of simulation time (default 5).
    CRAB_JUMPER_FRAME_SIZE      "WIDTHxHEIGHT" (default 480x360).

The frame hook is best effort: if the renderer cannot start (no GL context,
mjlab internals changed), training continues untouched. The app falls back to a
static pane; nothing about the run depends on this file.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path


def _load_train_module():
    """Load <cwd>/scripts/train.py without executing its __main__ block."""
    root = Path.cwd()
    scripts = root / "scripts"
    train = scripts / "train.py"
    if not train.is_file():
        raise SystemExit(
            f"[crab] this script must be run with the Jumper checkout as the "
            f"working directory; {train} does not exist")
    # train.py uses `from _cli import ...`, so scripts/ must be importable.
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("jumper_train_entry", train)
    module = importlib.util.module_from_spec(spec)
    sys.modules["jumper_train_entry"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _writer_class():
    """A JPEG writer built on Jumper's replay renderer, or None."""
    try:
        import imageio.v2 as imageio
        from mjrl.viewer.video import (
            _ReplayRenderer, _camera_config, _first_pose,
        )
    except Exception:  # noqa: BLE001 - preview only
        return None

    class LiveFrameWriter:
        def __init__(self, env, folder: Path, fps: float,
                     width: int = 480, height: int = 360,
                     env_index: int = 0):
            self.env = getattr(env, "unwrapped", env)
            self.folder = Path(folder)
            self.folder.mkdir(parents=True, exist_ok=True)
            self.fps = float(fps)
            self._dt = float(self.env.step_dt)
            self._steps = 0
            self._drawn = 0
            self._seq = 0
            self._imageio = imageio
            sim = self.env.sim
            model = getattr(sim, "render_model", sim.mj_model)
            cfg = _camera_config(
                self.env, model, width=width, height=height,
                env_index=env_index,
                camera_distance=None,
                pose=_first_pose(self.env, env_index),
                azimuth=None, elevation=None, lookat_height=None, follow=True)
            self._renderer = _ReplayRenderer(sim, cfg, self.env.scene)
            self._renderer.initialize()

        def maybe_update(self) -> None:
            """Render after one control step, at most `fps` frames per sim second."""
            self._steps += 1
            due = 1 + int(self._steps * self._dt * self.fps)
            if due <= self._drawn:
                return
            self._drawn = due
            self._renderer.update(self.env.sim.data)
            rgb = self._renderer.render()
            self._seq += 1
            final = self.folder / f"live_{self._seq:06d}.jpg"
            tmp = self.folder / f".tmp_{self._seq:06d}.jpg"
            self._imageio.imwrite(tmp, rgb, quality=82)
            os.replace(tmp, final)      # a complete frame appears at once
            # Keep the newest few: the app always reads the newest, which is
            # never the one pruned.
            frames = sorted(self.folder.glob("live_*.jpg"))
            for old in frames[:-3]:
                with contextlib.suppress(OSError):
                    old.unlink()

        def close(self) -> None:
            renderer, self._renderer = self._renderer, None
            if renderer is not None:
                with contextlib.suppress(Exception):
                    renderer.close()

    return LiveFrameWriter


def _live_viewer(module):
    """The `maybe_viewer(env, args)` replacement: an offscreen frame writer."""
    Writer = _writer_class()

    @contextlib.contextmanager
    def maybe_live(env, args):
        def size_arg(name):
            text = os.environ.get(name, "480x360")
            try:
                return tuple(int(v) for v in text.lower().split("x", 1))
            except ValueError:
                return 480, 360

        def make_writer(folder: str, env_index: int, fps_env: str,
                        size_env: str):
            folder = (folder or "").strip()
            if not folder or Writer is None:
                return None
            fps = float(os.environ.get(fps_env, "5") or 5)
            width, height = size_arg(size_env)
            try:
                w = Writer(env, Path(folder), fps, width, height,
                           env_index=env_index)
                print(f"[crab] live frames -> {folder} env{env_index} "
                      f"({width}x{height} @ {fps:g} fps)", flush=True)
                return w
            except Exception as exc:  # noqa: BLE001 - preview only
                print(f"[crab] live frames unavailable for env{env_index}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                return None

        writers = []
        primary = make_writer(
            os.environ.get("CRAB_JUMPER_FRAME_DIR", ""), 0,
            "CRAB_JUMPER_FRAME_FPS", "CRAB_JUMPER_FRAME_SIZE")
        if primary is not None:
            writers.append(primary)
        num_envs = int(getattr(getattr(env, "unwrapped", env), "num_envs", 1))
        if num_envs > 1:
            preview = make_writer(
                os.environ.get("CRAB_JUMPER_PREVIEW_DIR", ""), 1,
                "CRAB_JUMPER_PREVIEW_FPS", "CRAB_JUMPER_PREVIEW_SIZE")
            if preview is not None:
                writers.append(preview)
        if not writers:
            yield None
            return
        original_step = env.step

        def step(actions):
            result = original_step(actions)
            for w in list(writers):
                try:
                    w.maybe_update()
                except Exception as exc:  # noqa: BLE001
                    writers.remove(w)
                    print(f"[crab] live frames stopped: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            return result

        try:
            env.step = step          # instance attribute shadows the method
        except Exception as exc:  # noqa: BLE001 - never take the run down
            print(f"[crab] live frames off (cannot hook step): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            for w in writers:
                w.close()
            yield None
            return
        try:
            yield primary
        finally:
            with contextlib.suppress(Exception):
                env.step = original_step
            for w in writers:
                w.close()

    return maybe_live


def main() -> None:
    module = _load_train_module()
    module.maybe_viewer = _live_viewer(module)
    module.main()


if __name__ == "__main__":
    main()
