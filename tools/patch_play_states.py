"""Add `--states-out` to Jumper's scripts/play.py.

The app's policy preview replays the recorded root/joint states in its own
MuJoCo view so the user can orbit and pause. The stock replay only writes an
MP4 with a fixed camera, so this patch writes an NPZ alongside it.

Idempotent: if the flag is already present, nothing is written.

Usage:
    python tools/patch_play_states.py --play D:\\jumper_ws\\scripts\\play.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _replace(text: str, old: str, new: str, what: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"play.py anchor missing: {what}")
    return text.replace(old, new, 1)


def patch(text: str) -> str:
    text = _replace(
        text,
        '''    v.add_argument("--video-lookat-height", type=float, default=None, metavar="M",
                   help="height of the point a fixed camera looks at; needs --no-video-follow")
''',
        '''    v.add_argument("--video-lookat-height", type=float, default=None, metavar="M",
                   help="height of the point a fixed camera looks at; needs --no-video-follow")
    v.add_argument("--states-out", type=Path, default=None, metavar="NPZ",
                   help="also write the replayed root pose + joint positions per control "
                   "step (pose Nx7 wxyz, joint NxJ, joint_names), for a re-viewable replay")
''',
        "states-out option")
    text = _replace(
        text,
        '''            completed_steps = _loop(wrapped, policy, viewer, args.steps, speed,
                                    status or _speed_text, monitor, video, evaluation,
                                    stop_on_done=args.stop_on_done)''',
        '''            completed_steps = _loop(wrapped, policy, viewer, args.steps, speed,
                                    status or _speed_text, monitor, video, evaluation,
                                    stop_on_done=args.stop_on_done,
                                    states_out=args.states_out)''',
        "loop call")
    text = _replace(
        text,
        '''def _loop(env, policy, viewer, max_steps: int | None, speed: float = 1.0,
          readout=_speed_text, monitor=None, video=None, evaluation=None,
          stop_on_done: bool = False) -> int:''',
        '''def _loop(env, policy, viewer, max_steps: int | None, speed: float = 1.0,
          readout=_speed_text, monitor=None, video=None, evaluation=None,
          stop_on_done: bool = False, states_out=None) -> int:''',
        "loop signature")
    text = _replace(
        text,
        '''    obs = env.get_observations()
    step = 0''',
        '''    obs = env.get_observations()
    step = 0
    replay_states = []''',
        "state list")
    text = _replace(
        text,
        '''        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if stop_on_done and bool(dones.any()):''',
        '''        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if states_out is not None and entity is not None:
            replay_states.append((
                entity.data.root_link_pose_w[index].detach().cpu().numpy().copy(),
                entity.data.joint_pos[index].detach().cpu().numpy().copy(),
            ))
        if stop_on_done and bool(dones.any()):''',
        "state collection")
    text = _replace(
        text,
        '''        step += 1
    return step


if __name__ == "__main__":''',
        '''        step += 1
    if states_out is not None:
        import numpy as np
        states_out.parent.mkdir(parents=True, exist_ok=True)
        if replay_states:
            pose = np.stack([p for p, _ in replay_states])
            joint = np.stack([q for _, q in replay_states])
        else:
            pose = np.zeros((0, 7), dtype=np.float32)
            joint = np.zeros((0, 0), dtype=np.float32)
        names = list(entity.joint_names) if entity is not None else []
        dt = float(getattr(unwrapped, "step_dt", 0.02))
        np.savez(states_out, pose=pose, joint=joint,
                 joint_names=np.array(names), dt=np.array(dt),
                 fps=np.array(1.0 / dt if dt > 0 else 50.0))
        print(f"[mjrl] replay states {states_out.resolve()} "
              f"({len(replay_states)} steps)")
    return step


if __name__ == "__main__":''',
        "state save")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", type=Path, required=True)
    args = parser.parse_args()
    path = args.play
    if not path.is_file():
        print(f"play.py not found: {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    if "--states-out" in text:
        print("play.py already patched")
        return 0
    try:
        patched = patch(text)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"patched {path} (backup {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
