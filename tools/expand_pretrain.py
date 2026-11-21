"""Expand the dance pretrain checkpoint for three appended observations.

The curriculum task appends `ref_yaw_error` (sin, cos) and `ref_height_error`
to the actor and critic observation groups. This tool keeps every existing
weight and normalizer column, appends zero input columns for the new terms
(so the policy starts with exactly the old behaviour), and expands the Adam
moments for the two first layers to the same shape.

Usage:
    python tools/expand_pretrain.py --in assets/pretrained/dance_base.pt \
        --out assets/pretrained/dance_base_v2.pt --append 3
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def _grow_normalizer(state: dict, append: int) -> None:
    for key, fill in (("obs_normalizer._mean", 0.0),
                      ("obs_normalizer._var", 1.0),
                      ("obs_normalizer._std", 1.0)):
        if key not in state:
            continue
        old = state[key]
        new = torch.full((old.shape[0], old.shape[1] + append),
                         fill, dtype=old.dtype, device=old.device)
        new[:, :old.shape[1]] = old
        state[key] = new


def _grow_first_layer(state: dict, append: int) -> tuple[int, int]:
    """Append zero columns to the first MLP layer; return old/new input dims."""
    key = "mlp.0.weight"
    old = state[key]
    dims = (int(old.shape[1]), int(old.shape[1] + append))
    new = torch.zeros((old.shape[0], dims[1]), dtype=old.dtype,
                      device=old.device)
    new[:, :dims[0]] = old
    state[key] = new
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", dest="dst", type=Path, required=True)
    parser.add_argument("--append", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = torch.load(args.src, map_location="cpu", weights_only=False)
    out = copy.deepcopy(data)
    dims = {}
    for key in ("actor_state_dict", "critic_state_dict"):
        state = out[key]
        dims[key] = _grow_first_layer(state, args.append)
        _grow_normalizer(state, args.append)

    # Adam moments for the two expanded first layers. Identify by the old
    # trailing dimension, so this does not depend on parameter order.
    opt = out.get("optimizer_state_dict") or {}
    old_dims = {dims[k][0] for k in dims}
    for entry in (opt.get("state") or {}).values():
        for field in ("exp_avg", "exp_avg_sq"):
            moment = entry.get(field)
            if (moment is not None and moment.ndim == 2
                    and int(moment.shape[1]) in old_dims):
                new = torch.zeros((moment.shape[0],
                                   moment.shape[1] + args.append),
                                  dtype=moment.dtype, device=moment.device)
                new[:, :moment.shape[1]] = moment
                entry[field] = new

    # A checkpoint's Adam state carries the LR from the END of the base run
    # (often 1e-5 after the adaptive schedule). Resuming would restore that
    # tiny rate and a 300-round app fine-tune would barely move. Reset the
    # moments and use the configured fine-tune rate instead.
    for group in (opt.get("param_groups") or []):
        group["lr"] = float(args.lr)
    for entry in (opt.get("state") or {}).values():
        for field in ("exp_avg", "exp_avg_sq"):
            if field in entry:
                entry[field] = torch.zeros_like(entry[field])
        if "step" in entry:
            entry["step"] = torch.zeros_like(entry["step"])
    out["iter"] = int(args.iteration)
    infos = copy.deepcopy(out.get("infos") or {})
    if "env_state" in infos:
        infos["env_state"]["common_step_counter"] = 0
    out["infos"] = infos
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"wrote {args.dst}")
    for key, (old, new) in dims.items():
        print(f"  {key}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
