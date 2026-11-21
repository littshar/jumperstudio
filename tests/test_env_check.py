"""Environment inventory: component list, statuses and command builders.

Deep checks (torch/mjrl imports) are skipped here; the dialog runs them in
a worker. This guards the component contract the UI relies on.

Run:  .venv/Scripts/python -m tests.test_env_check
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from app.core.env_check import (
        diagnose, install_command, uninstall_command, SETUP_SCRIPT,
        UNINSTALL_SCRIPT,
    )

    failures: list[str] = []
    comps, info = diagnose(deep=False)
    keys = {c.key for c in comps}
    for key in ("app", "pretrain", "jumper", "patch", "shortcut", "cache"):
        if key not in keys:
            failures.append(f"missing component: {key}")
    for c in comps:
        if c.status not in ("ok", "warn", "missing", "error"):
            failures.append(f"bad status for {c.key}: {c.status}")
        if not c.title or not c.detail:
            failures.append(f"empty label/detail for {c.key}")
    if not SETUP_SCRIPT.is_file() or not UNINSTALL_SCRIPT.is_file():
        failures.append("install/uninstall scripts are missing")
    cmd = install_command(["jumper", "patch"], reinstall=True)
    if "-FromApp" not in cmd or "-Reinstall" not in cmd:
        failures.append("install command lacks FromApp/Reinstall")
    if "jumper,patch" not in cmd:
        failures.append("install command component list wrong")
    un = uninstall_command(["jumper", "shortcut"])
    if str(UNINSTALL_SCRIPT) not in un:
        failures.append("uninstall command points at the wrong script")
    if "jumper,shortcut" not in un:
        failures.append("uninstall component list wrong")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("environment inventory checks passed")
    print("  workspace:", info["workspace"])
    for c in comps:
        print(f"  {c.key:9s} {c.status:7s} {c.detail[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
