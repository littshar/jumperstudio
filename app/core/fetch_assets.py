"""One-time downloader for the jumper robot model.

Fetches assets/jumper/jumper.xml from the upstream repository, parses every
mesh it references, and downloads each STL to the same relative path so the
MJCF's meshdir="./" keeps working. Idempotent: existing non-empty files are
skipped, failures retry twice.

Network note: set JUMPER_MIRROR to a prefix such as
    https://ghproxy.net/https://raw.githubusercontent.com
if raw.githubusercontent.com is unreachable.
"""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from app.constants import JUMPER_DIR, JUMPER_RAW_BASE, MIRROR_ENV


def _base_url() -> str:
    mirror = os.environ.get(MIRROR_ENV, "").rstrip("/")
    if mirror:
        # Mirror prefixes usually want the full original URL appended.
        if "raw.githubusercontent.com" not in mirror:
            return f"{mirror}/https://raw.githubusercontent.com/KingKongRobotics/jumper/main/assets/jumper"
        return f"{mirror}/KingKongRobotics/jumper/main/assets/jumper"
    return JUMPER_RAW_BASE


def _download_curl(url: str, dest: Path) -> bool:
    """Fallback via the curl binary -- the local network to GitHub is flaky."""
    import shutil
    import subprocess

    curl = shutil.which("curl")
    if not curl:
        return False
    try:
        subprocess.run(
            [curl, "-sL", "--fail", "--max-time", "90", "--retry", "3", url, "-o", str(dest)],
            check=True,
        )
        return dest.exists() and dest.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crab-dance/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            if dest.stat().st_size > 0:
                return True
        except Exception as exc:  # noqa: BLE001 - report and retry
            if attempt < retries:
                time.sleep(1.5 + attempt * 1.5)
                continue
        dest.unlink(missing_ok=True)
    if _download_curl(url, dest):
        return True
    print(f"  FAILED {url}")
    return False


def _mesh_files_in_xml(xml_path: Path) -> list[str]:
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r'file="([^"]+)"', text)


def fetch_jumper_assets(force: bool = False) -> bool:
    """Download jumper.xml plus all referenced meshes. True on success."""
    base = _base_url()
    JUMPER_DIR.mkdir(parents=True, exist_ok=True)

    xml_path = JUMPER_DIR / "jumper.xml"
    if force or not xml_path.exists() or xml_path.stat().st_size == 0:
        print(f"downloading jumper.xml from {base}")
        if not _download(f"{base}/jumper.xml", xml_path):
            return False
    else:
        print("jumper.xml already present")

    meshes = _mesh_files_in_xml(xml_path)
    print(f"{len(meshes)} meshes referenced by the model")

    failures = 0
    for rel in meshes:
        dest = (JUMPER_DIR / rel).resolve()
        # Guard against path escape from malformed XML.
        if JUMPER_DIR.resolve() not in dest.parents and dest != JUMPER_DIR:
            print(f"  SKIP suspicious path {rel!r}")
            failures += 1
            continue
        if not force and dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = _download(f"{base}/{rel.replace(chr(92), '/')}", dest)
        if not ok:
            failures += 1
        else:
            print(f"  ok {rel}")

    # Attribution: keep upstream license alongside the model files.
    notice = JUMPER_DIR / "NOTICE"
    if not notice.exists():
        notice.write_text(
            "Robot model (jumper.xml, meshes) from KingKongRobotics/jumper,\n"
            "Copyright 2026 KingKong Robotics, Apache-2.0.\n"
            "https://github.com/KingKongRobotics/jumper\n",
            encoding="utf-8",
        )

    total = 1 + len(meshes)
    present = sum(
        1 for rel in meshes
        if (JUMPER_DIR / rel).exists() and (JUMPER_DIR / rel).stat().st_size > 0
    )
    print(f"{present + 1}/{total} files present "
          f"({'complete' if failures == 0 else f'{failures} FAILED'})")
    return failures == 0


if __name__ == "__main__":
    ok = fetch_jumper_assets(force="--force" in sys.argv)
    sys.exit(0 if ok else 1)
