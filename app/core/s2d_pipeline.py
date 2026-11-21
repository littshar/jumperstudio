"""Sports2D invocation and result parsing.

Strategy: run Sports2D programmatically (kwargs, never touching internals),
then read back the TRC pose file it writes into result_dir. TRC marker names
are matched tolerantly (upper/lower, with/without LEFT/RIGHT prefixes, plus
derived neck / mid-hip fallbacks), because they change across Sports2D
versions and pose models (COCO17 vs HALPE26).

Results are cached per video content hash, so re-importing a clip is instant.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.constants import CACHE_DIR, MAX_PERSONS
from app.core.retarget import KP_NAMES, Keypoints

# Alias table: canonical -> acceptable normalised marker names.
_ALIASES: dict[str, tuple[str, ...]] = {
    "nose": ("nose", "head"),
    "neck": ("neck", "cervical", "throat"),
    "shoulder_l": ("lshoulder", "leftshoulder", "lsho", "shoulderl"),
    "shoulder_r": ("rshoulder", "rightshoulder", "rsho", "shoulderr"),
    "elbow_l": ("lelbow", "leftelbow", "lelb"),
    "elbow_r": ("relbow", "rightelbow", "relb"),
    "wrist_l": ("lwrist", "leftwrist", "lwri"),
    "wrist_r": ("rwrist", "rightwrist", "rwri"),
    "hip_l": ("lhip", "lefthip", "lhipjoint", "lhiptest"),
    "hip_r": ("rhip", "righthip", "rhipjoint", "rhiptest"),
    "knee_l": ("lknee", "leftknee", "lkne"),
    "knee_r": ("rknee", "rightknee", "rkne"),
    "ankle_l": ("lankle", "leftankle", "lank", "lfoot"),
    "ankle_r": ("rankle", "rightankle", "rank", "rfoot"),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


@dataclass
class S2DResult:
    kps: Keypoints                 # primary (largest) person
    result_dir: Path
    fps_sample: float      # analysed sample rate
    n_frames: int
    persons: list[Keypoints] = None  # all detected, largest first
    repair: dict = None    # track-repair stats (raw / merged / dropped / valid%)


def video_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            break  # path + first MB is plenty for cache identity
    return h.hexdigest()[:16]


class _ProgressHandler(logging.Handler):
    """Kept for potential logging-based progress; Sports2D 0.8 uses tqdm."""

    frame_re = re.compile(r"(\d+)/(\d+)")

    def __init__(self, cb):
        super().__init__(level=logging.INFO)
        self.cb = cb
        self.last = -1

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        m = self.frame_re.search(msg)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            if total > 1 and cur != self.last:
                self.last = cur
                self.cb(min(cur / total, 1.0), cur, total)


class _TqdmTail(io.TextIOBase):
    """stderr tee that reads Sports2D's tqdm bar ("Processing video
    stream...: 42%|## | 101/240") and reports the frame fraction.

    tqdm is constructed inside Sports2D.process() and captures whatever
    sys.stderr is at that moment, so swapping it around the call is
    enough. Everything written is passed through to the real stderr.
    """

    pct_re = re.compile(r"(\d{1,3})%")
    frac_re = re.compile(r"(\d+)/(\d+)")

    def __init__(self, real, cb, log=None):
        super().__init__()
        self._real = real
        self._cb = cb
        self._log = log
        self._buf = ""
        self._last = -1.0

    def _report(self, s: str) -> None:
        m = self.pct_re.search(s) or None
        if m is None:
            m2 = self.frac_re.search(s)
            if m2 and int(m2.group(2)) > 1:
                frac = int(m2.group(1)) / int(m2.group(2))
            else:
                return
        else:
            frac = int(m.group(1)) / 100.0
        frac = min(frac, 0.99)  # 1.0 is emitted by the caller when done
        if frac > self._last:
            self._last = frac
            self._cb(frac)

    def write(self, s) -> int:
        try:
            if self._real is not None:
                self._real.write(s)
        except Exception:  # noqa: BLE001 - progress must never crash analysis
            pass
        try:
            self._report(s)
        except Exception:  # noqa: BLE001
            pass
        if self._log is not None and isinstance(s, str):
            try:
                self._buf += s
                parts = re.split(r"[\r\n]+", self._buf)
                self._buf = parts.pop()
                pieces = [p for p in parts if p.strip()]
                if self._buf.strip():
                    pieces.append(self._buf)
                for piece in pieces:
                    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", piece.strip())
                    if clean:
                        self._log(clean)
            except Exception:  # noqa: BLE001
                pass
        return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        try:
            if self._real is not None:
                self._real.flush()
        except Exception:  # noqa: BLE001
            pass

    def writable(self) -> bool:
        return True


def _tracking_max_distance(video: Path) -> int:
    """Association gate scaled to the video resolution (px)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280.0
    finally:
        cap.release()
    return int(max(250, round(width / 8.0)))


def run_sports2d(video: Path, progress=lambda *a: None,
                 log=None) -> Path:
    """Run Sports2D on one video; returns the result directory.

    Only the config dict goes in and only the output FILES are read back --
    the library's return types are unstable across versions. Progress is a
    tqdm bar on stderr, so the UI shows an indeterminate spinner instead.
    """
    out_dir = CACHE_DIR / video_hash(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = out_dir / "_done"
    # Trust the flag only when a pose file is actually there: Sports2D can
    # finish "successfully" yet delete the only detected person (e.g. a
    # cartoon stick figure), which would poison the cache forever.
    if done_flag.exists() and any(out_dir.rglob("*.trc")):
        if log:
            log("使用已缓存的姿态识别结果")
        return out_dir
    done_flag.unlink(missing_ok=True)

    from Sports2D import Sports2D

    cfg = {
        "base": {
            "video_input": [str(video)],
            "result_dir": str(out_dir),
            "save_vid": False,
            "save_img": False,
            "save_pose": True,
            # Sports2D 0.8.34 upstream bug: all_frames_angles is FILLED only
            # when (save_angles AND calculate_angles) but USED when either
            # is on, and its bad-person deletion branch needs the "flipped"
            # pose array built only when angles are on. Both True (their
            # default demo config) is the only crash-free combination; the
            # angle CSVs it writes are simply ignored.
            "calculate_angles": True,
            "save_angles": True,
            "show_realtime_results": False,
            # 'all' -- let every viable track survive to a TRC, and OUR
            # repair pass (load_persons) do the selecting. Reason: on
            # crowded clips the tracker spawns dozens of fragment slots and
            # largest_size then ranks glitch slots (a few huge-bbox frames)
            # over steady dancers -- measured on the 4K group dance: 5 of
            # the 6 "largest" slots got deleted as unusable right after
            # selection. Junk tracks never write a TRC (Sports2D deletes
            # them first), so 'all' only adds cheap per-person postproc.
            "nb_persons_to_detect": "all",
            # 'on_click' (their default) pops a matplotlib window and blocks
            # forever waiting for a mouse click -- unusable from a worker
            # thread. 'largest_size' picks the biggest person in frame, which
            # is the dancer in practice, with zero interaction.
            "person_ordering_method": "largest_size",
            "time_range": [],
        },
        "pose": {
            "mode": "balanced",
            "det_frequency": 1,
            # Group dances: predicting displacement carries tracks through
            # crossings, and relaxed likelihood gates keep frames that the
            # strict defaults reject (which deletes whole people: "Less
            # than 10 valid frames in a row").
            "predict_displacement": True,
            "keypoint_likelihood_threshold": 0.25,
            "average_likelihood_threshold": 0.35,
            "keypoint_number_threshold": 0.2,
            # Pose2Sim's tracker RETIRES a slot after max_unseen_time and
            # never re-attaches to it -- with their 1 s default, a dancer
            # occluded for a moment reappears as a brand-new "person", so a
            # group dance explodes into a pile of short track fragments
            # (measured: 11+ slots for ~5 humans). 4 s keeps occluded
            # dancers re-attachable.
            "max_unseen_time": 4.0,
            # Association gate, in PIXELS: the 250 px default is tuned for
            # ~720p. Scale with the video width so 1080p/4K clips do not
            # shatter tracks just because everything is bigger.
            "max_distance": _tracking_max_distance(video),
        },
        "px_to_meters_conversion": {
            "to_meters": False,   # keep pixel coordinates for the overlay
            "make_c3d": False,
            "save_calib": False,
        },
        # NOTE: the section key is 'post-processing' (hyphen) in Sports2D's
        # config schema. A wrong key here silently merges as a new section
        # and their defaults (show_graphs=True) pop an interactive plot
        # window that blocks forever in a worker thread.
        "post-processing": {
            "show_graphs": False,
            "save_graphs": False,
        },
    }
    # tqdm builds its bar on sys.stderr as captured HERE, so the tee sees
    # every "Processing video stream...: N%" refresh and the UI gets a
    # determinate progress bar instead of an indeterminate spinner.
    real_stderr = sys.stderr
    sys.stderr = _TqdmTail(real_stderr, progress, log)
    try:
        Sports2D.process(cfg)
    finally:
        sys.stderr = real_stderr

    if not any(out_dir.rglob("*.trc")):
        raise RuntimeError(
            "姿态识别没有输出结果 · 换一段真人舞蹈视频试试")
    done_flag.touch()
    return out_dir


# ------------------------------------------------------------------ TRC ----
def _read_trc(path: Path) -> tuple[np.ndarray, list[str], dict[int, dict[str, int]], np.ndarray]:
    """Parse a TRC file.

    Returns (times, marker_names, colmap) where colmap maps marker index ->
    {'x': col, 'y': col} zero-based data column offsets. Handles both the
    OpenSim layout (X1 X2 Y1 Y2 subheader) and plain interleaved triples.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise ValueError(f"TRC too short: {path}")

    # Header row 3 holds Frame# Time and marker names (each repeated 3x).
    name_row = None
    for i, line in enumerate(lines[:8]):
        if re.match(r"\s*Frame#", line):
            name_row = i
            break
    if name_row is None:
        raise ValueError(f"No Frame# header row in {path}")

    names = [c.strip() for c in lines[name_row].split("\t")]
    markers = [n for n in names[2:] if n]
    # de-duplicate keeping order (X Y Z repeats)
    seen: list[str] = []
    for n in markers:
        if not seen or seen[-1] != n:
            seen.append(n)
    markers = seen

    # Sub-header: coordinate labels like X1 X2 Y1 Y2 Z1 Z2.
    colmap: dict[int, dict[str, int]] = {}
    sub = [c.strip() for c in lines[name_row + 1].split("\t")]
    sub = sub[2:]
    if sub and all(re.fullmatch(r"[XYZxyz]\d*", s) for s in sub if s):
        # OpenSim layout: label letter + marker index (1-based).
        for data_col, label in enumerate(sub):
            if not label:
                continue
            coord = label[0].lower()
            idx_s = label[1:]
            idx = int(idx_s) - 1 if idx_s.isdigit() else None
            if idx is None:
                continue
            colmap.setdefault(idx, {})[coord] = data_col
    else:
        # Interleaved triples: m0x m0y m0z m1x ...
        for m_idx in range(len(markers)):
            colmap[m_idx] = {"x": 3 * m_idx, "y": 3 * m_idx + 1}

    # Data rows.
    rows = []
    for line in lines[name_row + 2:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        try:
            rows.append([float(p) if p.strip() else np.nan for p in parts[2:]])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No data rows in {path}")
    data = np.asarray(rows, dtype=float)
    times = np.arange(len(data), dtype=float)  # replaced by caller if real
    # Frame# and Time are the first two original columns; recover Time.
    raw = []
    for line in lines[name_row + 2:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                raw.append(float(parts[1]))
            except ValueError:
                pass
    if raw:
        times = np.asarray(raw)
    return times, markers, colmap, data


def load_persons(result_dir: Path) -> list[Keypoints]:
    """Read every person's px TRC under result_dir, best first.

    Sports2D 0.8 writes <result_dir>/<video>_Sports2D/<video>_px_personNN.trc
    (one file per detected person, ordered by the configured
    person_ordering_method); its post-processing may drop people entirely,
    so fewer files than requested detections is normal. Falls back to any
    TRC found for older versions. The parsed tracks then go through the
    repair pass below (fragment stitching + duplicate merge + gating).
    """
    return load_persons_with_stats(result_dir)[0]


def load_keypoints(result_dir: Path) -> Keypoints:
    """Primary (largest) person, kept for compatibility."""
    return load_persons(result_dir)[0]


# ------------------------------------------------------ track repair ----
# Pose2Sim's tracker retires a person slot after max_unseen_time (1 s in
# Sports2D's defaults) and NEVER re-attaches to a retired slot, so in group
# dances -- where dancers occlude each other for a second or two all the
# time -- every reappearance spawns a new "person". One clip measured 11+
# track slots for ~5 humans, and each surviving TRC held a ~2 s fragment
# (crabs then froze most of the timeline). The pass below repairs what the
# tracker broke, purely from the saved TRCs -- cached results improve
# without re-running detection:
#   * sequential fragments whose endpoints nearly touch get stitched
#   * near-coincident co-existing tracks (same human, two slots) merge
#   * mostly-static tracks (Sports2D fills occlusion with the last value,
#     so filler frames are exactly frozen) get dropped
#   * survivors re-rank by real (moving) frame count, best first
MIN_TRACK_FRAMES = 5           # below this a track can't anchor decisions
STITCH_MAX_GAP_S = 10.0        # fragments farther apart in time stay split
STITCH_MAX_DIST_TORSO = 1.15   # endpoint distance, in mean torso units
DUPLICATE_MAX_DIST_TORSO = 0.8
SIZE_MAX_RATIO = 1.7           # same human => similar body size
MIN_LIVE_RATIO = 0.15          # drop tracks that move in < 15% of frames
                               # (validated: healthy group-dance tracks land
                               # at 53-84%, camera-pan intro artefacts ~9%)


def _person_series(kps: Keypoints) -> tuple:
    """Per-frame (valid, live, centroid, torso) for repair decisions.

    `live` = the frame actually moved. Real detections at dance framerates
    never come out bit-identical across every keypoint; held filler does.
    """
    n = len(kps.times)
    hip = (kps.xy("hip_l") + kps.xy("hip_r")) / 2
    sho = (kps.xy("shoulder_l") + kps.xy("shoulder_r")) / 2
    torso = np.linalg.norm(sho - hip, axis=1)
    valid = ~(np.isnan(hip[:, 0]) | np.isnan(sho[:, 0]))
    live = valid.copy()
    if n > 1:
        allxy = np.concatenate([kps.xy(name) for name in KP_NAMES], axis=1)
        disp = np.abs(np.nan_to_num(np.diff(allxy, axis=0))).sum(axis=1)
        live = valid & np.concatenate([[True], disp >= 1e-9])
    return valid, live, (hip + sho) / 2, torso


def _merge_group(members: list[tuple]) -> Keypoints:
    """One Keypoints out of several tracks of the same human.

    Members are (kps, valid, live). Each frame takes the first member that
    is LIVE there, else the first that is merely valid -- a real
    reappearing detection always beats another member's held filler.
    """
    order = sorted(members, key=lambda m: int(m[1].sum()), reverse=True)
    base = order[0][0]
    n = len(base.times)
    pick = np.full(n, -1)
    for mask_of in (lambda m: m[2], lambda m: m[1]):   # live pass, then valid
        for idx, m in enumerate(order):
            free = pick < 0
            pick[mask_of(m) & free] = idx
    xyz: dict[str, np.ndarray] = {}
    conf: dict[str, np.ndarray] = {}
    for name in KP_NAMES:
        col = np.full((n, 2), np.nan)
        for idx, (kps, _v, _l) in enumerate(order):
            rows = pick == idx
            if rows.any():
                col[rows] = kps.xy(name)[rows]
        xyz[name] = col
        conf[name] = np.where(np.isnan(col[:, 0]), 0.0, 1.0)
    merged = Keypoints(xyz=xyz, conf=conf, times=base.times)
    merged.__dict__["trc_path"] = base.__dict__.get("trc_path")
    return merged


def _repair_persons(persons: list[Keypoints]) -> tuple[list[Keypoints], dict]:
    """Stitch / merge / gate the parsed tracks; returns (best-first, stats)."""
    n_raw = len(persons)
    stats = {"raw": n_raw, "merged": 0, "dropped": 0, "final": 0,
             "live_pct": []}
    if n_raw == 0:
        return persons, stats
    series = [_person_series(kps) for kps in persons]

    if n_raw > 1:
        parent = list(range(n_raw))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n_raw):
            vi, _li, ci, ti = series[i]
            fi = np.where(vi)[0]
            if len(fi) < MIN_TRACK_FRAMES:
                continue
            ti_med = float(np.nanmedian(ti)) or 1.0
            for j in range(i + 1, n_raw):
                vj, _lj, cj, tj = series[j]
                fj = np.where(vj)[0]
                if len(fj) < MIN_TRACK_FRAMES:
                    continue
                tj_med = float(np.nanmedian(tj)) or 1.0
                scale = (ti_med + tj_med) / 2.0
                if (max(ti_med, tj_med)
                        > SIZE_MAX_RATIO * min(ti_med, tj_med)):
                    continue   # very different body sizes: not the same human
                both = vi & vj
                shorter = min(len(fi), len(fj))
                if both.sum() >= 0.3 * shorter:
                    # Co-existing tracks: duplicates only if coincident.
                    if both.any():
                        d = float(np.median(np.linalg.norm(
                            ci[both] - cj[both], axis=1)))
                        if d < DUPLICATE_MAX_DIST_TORSO * scale:
                            parent[find(j)] = find(i)
                    continue
                if both.sum() >= 0.1 * shorter:
                    continue    # neighbours: overlap but clearly 2 people
                # Sequential windows -> stitch if the endpoints nearly touch.
                if fi[-1] <= fj[0]:
                    d = float(np.linalg.norm(ci[fi[-1]] - cj[fj[0]]))
                    gap = float(persons[j].times[fj[0]]
                                - persons[i].times[fi[-1]])
                elif fj[-1] <= fi[0]:
                    d = float(np.linalg.norm(cj[fj[-1]] - ci[fi[0]]))
                    gap = float(persons[i].times[fi[0]]
                                - persons[j].times[fj[-1]])
                else:
                    continue
                if gap <= STITCH_MAX_GAP_S and d < STITCH_MAX_DIST_TORSO * scale:
                    parent[find(j)] = find(i)

        groups: dict[int, list[int]] = {}
        for i in range(n_raw):
            groups.setdefault(find(i), []).append(i)
    else:
        groups = {0: [0]}

    ranked: list[tuple[Keypoints, int, float]] = []
    for idxs in groups.values():
        n0 = len(persons[idxs[0]].times)
        members = [(persons[i], series[i][0], series[i][1])
                   for i in idxs if len(persons[i].times) == n0]
        if not members:
            continue
        if len(members) > 1:
            merged = _merge_group(members)
            stats["merged"] += len(members) - 1
            valid, live, _c, _t = _person_series(merged)
        else:
            merged = persons[idxs[0]]
            valid, live = series[idxs[0]][0], series[idxs[0]][1]
        n = len(merged.times) or 1
        ranked.append((merged, int(live.sum()), float(live.sum() / n)))

    ranked.sort(key=lambda r: -r[1])
    kept = []
    for kps, _live_n, live_ratio in ranked:
        # Always keep the best track, however poor; drop the rest when they
        # are essentially frozen filler (they render as statues).
        if kept and live_ratio < MIN_LIVE_RATIO:
            stats["dropped"] += 1
            continue
        kept.append((kps, live_ratio))

    stats["final"] = len(kept)
    stats["live_pct"] = [round(r * 100) for _k, r in kept]
    return [k for k, _r in kept], stats


def load_persons_with_stats(result_dir: Path) -> tuple[list[Keypoints], dict]:
    """Parse + repair; returns (persons best-first, repair stats)."""
    trcs = list(result_dir.rglob("*_px_person*.trc"))
    if trcs:
        def person_no(p: Path):
            m = re.search(r"person(\d+)", p.stem)
            return int(m.group(1)) if m else 0
        trcs = sorted(trcs, key=person_no)
    else:
        trcs = sorted(result_dir.rglob("*.trc"))
    if not trcs:
        raise FileNotFoundError(f"no TRC pose file in {result_dir}")

    norm_to_canon: dict[str, str] = {}
    for canon, aliases in _ALIASES.items():
        for a in aliases:
            norm_to_canon[a] = canon

    persons: list[Keypoints] = []
    for trc in trcs:
        times, markers, colmap, data = _read_trc(trc)
        n = data.shape[0]
        found: dict[str, np.ndarray] = {}
        conf: dict[str, np.ndarray] = {}
        for m_idx, mname in enumerate(markers):
            canon = norm_to_canon.get(_norm(mname))
            if canon is None or canon in found:
                continue
            cols = colmap.get(m_idx)
            if not cols or "x" not in cols or "y" not in cols:
                continue
            found[canon] = data[:, [cols["x"], cols["y"]]]
            # TRC carries no confidence; presence == confident.
            conf[canon] = np.ones(n)

        # Derived: neck from shoulders, hips already left/right.
        if "neck" not in found and {"shoulder_l", "shoulder_r"} <= found.keys():
            found["neck"] = (found["shoulder_l"] + found["shoulder_r"]) / 2
            conf["neck"] = np.ones(n)

        missing = [k for k in KP_NAMES if k not in found]
        if missing:
            continue  # this person's file is unusable; skip, don't fail all

        kps = Keypoints(xyz=found, conf=conf, times=times)
        kps.__dict__["trc_path"] = trc  # noqa: SLF001 - provenance only
        persons.append(kps)

    if not persons:
        raise ValueError("TRC files present but no usable person keypoints")
    return _repair_persons(persons)


def process_video(video: Path, progress=lambda *a: None,
                  log=None) -> S2DResult:
    """Full pipeline step: run Sports2D and parse keypoints."""
    result_dir = run_sports2d(Path(video), progress=progress, log=log)
    if log:
        log("姿态识别完成，正在解析并修复轨迹…")
    persons, repair = load_persons_with_stats(result_dir)
    if log:
        log(f"轨迹就绪 · {len(persons)} 人")
    kps = persons[0]
    t = kps.times
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    fps = 1.0 / dt if dt > 0 else 30.0
    return S2DResult(kps=kps, result_dir=result_dir, fps_sample=fps,
                     n_frames=len(t), persons=persons, repair=repair)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    video = Path(sys.argv[1])
    res = process_video(video)
    print(f"frames={res.n_frames} sample_fps={res.fps_sample:.2f}")
    print("markers:", sorted(res.kps.xyz))
    print("t:", res.kps.times[:3], "...", res.kps.times[-1])
