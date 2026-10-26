"""Real per-video fps, replacing the hard-coded /30 assumption.

Measured 2026-07-27 over the 413 EN videos: 26 distinct fps values, 35% of the
corpus is NOT ~30fps (12, 15, 20, 23.98, 24, 25, 50, 59.94, 60 all present),
and two entries are container artefacts (90000 = timebase, 120 on a 15s clip).
Converting a spatial annotation's `frame` index with /30 therefore mislocated
the evidence for 32% of annotations by >1s and 24% by >3s (max 61s) -- and in
news footage a >1s error is frequently a different shot. Affected: 39% of VIS
claims and 39% of the mined vis_value_swap twin pairs.
"""
import json, os

_MAP = None
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "multiventG", "fps_map.json")


def load_fps_map(path=None):
    global _MAP
    if _MAP is None:
        try:
            _MAP = json.load(open(path or _PATH))
        except Exception:
            _MAP = {}
    return _MAP


def fps_of(video_id, default=30.0):
    m = load_fps_map().get(video_id) or {}
    f = m.get("fps")
    # guard container artefacts: a timebase like 90000, or an implausible rate
    if not f or f < 5.0 or f > 70.0:
        return default
    return float(f)


def frame_to_seconds(video_id, frame_index, default_fps=30.0):
    """Annotation frame index -> seconds, using the video's REAL fps."""
    return float(frame_index) / fps_of(video_id, default_fps)


def duration_of(video_id, default=None):
    return (load_fps_map().get(video_id) or {}).get("duration", default)
