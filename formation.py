"""
FORMATION KEY + POSITION GRADING
================================
The "answer key": you tell us, per formation, where each named dancer SHOULD be.
We then (1) attach real names to the 16 stable track IDs, and (2) grade who was
OUT OF POSITION in each formation. This is a geometric measurement (distance from
the charted spot), so unlike fine-grained beat-timing it is NOT noise-limited.

COORDINATE CONVENTION (v1, image-normalized)
--------------------------------------------
Positions are in normalized image space: x in [0,1] left->right, y in [0,1]
top->bottom, as the dancer appears in the camera. (0.5, 0.5) is frame center.
This needs no floor-corner tapping. A later precision upgrade can swap in the
verified floor homography so spacing is perspective-corrected.

KEY FILE FORMAT (formation_key.json)
------------------------------------
{
  "roster": ["Aanya", "Bri", ... 16 names ...],
  "name_anchor_formation": "opening",     # which formation to attach names on
  "out_of_position_threshold": 0.07,      # fraction of frame; >this = flagged
  "formations": [
    {
      "name": "opening",
      "start": "0:00", "end": "0:18",     # when this formation is HELD
      "positions": {                      # normalized (x,y) per dancer
        "Aanya": [0.20, 0.55], "Bri": [0.32, 0.55], ...
      }
    },
    ...
  ]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------
# Key loading / validation
# ----------------------------------------------------------------------------
def _parse_time(t, fps: float) -> int:
    """'1:23' or '83' or 83.0 -> frame index at fps (processed-frame space)."""
    if isinstance(t, (int, float)):
        sec = float(t)
    else:
        parts = str(t).split(":")
        sec = float(parts[0]) if len(parts) == 1 else int(parts[0]) * 60 + float(parts[1])
    return int(round(sec * fps))


@dataclass
class Formation:
    name: str
    start_f: int
    end_f: int
    positions: Dict[str, Tuple[float, float]]


@dataclass
class FormationKey:
    roster: List[str]
    formations: List[Formation]
    name_anchor: str
    threshold: float

    @property
    def anchor_formation(self) -> Formation:
        for f in self.formations:
            if f.name == self.name_anchor:
                return f
        return self.formations[0]


def load_key(path: str, fps: float) -> FormationKey:
    with open(path) as fh:
        raw = json.load(fh)
    roster = list(raw["roster"])
    thr = float(raw.get("out_of_position_threshold", 0.07))
    anchor = raw.get("name_anchor_formation") or raw["formations"][0]["name"]
    forms = []
    for f in raw["formations"]:
        pos = {n: (float(p[0]), float(p[1])) for n, p in f["positions"].items()}
        forms.append(Formation(f["name"], _parse_time(f["start"], fps),
                               _parse_time(f["end"], fps), pos))
    return FormationKey(roster=roster, formations=forms,
                        name_anchor=anchor, threshold=thr)


def write_template(path: str, roster: Optional[List[str]] = None,
                   n: int = 16) -> None:
    """Write a fill-in template with a 4x4 grid example the user edits."""
    roster = roster or [f"Dancer{i+1}" for i in range(n)]
    # a simple 4x4 grid spanning the middle of the frame, as an example layout
    cols, rows = 4, 4
    grid = {}
    for i, name in enumerate(roster[:n]):
        r, c = divmod(i, cols)
        x = round(0.18 + c * (0.64 / (cols - 1)), 3)
        y = round(0.40 + r * (0.40 / (rows - 1)), 3)
        grid[name] = [x, y]
    key = {
        "roster": roster[:n],
        "name_anchor_formation": "formation_1",
        "out_of_position_threshold": 0.07,
        "_README": ("positions are normalized image coords: x 0=left..1=right, "
                    "y 0=top..1=bottom. Set start/end to when each formation is "
                    "HELD (mm:ss). Edit names + positions to match your routine."),
        "formations": [
            {"name": "formation_1", "start": "0:00", "end": "0:15",
             "positions": grid},
            {"name": "formation_2", "start": "0:20", "end": "0:35",
             "positions": grid},
        ],
    }
    with open(path, "w") as fh:
        json.dump(key, fh, indent=2)


# ----------------------------------------------------------------------------
# Position helpers
# ----------------------------------------------------------------------------
def normalized_positions(frames_full, W: int, H: int):
    """Per processed-frame dict {stable_id: (x,y) normalized centroid}."""
    out = []
    for ft in frames_full:
        out.append({sid: (det.centroid()[0] / W, det.centroid()[1] / H)
                    for sid, det in ft.items()})
    return out


def avg_positions_in_window(norm_pos, start_f: int, end_f: int
                            ) -> Dict[int, Tuple[float, float]]:
    """Average normalized position of each stable id over a frame window."""
    acc: Dict[int, List[Tuple[float, float]]] = {}
    for fi in range(max(0, start_f), min(len(norm_pos), end_f)):
        for sid, p in norm_pos[fi].items():
            acc.setdefault(sid, []).append(p)
    return {sid: (float(np.mean([p[0] for p in pts])),
                  float(np.mean([p[1] for p in pts])))
            for sid, pts in acc.items() if pts}


# ----------------------------------------------------------------------------
# Naming: attach roster names to stable ids via the anchor formation
# ----------------------------------------------------------------------------
def assign_names(key: FormationKey, norm_pos) -> Dict[int, str]:
    """Hungarian-match stable ids to chart names using the anchor formation's
    held positions. Returns {stable_id: name}."""
    f = key.anchor_formation
    actual = avg_positions_in_window(norm_pos, f.start_f, f.end_f)
    ids = sorted(actual.keys())
    names = [n for n in key.roster if n in f.positions]
    if not ids or not names:
        return {}
    cost = np.zeros((len(ids), len(names)))
    for i, sid in enumerate(ids):
        ax, ay = actual[sid]
        for j, nm in enumerate(names):
            cx, cy = f.positions[nm]
            cost[i, j] = np.hypot(ax - cx, ay - cy)
    ri, ci = linear_sum_assignment(cost)
    return {ids[i]: names[j] for i, j in zip(ri, ci)}


# ----------------------------------------------------------------------------
# Grading: who was out of position in each formation
# ----------------------------------------------------------------------------
@dataclass
class PositionEvent:
    formation: str
    dancer: str
    start_f: int
    end_f: int
    error: float            # normalized distance from charted spot
    status: str             # 'out_of_position' | 'missing' | 'ok'
    actual: Optional[Tuple[float, float]]
    intended: Tuple[float, float]


def grade_positions(key: FormationKey, norm_pos, name_map: Dict[int, str]
                    ) -> List[PositionEvent]:
    id_of_name = {nm: sid for sid, nm in name_map.items()}
    events: List[PositionEvent] = []
    for f in key.formations:
        actual = avg_positions_in_window(norm_pos, f.start_f, f.end_f)
        for nm, intended in f.positions.items():
            sid = id_of_name.get(nm)
            if sid is None or sid not in actual:
                events.append(PositionEvent(f.name, nm, f.start_f, f.end_f,
                                            float("nan"), "missing", None, intended))
                continue
            ax, ay = actual[sid]
            err = float(np.hypot(ax - intended[0], ay - intended[1]))
            status = "out_of_position" if err > key.threshold else "ok"
            events.append(PositionEvent(f.name, nm, f.start_f, f.end_f,
                                        err, status, (ax, ay), intended))
    return events
