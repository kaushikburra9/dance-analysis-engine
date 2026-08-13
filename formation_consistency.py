"""
SELF-REFERENTIAL FORMATION CONSISTENCY (no chart needed)
========================================================
On a single video, flag dancers who break the GROUP's own formation:
  * isolated / out-of-line   — far from neighbors or off the formation's line;
  * late to the formation    — settles into the held shape later than the group.

This uses the group's geometry as the reference, so it needs no answer key. It is
POSITION-based, so unlike fine beat-timing it survives the keypoint noise floor.
Formations are auto-segmented from the group's collective motion (HOLD vs MOVE).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FormationEvent:
    formation_idx: int
    start_f: int
    end_f: int
    dancer: int                 # stable id
    kind: str                   # 'out_of_line' | 'late_to_formation'
    severity: float             # normalized distance / seconds late
    detail: str


# ----------------------------------------------------------------------------
def group_speed(norm_pos, fps: float) -> np.ndarray:
    n = len(norm_pos)
    gs = np.full(n, np.nan)
    prev = {}
    for fi in range(n):
        ds = [np.hypot(p[0] - prev[s][0], p[1] - prev[s][1])
              for s, p in norm_pos[fi].items() if s in prev]
        gs[fi] = np.mean(ds) if ds else np.nan
        prev = dict(norm_pos[fi])
    k = max(1, int(fps * 0.5) | 1)
    return np.convolve(np.nan_to_num(gs), np.ones(k) / k, mode="same")


def segment_holds(norm_pos, fps: float, hold_pct: float = 40,
                  min_hold_s: float = 1.0) -> List[Tuple[int, int]]:
    sm = group_speed(norm_pos, fps)
    thr = np.nanpercentile(sm, hold_pct)
    hold = sm < thr
    holds, i, n = [], 0, len(norm_pos)
    while i < n:
        j = i
        while j < n and hold[j] == hold[i]:
            j += 1
        if hold[i] and (j - i) >= int(fps * min_hold_s):
            holds.append((i, j))
        i = j
    return holds


def _hold_positions(norm_pos, s: int, e: int):
    """median position + presence fraction per dancer over a hold."""
    acc: Dict[int, List[Tuple[float, float]]] = {}
    for fi in range(s, e):
        for sid, p in norm_pos[fi].items():
            acc.setdefault(sid, []).append(p)
    dur = max(1, e - s)
    out = {}
    for sid, pts in acc.items():
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        out[sid] = ((float(np.median(xs)), float(np.median(ys))), len(pts) / dur)
    return out


def _line_residuals(pos: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
    """Distance of each dancer from the best-fit line through the group (PCA).
    Meaningful when the formation is line-like; small for compact blobs."""
    ids = list(pos)
    if len(ids) < 3:
        return {i: 0.0 for i in ids}
    P = np.array([pos[i] for i in ids])
    c = P.mean(axis=0)
    u, s, vt = np.linalg.svd(P - c)
    normal = vt[1]                       # direction of least variance
    res = np.abs((P - c) @ normal)
    return {ids[k]: float(res[k]) for k in range(len(ids))}


def detect_events(norm_pos, fps: float,
                  iso_k: float = 2.2, iso_abs: float = 0.12,
                  line_k: float = 2.5, line_abs: float = 0.08,
                  present_min: float = 0.5, formation_margin_s: float = 0.6,
                  min_hold_s: float = 1.0, structural_frac: float = 0.6
                  ) -> Tuple[List[FormationEvent], List[Tuple[int, int]]]:
    """`formation_margin_s` is the timing margin of error: a dancer is only
    called EARLY/LATE if they reach the new formation more than this many
    seconds before/after the group. Small differences read as on-time."""
    holds = segment_holds(norm_pos, fps, min_hold_s=min_hold_s)
    events: List[FormationEvent] = []
    holds_present: Dict[int, int] = {}    # how many holds each dancer appears in

    for fidx, (s, e) in enumerate(holds):
        hp = _hold_positions(norm_pos, s, e)
        present = {sid: p for sid, (p, frac) in hp.items() if frac >= present_min}
        if len(present) < 3:
            continue
        ids = list(present)
        for a in ids:
            holds_present[a] = holds_present.get(a, 0) + 1

        # --- spacing: nearest-neighbour isolation ---
        nn = {}
        for a in ids:
            nn[a] = min(np.hypot(present[a][0] - present[b][0],
                                 present[a][1] - present[b][1])
                        for b in ids if b != a)
        med_nn = float(np.median(list(nn.values())))
        for a in ids:
            if nn[a] > max(iso_abs, iso_k * med_nn):
                events.append(FormationEvent(
                    fidx, s, e, a, "out_of_line", nn[a],
                    f"isolated: nearest dancer {nn[a]:.2f} vs typical {med_nn:.2f}"))

        # --- alignment: off the formation's line ---
        res = _line_residuals(present)
        med_res = float(np.median(list(res.values()))) + 1e-6
        for a in ids:
            already = any(ev.dancer == a and ev.formation_idx == fidx and
                          ev.kind == "out_of_line" for ev in events)
            if not already and res[a] > max(line_abs, line_k * med_res):
                events.append(FormationEvent(
                    fidx, s, e, a, "out_of_line", res[a],
                    f"off the line by {res[a]:.2f}"))

        # --- late to formation: settle time vs group ---
        arrivals = {}
        for a in ids:
            tx, ty = present[a]
            arr = None
            for fi in range(s, e):
                p = norm_pos[fi].get(a)
                if p and np.hypot(p[0] - tx, p[1] - ty) < 0.05:
                    arr = fi; break
            if arr is not None:
                arrivals[a] = arr
        if len(arrivals) >= 3:
            grp = np.median(list(arrivals.values()))
            for a, arr in arrivals.items():
                d = (arr - grp) / fps          # +late / -early vs the group
                if d > formation_margin_s:
                    events.append(FormationEvent(
                        fidx, s, e, a, "late_to_formation", d,
                        f"reached formation {d:.1f}s after the group"))
                elif -d > formation_margin_s:
                    events.append(FormationEvent(
                        fidx, s, e, a, "early_to_formation", -d,
                        f"reached formation {-d:.1f}s before the group"))

    # --- reclassify persistent isolates as structural (likely not a dancer) ---
    ool = {}
    for ev in events:
        if ev.kind == "out_of_line":
            ool.setdefault(ev.dancer, []).append(ev)
    # Structural = a slot that's isolated either in MOST holds it appears in, or
    # consistently FAR from everyone (a side/stationary figure or an over-counted
    # roster slot) - as opposed to a real dancer who's occasionally out of line.
    structural = {
        sid for sid, evs in ool.items()
        if len(evs) >= 3 and (
            len(evs) >= structural_frac * holds_present.get(sid, 1)
            or float(np.mean([e.severity for e in evs])) >= 0.30)
    }
    out: List[FormationEvent] = [ev for ev in events
                                 if not (ev.kind == "out_of_line" and ev.dancer in structural)]
    for sid in structural:
        evs = ool[sid]
        out.append(FormationEvent(
            evs[0].formation_idx, evs[0].start_f, evs[-1].end_f, sid,
            "structural_outlier", float(np.mean([e.severity for e in evs])),
            f"isolated in {len(evs)}/{holds_present.get(sid,0)} formations "
            f"(~{np.mean([e.severity for e in evs]):.2f} away); likely a side "
            f"figure, not a dancer"))
    return out, holds
