"""
Match a front-camera dance video to a top-down formation chart WITHOUT manual
homography.

The chart is a bird's-eye map; the video is a front view, so depth is squished.
But within one held formation the *arrangement* is the same, so we fit the best
affine transform (which absorbs translation/scale/rotation/shear/mirror) from the
video's on-floor positions onto each chart formation, via ICP (alternate:
assign -> refit). The chart formation with the lowest residual is the one being
performed, and the assignment tells us which tracked dancer is which named dancer.

Outputs:
  * a global stable_id -> name map (majority vote across all matched holds),
  * per-hold formation label + which named dancers were out of position.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from formation_consistency import segment_holds

L_ANKLE, R_ANKLE = 27, 28


def foot_point(det, W: int, H: int) -> np.ndarray:
    """On-floor contact point (mid-ankle if visible, else bottom-center of box),
    normalized to [0,1]. Feet sit on the floor plane, so they map best to a map."""
    kp, vis = det.keypoints, det.visibility
    pts = [kp[j] for j in (L_ANKLE, R_ANKLE) if vis[j] > 0.3]
    if pts:
        p = np.mean(pts, axis=0)
    else:
        x1, y1, x2, y2 = det.bbox
        p = np.array([(x1 + x2) / 2, y2])
    return np.array([p[0] / W, p[1] / H])


def hold_positions(frames_full, holds, W: int, H: int, present_min: float = 0.3):
    out = []
    for s, e in holds:
        acc = defaultdict(list)
        for fi in range(s, e):
            for sid, det in frames_full[fi].items():
                acc[sid].append(foot_point(det, W, H))
        dur = max(1, e - s)
        out.append({sid: np.median(v, axis=0) for sid, v in acc.items()
                    if len(v) >= present_min * dur})
    return out


def _bbox_center_positions(frames_full, W: int, H: int):
    """Per-frame {sid: bbox-center normalized}. Works for boxes-only tracks too
    (unlike mid-hip, which is absent when there are no pose keypoints)."""
    out = []
    for ft in frames_full:
        d = {}
        for sid, det in ft.items():
            x1, y1, x2, y2 = det.bbox
            d[sid] = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)
        out.append(d)
    return out


def _apply(P, A):
    return np.hstack([P, np.ones((len(P), 1))]) @ A


def _fit_affine(P, Q):
    X = np.hstack([P, np.ones((len(P), 1))])
    A, *_ = np.linalg.lstsq(X, Q, rcond=None)
    return A  # (3,2)


def _fit_similarity(P, Q):
    """Best translation + rotation(+reflection) + UNIFORM scale mapping P->Q
    (Procrustes). Rigid enough that a formation's shape must actually match,
    unlike affine which can shear any blob onto any layout."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    R = (Vt.T @ U.T)                       # 2x2 orthogonal (allows reflection)
    s = S.sum() / ((Pc ** 2).sum() + 1e-12)
    lin = s * R.T                          # row-vector convention: P @ lin
    t = Q.mean(0) - P.mean(0) @ lin
    return np.vstack([lin, t])             # (3,2)


def _init_affine(P, Q):
    """Coarse align by matching centroid + per-axis spread (allows mirror)."""
    pc, qc = P.mean(0), Q.mean(0)
    ps, qs = P.std(0) + 1e-6, Q.std(0) + 1e-6
    s = qs / ps
    A = np.array([[s[0], 0], [0, s[1]], [qc[0] - pc[0] * s[0], qc[1] - pc[1] * s[1]]])
    return A


def match_hold(pos: Dict[int, np.ndarray], chart: Dict[str, list],
               iters: int = 6) -> Optional[Tuple[float, Dict[int, str]]]:
    sids = list(pos)
    if len(sids) < 4:
        return None
    P = np.array([pos[s] for s in sids], float)
    names = list(chart)
    Q = np.array([chart[n] for n in names], float)
    best = None
    # try both mirror orientations of the init to avoid left/right lock-in
    for flip in (1.0, -1.0):
        Pf = P.copy(); Pf[:, 0] = P[:, 0].mean() + flip * (P[:, 0] - P[:, 0].mean())
        A = _init_affine(Pf, Q)
        for _ in range(iters):
            C = np.linalg.norm(_apply(Pf, A)[:, None, :] - Q[None, :, :], axis=2)
            r, c = linear_sum_assignment(C)
            A = _fit_similarity(Pf[r], Q[c])
        C = np.linalg.norm(_apply(Pf, A)[:, None, :] - Q[None, :, :], axis=2)
        r, c = linear_sum_assignment(C)
        cost = float(C[r, c].mean())
        if best is None or cost < best[0]:
            best = (cost, {sids[i]: names[j] for i, j in zip(r, c)})
    return best


def match_video(frames_full, key_formations: List[dict], fps: float, W: int, H: int):
    """Returns (name_map, hold_results). hold_results: list of dicts with the
    chosen formation, cost, and per-dancer assignment for each detected hold."""
    norm = _bbox_center_positions(frames_full, W, H)
    holds = segment_holds(norm, fps)
    hpos = hold_positions(frames_full, holds, W, H)

    charts = [(f["name"], {n: p for n, p in f["positions"].items()})
              for f in key_formations]
    hold_results = []
    votes = defaultdict(Counter)   # sid -> Counter(name)
    for (s, e), pos in zip(holds, hpos):
        best_form, best = None, None
        for fname, chart in charts:
            m = match_hold(pos, chart)
            if m is None:
                continue
            cost, mapping = m
            if best is None or cost < best[0]:
                best, best_form = (cost, mapping), fname
        if best is None:
            continue
        cost, mapping = best
        for sid, name in mapping.items():
            votes[sid][name] += 1
        hold_results.append({"start": s, "end": e, "formation": best_form,
                             "cost": cost, "mapping": mapping})

    # global identity = majority vote, resolving collisions greedily by count
    name_map = {}
    taken = {}
    for sid, ctr in sorted(votes.items(), key=lambda kv: -sum(kv[1].values())):
        for name, _ in ctr.most_common():
            if name not in taken:
                name_map[sid] = name; taken[name] = sid; break
    return name_map, hold_results, holds


def match_video_ordered(frames_full, key_formations: List[dict], fps: float,
                        W: int, H: int):
    """Like match_video but enforces that holds map to formations in the order
    the routine is performed (non-decreasing formation index). This resolves the
    ambiguity where many formations fit a squished, partial front-camera view
    almost equally well."""
    norm = _bbox_center_positions(frames_full, W, H)
    holds = segment_holds(norm, fps)
    hpos = hold_positions(frames_full, holds, W, H)
    charts = [(f["name"], f["positions"]) for f in key_formations]

    # cost[h][f] and mapping[h][f] for every hold/formation pair
    Hn, Fn = len(hpos), len(charts)
    BIG = 1e9
    cost = np.full((Hn, Fn), BIG)
    mp = [[None] * Fn for _ in range(Hn)]
    for h, pos in enumerate(hpos):
        for f, (_, chart) in enumerate(charts):
            m = match_hold(pos, chart)
            if m is not None:
                cost[h][f], mp[h][f] = m
    # DP: min-cost assignment with non-decreasing formation index
    dp = np.full((Hn, Fn), BIG); back = [[-1] * Fn for _ in range(Hn)]
    for f in range(Fn):
        dp[0][f] = cost[0][f]
    for h in range(1, Hn):
        best_prev = BIG; best_f = -1
        for f in range(Fn):
            if dp[h - 1][f] < best_prev:
                best_prev, best_f = dp[h - 1][f], f
            dp[h][f] = cost[h][f] + best_prev
            back[h][f] = best_f
    # backtrack
    f = int(np.argmin(dp[Hn - 1]))
    chosen = [0] * Hn
    for h in range(Hn - 1, -1, -1):
        chosen[h] = f
        f = back[h][f] if h > 0 else f
    votes = defaultdict(Counter); hold_results = []
    for h, (s, e) in enumerate(holds):
        f = chosen[h]
        mapping = mp[h][f] or {}
        for sid, name in mapping.items():
            votes[sid][name] += 1
        hold_results.append({"start": s, "end": e, "formation": charts[f][0],
                             "cost": float(cost[h][f]), "mapping": mapping})
    name_map, taken = {}, {}
    for sid, ctr in sorted(votes.items(), key=lambda kv: -sum(kv[1].values())):
        for name, _ in ctr.most_common():
            if name not in taken:
                name_map[sid] = name; taken[name] = sid; break
    return name_map, hold_results, holds
