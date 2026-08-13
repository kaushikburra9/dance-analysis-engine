"""
STEP 4 — annotated video. Draws, per frame:
  * a colored skeleton per dancer, labelled with their STABLE roster id;
  * near each dancer at beat moments, an EARLY / LATE / ON-TIME tag;
  * a persistent diagnostics banner (stable-id count, avg lifetime, ID-switch est).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import cv2

from tracking import POSE_EDGES, PoseDetection


def _palette(n: int) -> List[Tuple[int, int, int]]:
    """n visually distinct BGR colors."""
    cols = []
    for i in range(max(n, 1)):
        h = int(180 * i / max(n, 1))
        hsv = np.uint8([[[h, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        cols.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return cols


def per_beat_tags(frames6: List[Dict[int, np.ndarray]], beats: List[int],
                  win: int = 8) -> Dict[int, Dict[int, str]]:
    """For each beat, classify each present dancer EARLY/LATE/on-time vs the group.

    Mirrors the verified analyze_sync timing (group = median motion per frame,
    local motion-peak offset within +/-win of the beat) but keeps the per-beat
    result so we can annotate at the moment it happened.
    """
    from analyze_run import motion_signal
    ids = sorted({tid for ft in frames6 for tid in ft})
    if not ids:
        return {}
    sigs = {t: motion_signal(frames6, t) for t in ids}
    group = np.nanmedian(np.vstack([sigs[t] for t in ids]), axis=0)

    tags: Dict[int, Dict[int, str]] = {}
    for b in beats:
        lo, hi = max(0, b - win), min(len(group), b + win)
        gseg = group[lo:hi]
        if gseg.size == 0 or np.all(np.isnan(gseg)):
            continue
        g_peak = int(np.nanargmax(gseg))
        per: Dict[int, str] = {}
        for t in ids:
            seg = sigs[t][lo:hi]
            if seg.size == 0 or np.all(np.isnan(seg)):
                continue
            off = int(np.nanargmax(seg)) - g_peak
            per[t] = "EARLY" if off < -1 else "LATE" if off > 1 else "ON-TIME"
        tags[b] = per
    return tags


def render_annotated(video_path: str, frames_full: List[Dict[int, PoseDetection]],
                     frames6: List[Dict[int, np.ndarray]], beats: List[int],
                     out_path: str, fps: float, sample_every: int = 1,
                     diagnostics: Dict[str, float] = None,
                     tempo: float = 0.0, beat_source: str = "",
                     name_map: Dict[int, str] = None,
                     position_events: list = None, self_events: list = None,
                     show_timing: bool = False, tag_window: int = 6) -> None:
    """Annotate the video. By design the noisy per-beat EARLY/LATE tags are OFF
    (`show_timing=False`) — they flickered because fine timing is below the
    measurement noise floor. Instead we draw stable identities (real names if a
    key is supplied) and, during formations, a steady OFF-SPOT callout for
    dancers the position grading flagged."""
    n = len(frames_full)
    stable_ids = sorted({tid for ft in frames_full for tid in ft})
    colors = {tid: c for tid, c in zip(stable_ids, _palette(len(stable_ids)))}
    beat_set = set(beats)
    name_map = name_map or {}
    id_of_name = {nm: sid for sid, nm in name_map.items()}
    tags = per_beat_tags(frames6, beats, win=8) if show_timing else {}
    beats_sorted = sorted(beats)

    # map each processed frame -> {stable id: callout label}
    callout = [dict() for _ in range(n)]
    formation_at = [None] * n
    for e in (position_events or []):
        sid = id_of_name.get(e.dancer)
        for fi_ in range(max(0, e.start_f), min(n, e.end_f)):
            formation_at[fi_] = e.formation
            if sid is not None and e.status == "out_of_position":
                callout[fi_][sid] = "OFF SPOT"
    lbl = {"out_of_line": "OUT OF LINE", "late_to_formation": "LATE",
           "early_to_formation": "EARLY"}
    for e in (self_events or []):
        if e.kind not in lbl:
            continue
        for fi_ in range(max(0, e.start_f), min(n, e.end_f)):
            callout[fi_][e.dancer] = lbl[e.kind]

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps / max(1, sample_every)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         out_fps, (W, H))
    tag_color = {"EARLY": (255, 160, 0), "LATE": (0, 0, 255),
                 "ON-TIME": (0, 220, 0)}

    fi = 0
    proc = 0
    while proc < n:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % sample_every != 0:
            fi += 1
            continue
        ftracks = frames_full[proc]
        is_beat_flash = proc in beat_set

        nearest_b, near_d = None, 10 ** 9
        if show_timing:
            for b in beats_sorted:
                d = abs(b - proc)
                if d < near_d:
                    near_d, nearest_b = d, b
                if b > proc + tag_window:
                    break

        for tid, det in ftracks.items():
            col = colors.get(tid, (200, 200, 200))
            kp, vis = det.keypoints, det.visibility
            for a, b in POSE_EDGES:
                if a < len(kp) and b < len(kp) and vis[a] > 0.3 and vis[b] > 0.3:
                    cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])), col, 2)
            for j in range(len(kp)):
                if vis[j] > 0.3:
                    cv2.circle(frame, (int(kp[j][0]), int(kp[j][1])), 2, col, -1)
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            label = name_map.get(tid, f"D{tid}")
            cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            # steady callout (held for the whole formation — not flickering)
            if tid in callout[proc]:
                cv2.putText(frame, callout[proc][tid], (x1, min(H - 4, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                            cv2.LINE_AA)
            elif show_timing and nearest_b in tags and tid in tags[nearest_b]:
                tg = tags[nearest_b][tid]
                cv2.putText(frame, tg, (x1, min(H - 4, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color[tg], 2,
                            cv2.LINE_AA)

        _banner(frame, W, proc, n, len(stable_ids), diagnostics, tempo,
                beat_source, is_beat_flash, len(ftracks))
        if formation_at[proc]:
            cv2.putText(frame, f"FORMATION: {formation_at[proc]}", (8, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        if is_beat_flash:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 255, 255), 4)

        vw.write(frame)
        proc += 1
        fi += 1

    cap.release()
    vw.release()


def _banner(frame, W, proc, n, n_ids, diag, tempo, src, is_beat, n_present):
    cv2.rectangle(frame, (0, 0), (W, 54), (0, 0, 0), -1)
    line1 = (f"frame {proc}/{n}  |  stable dancers: {n_ids}  |  "
             f"visible now: {n_present}  |  tempo ~{tempo:.0f}bpm ({src})")
    cv2.putText(frame, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    if diag:
        line2 = (f"avg track life {diag.get('avg_lifetime_frames',0):.0f}f  |  "
                 f"raw->stable {diag.get('raw_fragments',0)}->{diag.get('stable_ids',0)}  |  "
                 f"ID-switch est {diag.get('id_switch_per_min',0):.1f}/min")
        cv2.putText(frame, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 220, 255), 1, cv2.LINE_AA)
    if is_beat:
        cv2.putText(frame, "BEAT", (W - 70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
