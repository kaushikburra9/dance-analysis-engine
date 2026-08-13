"""
Build a captain-facing PDF from the formation grading: an identity map, a
per-formation floor diagram (intended vs actual), and a per-dancer summary of
out-of-position / missing moments. Uses matplotlib's PDF backend (no new deps).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from formation import FormationKey, PositionEvent


def _fmt_time(frame_idx: int, fps: float) -> str:
    s = frame_idx / max(fps, 1e-9)
    return f"{int(s // 60)}:{int(s % 60):02d}"


def build_report(key: FormationKey, name_map: Dict[int, str],
                 events: List[PositionEvent], out_path: str,
                 fps: float, video: str) -> Dict:
    by_dancer = defaultdict(list)
    for e in events:
        by_dancer[e.dancer].append(e)
    flagged = [e for e in events if e.status == "out_of_position"]
    missing = [e for e in events if e.status == "missing"]

    with PdfPages(out_path) as pdf:
        # ---- page 1: summary + identity map ----
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.95, "Formation & Position Report", ha="center",
                 fontsize=20, weight="bold")
        fig.text(0.5, 0.915, f"{video}", ha="center", fontsize=10, color="gray")
        lines = [
            f"Formations graded: {len(key.formations)}",
            f"Roster: {len(key.roster)} dancers",
            f"Out-of-position flags: {len(flagged)}",
            f"Missing (not tracked in a formation): {len(missing)}",
            f"Threshold: {key.threshold:.2f} of frame width",
            "",
            "Identity map (tracked ID -> dancer):",
        ]
        for sid in sorted(name_map):
            lines.append(f"   D{sid}  ->  {name_map[sid]}")
        fig.text(0.1, 0.84, "\n".join(lines), va="top", fontsize=11,
                 family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # ---- per-formation floor diagram ----
        for f in key.formations:
            fig, ax = plt.subplots(figsize=(8.5, 8.5))
            ax.set_title(f"Formation '{f.name}'  "
                         f"({_fmt_time(f.start_f, fps)}-{_fmt_time(f.end_f, fps)})",
                         fontsize=14, weight="bold")
            fevents = [e for e in events if e.formation == f.name]
            for e in fevents:
                ix, iy = e.intended
                ax.scatter([ix], [iy], s=260, facecolors="none",
                           edgecolors="#888", linewidths=1.5, zorder=2)
                ax.annotate(e.dancer, (ix, iy), fontsize=7, ha="center",
                            va="center", color="#444")
                if e.actual is not None:
                    axp, ayp = e.actual
                    col = "#d62728" if e.status == "out_of_position" else "#2ca02c"
                    ax.scatter([axp], [ayp], s=70, color=col, zorder=3)
                    if e.status == "out_of_position":
                        ax.plot([ix, axp], [iy, ayp], color=col, lw=1.2, zorder=1)
                else:
                    ax.scatter([ix], [iy], marker="x", s=120, color="#ff7f0e",
                               zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(1, 0)  # image coords: y down
            ax.set_xlabel("stage left  ->  stage right\n"
                          "o intended   . on-spot   . off-spot(red)   x missing")
            ax.set_ylabel("upstage (back)  <-  downstage (front)")
            ax.set_aspect("equal"); ax.grid(alpha=0.25)
            pdf.savefig(fig); plt.close(fig)

        # ---- per-dancer summary ----
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.96, "Per-dancer summary", ha="center", fontsize=16,
                 weight="bold")
        y = 0.91
        for name in key.roster:
            evs = by_dancer.get(name, [])
            probs = [e for e in evs if e.status in ("out_of_position", "missing")]
            if not probs:
                txt = f"{name}: clean (in position all formations)"
            else:
                bits = []
                for e in probs:
                    when = f"{e.formation} @ {_fmt_time(e.start_f, fps)}"
                    if e.status == "missing":
                        bits.append(f"{when}: NOT TRACKED")
                    else:
                        bits.append(f"{when}: off by {e.error:.2f}")
                txt = f"{name}: " + "; ".join(bits)
            color = "#2ca02c" if not probs else "#d62728"
            fig.text(0.07, y, txt, fontsize=9, color=color, va="top", wrap=True)
            y -= 0.045
            if y < 0.06:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11)); fig.clf(); y = 0.93
        pdf.savefig(fig); plt.close(fig)

    return {"flagged": len(flagged), "missing": len(missing),
            "pages": 2 + len(key.formations)}


def build_self_report(events, holds, out_path: str, fps: float, video: str,
                      n_dancers: int, name_map: Dict[int, str] = None) -> Dict:
    """PDF for the self-referential (no-chart) formation-consistency findings."""
    name_map = name_map or {}
    def nm(sid): return name_map.get(sid, f"D{sid}")
    timing_kinds = ("late_to_formation", "early_to_formation")
    real = [e for e in events if e.kind in ("out_of_line",) + timing_kinds]
    structural = [e for e in events if e.kind == "structural_outlier"]
    by_dancer = defaultdict(list)
    for e in real:
        by_dancer[e.dancer].append(e)

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.95, "Formation Consistency Report", ha="center",
                 fontsize=20, weight="bold")
        fig.text(0.5, 0.915, f"{video}  (no answer-key; graded vs the group)",
                 ha="center", fontsize=10, color="gray")
        lines = [
            f"Held formations detected: {len(holds)}",
            f"Dancers graded: {n_dancers}",
            f"Out-of-line moments: {sum(1 for e in real if e.kind=='out_of_line')}",
            f"Late-to-formation moments: {sum(1 for e in real if e.kind=='late_to_formation')}",
            f"Early-to-formation moments: {sum(1 for e in real if e.kind=='early_to_formation')}",
            "",
            "Reviewed / excluded (likely not dancers):",
        ]
        for e in structural:
            lines.append(f"   {nm(e.dancer)}: {e.detail}")
        if not structural:
            lines.append("   (none)")
        lines += ["", "How to read: 'out-of-line' = isolated/off the formation's",
                  "line vs neighbors. 'late-to-formation' = settled into the",
                  "held shape after the group. Position-based, so reliable;",
                  "fine beat-by-beat timing is NOT (needs higher-fps/GPU capture)."]
        fig.text(0.08, 0.86, "\n".join(lines), va="top", fontsize=11,
                 family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # per-dancer summary
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.96, "Per-dancer findings", ha="center", fontsize=16,
                 weight="bold")
        y = 0.91
        flagged_ids = sorted(by_dancer.keys())
        if not flagged_ids:
            fig.text(0.5, 0.5, "No dancers flagged - clean run.", ha="center",
                     fontsize=13, color="#2ca02c")
        for sid in flagged_ids:
            evs = sorted(by_dancer[sid], key=lambda e: e.start_f)
            bits = []
            for e in evs:
                t = f"{int(e.start_f/fps//60)}:{int(e.start_f/fps%60):02d}"
                if e.kind == "out_of_line":
                    tag = "out of line"
                elif e.kind == "late_to_formation":
                    tag = f"late {e.severity:.1f}s"
                else:
                    tag = f"early {e.severity:.1f}s"
                bits.append(f"{t} ({tag})")
            txt = f"{nm(sid)}:  " + ",  ".join(bits)
            fig.text(0.07, y, txt, fontsize=10, color="#d62728", va="top",
                     wrap=True)
            y -= 0.05
            if y < 0.06:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11)); fig.clf(); y = 0.93
        pdf.savefig(fig); plt.close(fig)

    return {"real_events": len(real), "structural": len(structural)}
