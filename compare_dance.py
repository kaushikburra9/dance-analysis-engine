"""
Compare a STUDENT dancer to a REFERENCE dancer doing the SAME choreography.

Pipeline (robust to different people / cameras / body size / length):
  1. extract each dancer's skeleton per frame (MediaPipe, single person);
  2. reduce to 8 body-relative LIMB ANGLES (elbows, shoulders, hips, knees);
  3. DTW-align the two angle sequences (handles tempo / start / length);
  4. flag ONLY coarse issues: limbs off beyond a MARGIN (default 30 deg),
     sustained (debounced), plus overall ahead/behind timing.

Optional --render writes a side-by-side video: both skeletons in synced time,
the student's off-limbs lit red, with a timing + issue banner.
"""
import argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL = "pose_landmarker.task"
LIMBS = [
    ("L elbow", 11, 13, 15), ("R elbow", 12, 14, 16),
    ("L shoulder", 13, 11, 23), ("R shoulder", 14, 12, 24),
    ("L hip", 11, 23, 25), ("R hip", 12, 24, 26),
    ("L knee", 23, 25, 27), ("R knee", 24, 26, 28),
]
# bones to recolor red when a given limb is flagged
LIMB_BONES = [
    [(11, 13), (13, 15)], [(12, 14), (14, 16)],
    [(11, 13), (11, 23)], [(12, 14), (12, 24)],
    [(11, 23), (23, 25)], [(12, 24), (24, 26)],
    [(23, 25), (25, 27)], [(24, 26), (26, 28)],
]
EDGES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15),
         (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
         (27, 31), (28, 32), (0, 11), (0, 12)]


def _side(name):
    return "right" if name[0] == "R" else "left"


def correction(name, s_ang, r_ang, terse=False):
    """Plain-language fix from the angle gap. Direction is exact; magnitude is
    deliberately coarse (slightly / more / a lot) - no fake-precise degrees."""
    d = s_ang - r_ang
    mag = "a lot" if abs(d) > 65 else "more" if abs(d) > 42 else "slightly"
    side, part = _side(name), name.split()[1]
    if part == "elbow":
        verb = "Bend" if d > 0 else "Straighten"; what = f"{side} elbow" if d > 0 else f"{side} arm"
    elif part == "shoulder":
        verb = "Lower" if d > 0 else "Raise"; what = f"{side} arm"
    elif part == "knee":
        verb = "Bend" if d > 0 else "Straighten"; what = f"{side} knee" if d > 0 else f"{side} leg"
    else:  # hip
        verb = "Sink lower on" if d > 0 else "Stand taller on"; what = f"{side} side"
    return f"{verb} {what}" if terse else f"{verb} your {what} {mag}"


def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def extract_pose(video):
    base = mp_python.BaseOptions(model_asset_path=MODEL)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5)
    lm = mp_vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    angs, viss, lms, fi = [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        res = lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(fi / fps * 1000))
        row = np.full(len(LIMBS), np.nan); vis = np.zeros(len(LIMBS))
        lm_norm = np.full((33, 2), np.nan)
        if res.pose_landmarks:
            L = res.pose_landmarks[0]
            lm_norm = np.array([[p.x, p.y] for p in L])
            P = lm_norm * [W, H]
            V = np.array([getattr(p, "visibility", 1.0) for p in L])
            for k, (_, a, b, c) in enumerate(LIMBS):
                if min(V[a], V[b], V[c]) > 0.5:
                    row[k] = angle(P[a], P[b], P[c]); vis[k] = 1
        angs.append(row); viss.append(vis); lms.append(lm_norm); fi += 1
    cap.release(); lm.close()
    return {"ang": np.array(angs), "vis": np.array(viss),
            "lm": np.array(lms), "fps": fps}


def _fill(X):
    Y = X.copy()
    for j in range(Y.shape[1]):
        m = np.nanmean(Y[:, j]); Y[np.isnan(Y[:, j]), j] = m if not np.isnan(m) else 0
    return Y


def align_and_flag(S, R, margin, min_seconds):
    import librosa
    D, wp = librosa.sequence.dtw(X=_fill(S["ang"]).T, Y=_fill(R["ang"]).T,
                                 metric="euclidean")
    wp = wp[::-1]
    sfps = S["fps"]
    min_frames = max(1, int(min_seconds * sfps))
    events = []

    def close(k, name, run, step):
        sa = np.nanmedian([S["ang"][wp[t][0], k] for t in range(run, step)])
        ra = np.nanmedian([R["ang"][wp[t][1], k] for t in range(run, step)])
        events.append((wp[run][0] / sfps, wp[step - 1][0] / sfps, name,
                       correction(name, sa, ra), float(abs(sa - ra))))

    for k, (name, *_) in enumerate(LIMBS):
        run = None
        for step, (si, ri) in enumerate(wp):
            a, b = S["ang"][si, k], R["ang"][ri, k]
            off = (not np.isnan(a)) and (not np.isnan(b)) and abs(a - b) > margin
            if off and run is None:
                run = step
            elif not off and run is not None:
                if step - run >= min_frames:
                    close(k, name, run, step)
                run = None
        if run is not None and len(wp) - run >= min_frames:
            close(k, name, run, len(wp))
    si = np.array([p[0] for p in wp]) / max(1, len(S["ang"]) - 1)
    ri = np.array([p[1] for p in wp]) / max(1, len(R["ang"]) - 1)
    return wp, events, float(np.mean(si - ri))


def _load_small(video, height=460):
    cap = cv2.VideoCapture(video); frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        w = int(fr.shape[1] * height / fr.shape[0])
        frames.append(cv2.resize(fr, (w, height)))
    cap.release(); return frames


def draw_skel(img, lm_norm, off_limbs):
    if np.isnan(lm_norm).all():
        return
    h, w = img.shape[:2]
    P = [(int(x * w), int(y * h)) for x, y in lm_norm]
    for a, b in EDGES:
        cv2.line(img, P[a], P[b], (0, 200, 0), 2)
    for k in off_limbs:
        for a, b in LIMB_BONES[k]:
            cv2.line(img, P[a], P[b], (0, 0, 255), 5)


def _mux(silent, audio_src, out):
    import os, shutil, subprocess
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-y", "-i", silent, "-i", audio_src,
                            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0",
                            "-map", "1:a:0?", "-shortest", out, "-loglevel", "error"],
                           check=True)
            os.remove(silent); return
        except Exception as e:
            print(f"  (audio mux failed: {e})")
    os.replace(silent, out)


def group_steps(events):
    """Merge overlapping-in-time limb flags into ordered correction steps."""
    steps, cur = [], []
    for e in sorted(events):
        if cur and e[0] <= max(g[1] for g in cur) + 0.4:
            cur.append(e)
        else:
            if cur:
                steps.append(cur)
            cur = [e]
    if cur:
        steps.append(cur)
    out = []
    for grp in steps:
        t0 = min(g[0] for g in grp); t1 = max(g[1] for g in grp)
        cues = sorted({g[3] for g in grp},
                      key=lambda c: -max(g[4] for g in grp if g[3] == c))
        out.append({"t0": t0, "t1": t1, "mid": (t0 + t1) / 2, "cues": cues})
    return out


def _ref2stu(wp):
    from collections import defaultdict
    r2s = defaultdict(list)
    for si, ri in wp:
        r2s[ri].append(si)
    return {ri: int(np.median(v)) for ri, v in r2s.items()}


def _off_limbs(S, R, si, rr, margin):
    return [k for k in range(len(LIMBS))
            if not np.isnan(S["ang"][si, k]) and not np.isnan(R["ang"][rr, k])
            and abs(S["ang"][si, k] - R["ang"][rr, k]) > margin]


def compose(Sf, Rf, S, R, si, rr, margin, Ns, Nr, big_cue=None):
    """One side-by-side frame: skeletons, cue banner, timing bar."""
    band = 100
    h, aw = Sf[0].shape[0], Sf[0].shape[1]
    Wt = aw + Rf[0].shape[1] + 20
    off = _off_limbs(S, R, si, rr, margin)
    a, b = Sf[si].copy(), Rf[rr].copy()
    draw_skel(a, S["lm"][si], off); draw_skel(b, R["lm"][rr], [])
    c = np.full((h + band, Wt, 3), 20, np.uint8)
    c[band:band + h, :aw] = a; c[band:band + h, aw + 20:] = b
    cv2.putText(c, "STUDENT", (10, h + band - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(c, "REFERENCE", (aw + 30, h + band - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    if off:
        cues = [correction(LIMBS[k][0], S["ang"][si, k], R["ang"][rr, k], terse=True)
                for k in off[:2]]
        cv2.putText(c, "  |  ".join(cues), (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    else:
        cv2.putText(c, "MATCH", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 0), 2)
    # timing bar: how far ahead/behind the student is at this instant
    expected = rr * (Ns - 1) / max(1, Nr - 1)
    lag_s = (si - expected) / S["fps"]
    cx, halfw, y = Wt // 2, int(Wt * 0.22), 62
    cv2.line(c, (cx - halfw, y), (cx + halfw, y), (120, 120, 120), 2)
    cv2.line(c, (cx, y - 8), (cx, y + 8), (180, 180, 180), 1)
    mx = int(cx + np.clip(lag_s / 1.0, -1, 1) * halfw)
    tcol = (0, 220, 0) if abs(lag_s) < 0.15 else (0, 165, 255)
    cv2.circle(c, (mx, y), 7, tcol, -1)
    tlabel = "in sync" if abs(lag_s) < 0.15 else \
             (f"ahead {abs(lag_s):.1f}s" if lag_s > 0 else f"behind {abs(lag_s):.1f}s")
    cv2.putText(c, "BEHIND", (cx - halfw - 70, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (150, 150, 150), 1)
    cv2.putText(c, "AHEAD", (cx + halfw + 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (150, 150, 150), 1)
    cv2.putText(c, tlabel, (cx - 40, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tcol, 1)
    if big_cue:
        ov = c.copy(); cv2.rectangle(ov, (0, band), (Wt, band + h), (0, 0, 0), -1)
        c = cv2.addWeighted(ov, 0.55, c, 0.45, 0)
        yy = band + 60
        cv2.putText(c, "FIX:", (30, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 255), 3)
        for line in big_cue:
            yy += 46
            cv2.putText(c, "- " + line, (30, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (255, 255, 255), 2)
    return c


def render(student, reference, S, R, wp, margin, out_path):
    """Flowing side-by-side on the reference timeline (song synced), + timing bar."""
    Sf, Rf = _load_small(student), _load_small(reference)
    ref2stu = _ref2stu(wp); keys = sorted(ref2stu)
    Ns, Nr = len(S["ang"]), len(R["ang"])
    sample = compose(Sf, Rf, S, R, 0, 0, margin, Ns, Nr)
    silent = out_path + ".silent.mp4"
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), R["fps"],
                         (sample.shape[1], sample.shape[0]))
    for ri in range(len(Rf)):
        si = ref2stu.get(ri) or ref2stu[min(keys, key=lambda k: abs(k - ri))]
        vw.write(compose(Sf, Rf, S, R, min(si, len(Sf) - 1), min(ri, len(Rf) - 1),
                         margin, Ns, Nr))
    vw.release()
    _mux(silent, reference, out_path)


def render_coached(student, reference, S, R, wp, margin, steps, out_path,
                   freeze_sec=1.8):
    """Same, but pause on each correction step with a big FIX cue. Audio is padded
    with silence during each freeze so the song stays in sync afterward."""
    import os, shutil, subprocess
    Sf, Rf = _load_small(student), _load_small(reference)
    ref2stu = _ref2stu(wp); keys = sorted(ref2stu)
    Ns, Nr = len(S["ang"]), len(R["ang"])
    fps = R["fps"]; freeze_n = int(freeze_sec * fps)
    step_starts = {int(round(s["t0"] * fps)): s for s in steps}
    sample = compose(Sf, Rf, S, R, 0, 0, margin, Ns, Nr)
    silent = out_path + ".silent.mp4"
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (sample.shape[1], sample.shape[0]))
    freeze_times = []
    for ri in range(len(Rf)):
        si = ref2stu.get(ri) or ref2stu[min(keys, key=lambda k: abs(k - ri))]
        si, rr = min(si, len(Sf) - 1), min(ri, len(Rf) - 1)
        if ri in step_starts:                    # freeze + coach here
            freeze_times.append(ri / fps)
            frozen = compose(Sf, Rf, S, R, si, rr, margin, Ns, Nr,
                             big_cue=step_starts[ri]["cues"])
            for _ in range(freeze_n):
                vw.write(frozen)
        vw.write(compose(Sf, Rf, S, R, si, rr, margin, Ns, Nr))
    vw.release()
    # build audio: reference song with `freeze_sec` silence inserted at each freeze
    if shutil.which("ffmpeg") and freeze_times:
        bounds = [0.0] + freeze_times + [len(Rf) / fps]
        parts, labels, idx = [], [], 0
        for i in range(len(bounds) - 1):
            parts.append(f"[1:a]atrim={bounds[i]}:{bounds[i+1]},"
                         f"asetpts=PTS-STARTPTS[a{idx}]"); labels.append(f"[a{idx}]"); idx += 1
            if i < len(freeze_times):
                parts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{freeze_sec},"
                             f"asetpts=PTS-STARTPTS[a{idx}]"); labels.append(f"[a{idx}]"); idx += 1
        fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", silent, "-i", reference,
                            "-filter_complex", fc, "-map", "0:v:0", "-map", "[out]",
                            "-c:v", "copy", "-c:a", "aac", "-shortest", out_path,
                            "-loglevel", "error"], check=True)
            os.remove(silent); return
        except Exception as e:
            print(f"  (coached audio pad failed: {e}; leaving silent)")
    os.replace(silent, out_path)


def build_pdf(student, reference, S, R, wp, margin, steps, lag, state, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    Sf, Rf = _load_small(student, 360), _load_small(reference, 360)
    ref2stu = _ref2stu(wp); keys = sorted(ref2stu); Ns, Nr = len(S["ang"]), len(R["ang"])
    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.94, "Choreography Correction Sheet", ha="center",
                 fontsize=19, weight="bold")
        tline = (f"Timing: student is {state} the reference "
                 f"({lag*100:+.0f}% of the routine)")
        fig.text(0.5, 0.90, tline, ha="center", fontsize=11, color="#333")
        lines = [f"Margin: {margin:.0f} deg (smaller differences are ignored)", ""]
        if not steps:
            lines.append("No corrections - within margin the whole way through.")
        for i, s in enumerate(steps, 1):
            lines.append(f"Step {i}  ({s['t0']:.1f}-{s['t1']:.1f}s)")
            for cue in s["cues"]:
                lines.append(f"     - {cue}")
            lines.append("")
        fig.text(0.09, 0.85, "\n".join(lines), va="top", fontsize=11,
                 family="monospace")
        pdf.savefig(fig); plt.close(fig)
        for i, s in enumerate(steps, 1):
            ri = min(int(s["mid"] * R["fps"]), len(Rf) - 1)
            si = ref2stu.get(ri) or ref2stu[min(keys, key=lambda k: abs(k - ri))]
            img = compose(Sf, Rf, S, R, min(si, len(Sf) - 1), ri, margin, Ns, Nr)
            fig = plt.figure(figsize=(8.5, 7)); ax = fig.add_axes([0.05, 0.28, 0.9, 0.64])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
            ax.set_title(f"Step {i}  ({s['t0']:.1f}-{s['t1']:.1f}s)", fontsize=14,
                         weight="bold")
            fig.text(0.5, 0.22, "FIX:", ha="center", fontsize=13, weight="bold",
                     color="#c00")
            for j, cue in enumerate(s["cues"]):
                fig.text(0.5, 0.185 - j * 0.032, cue, ha="center", fontsize=11,
                         color="#c00")
            pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--min-seconds", type=float, default=0.4)
    ap.add_argument("--render", default=None, help="flowing side-by-side mp4 (+song)")
    ap.add_argument("--coach", default=None, help="freeze-and-coach mp4")
    ap.add_argument("--pdf", default=None, help="correction-sheet pdf")
    args = ap.parse_args()

    print("extracting student ...", flush=True); S = extract_pose(args.student)
    print("extracting reference ...", flush=True); R = extract_pose(args.reference)
    wp, events, lag = align_and_flag(S, R, args.margin, args.min_seconds)
    state = "in sync with" if abs(lag) < 0.05 else ("AHEAD of" if lag > 0 else "BEHIND")
    steps = group_steps(events)
    print(f"\n===== CHOREOGRAPHY COMPARISON (margin {args.margin:.0f} deg) =====")
    print(f"Timing: student is {state} the reference (avg {lag*100:+.0f}% of routine)")
    print("\nHOW TO FIX (in order):")
    if not steps:
        print("  Looking good - within margin the whole way through.")
    for i, s in enumerate(steps, 1):
        print(f"  Step {i}  ({s['t0']:.1f}-{s['t1']:.1f}s):  " + ";  ".join(s["cues"]))
    if abs(lag) >= 0.05:
        print(f"  Timing:  come in {'later' if lag > 0 else 'earlier'} "
              f"- you're {state.split()[0].lower()} the reference.")

    if args.render:
        print(f"rendering {args.render} ...", flush=True)
        render(args.student, args.reference, S, R, wp, args.margin, args.render)
        print(f"  wrote {args.render}")
    if args.coach:
        print(f"rendering coached {args.coach} ...", flush=True)
        render_coached(args.student, args.reference, S, R, wp, args.margin, steps,
                       args.coach)
        print(f"  wrote {args.coach}")
    if args.pdf:
        print(f"building {args.pdf} ...", flush=True)
        build_pdf(args.student, args.reference, S, R, wp, args.margin, steps, lag,
                  state, args.pdf)
        print(f"  wrote {args.pdf}")


if __name__ == "__main__":
    main()
