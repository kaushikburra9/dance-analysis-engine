"""
DANCE TEAM APP — PHASE 0 VALIDATION PIPELINE
=============================================
Goal of this script: prove the motion-capture engine works on YOUR real dance
footage before building any app. It does NOT need to be pretty. It answers ONE
question: "Can a computer reliably track each dancer, time them against the beat,
and place them on the floor well enough that a captain would trust it?"

WHAT IT DOES
------------
1. Reads a practice-run video.
2. Detects + tracks every dancer per frame (pose skeletons with stable IDs).
3. Pulls the music's beat grid from the audio.
4. Measures SYNC: who moves ahead/behind the group, by how much.
5. Projects dancers onto a top-down FLOOR map (after you tap the 4 floor corners).
6. Writes an annotated video + a JSON report you can eyeball.

This uses MediaPipe (free, runs on CPU) so you can run it on a normal laptop.
For a real product you'd swap in RTMPose+ByteTrack on a GPU, but the LOGIC you
are validating — the sync math and floor geometry — is identical. Prove it here
first.

HOW TO RUN (on your own computer, NOT in chat)
----------------------------------------------
    pip install mediapipe opencv-python numpy librosa scipy
    python analyze_run.py --video my_team_run.mp4

Then open annotated_output.mp4 and report.json and ask yourself the only
question that matters: "Would this have saved our captains an hour?"
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

# MediaPipe is the free, CPU-friendly capture layer for Phase 0.
# Newer MediaPipe uses the "Tasks" API and needs a model file downloaded once:
#   wget -O pose_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "pose_landmarker.task"  # download once (see README)


# ----------------------------------------------------------------------------
# STEP 1 — CAPTURE: detect + track dancers frame by frame
# ----------------------------------------------------------------------------
# NOTE ON THE HARD PROBLEM: MediaPipe's built-in solution tracks ONE primary
# person well. For a 16-dancer formation you need true multi-person tracking.
# In Phase 0 we use a lightweight multi-person approach: detect all pose
# regions, then keep IDs stable across frames using simple nearest-centroid
# matching. This is deliberately simple so you can SEE where it breaks
# (occlusion / crossing dancers) — which is exactly the risk you're testing.

class SimpleTracker:
    """Nearest-centroid ID tracker. Crude on purpose — it reveals ID-swap risk."""
    def __init__(self, max_dist=120, max_missing=15):
        self.next_id = 0
        self.tracks = {}            # id -> {'centroid':(x,y), 'missing':int}
        self.max_dist = max_dist
        self.max_missing = max_missing

    def update(self, centroids):
        # Mark all existing as unmatched this frame
        assigned = {}
        used = set()
        for cid, c in centroids.items():
            best_id, best_d = None, self.max_dist
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                d = np.hypot(c[0] - t['centroid'][0], c[1] - t['centroid'][1])
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {'centroid': c, 'missing': 0}
            self.tracks[best_id]['centroid'] = c
            self.tracks[best_id]['missing'] = 0
            used.add(best_id)
            assigned[cid] = best_id
        # Age out stale tracks
        for tid in list(self.tracks.keys()):
            if tid not in used:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]
        return assigned


def extract_poses(video_path, max_frames=None, sample_every=1, max_dancers=20):
    """
    Returns: list of frames; each frame is a dict {track_id: keypoints}.
    keypoints = np.array of (x,y) in pixel coords for major joints.
    Also returns fps and frame size.

    Uses MediaPipe's PoseLandmarker (Tasks API) with num_poses>1 for true
    multi-person detection, then keeps IDs stable across frames with the
    SimpleTracker. The SimpleTracker is intentionally crude so you can SEE
    where IDs swap when dancers cross — that is the #1 risk you're validating.
    Production swap: RTMPose + ByteTrack on GPU + the formation answer-key to
    anchor IDs. The downstream math is identical.

    Joint order returned per dancer (6 points):
      [L_wrist, R_wrist, L_ankle, R_ankle, L_hip, R_hip]
    """
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: missing {MODEL_PATH}. Download it once with:\n"
              "  wget -O pose_landmarker.task https://storage.googleapis.com/"
              "mediapipe-models/pose_landmarker/pose_landmarker_full/float16/"
              "latest/pose_landmarker_full.task")
        sys.exit(1)

    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=max_dancers,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    # MediaPipe landmark indices for the joints we track
    L_WRIST, R_WRIST = 15, 16
    L_ANKLE, R_ANKLE = 27, 28
    L_HIP, R_HIP = 23, 24
    IDX = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = SimpleTracker(max_dist=max(80, W // 12))
    frames = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % sample_every != 0:
            fi += 1
            continue
        if max_frames and len(frames) >= max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((fi / fps) * 1000)
        res = landmarker.detect_for_video(mp_img, ts)

        detections = {}
        if res.pose_landmarks:
            for pi, lms in enumerate(res.pose_landmarks):
                pts = np.array([[lms[i].x * W, lms[i].y * H] for i in IDX],
                               dtype=np.float32)
                cen = pts[4:6].mean(axis=0)  # mid-hip centroid
                detections[pi] = {'kp': pts, 'cen': (float(cen[0]), float(cen[1]))}

        cen_only = {k: v['cen'] for k, v in detections.items()}
        assigned = tracker.update(cen_only)
        frame_tracks = {assigned[t]: detections[t]['kp'] for t in assigned}
        frames.append(frame_tracks)
        fi += 1

    cap.release()
    landmarker.close()
    return frames, fps, (W, H)


# ----------------------------------------------------------------------------
# STEP 2 — BEAT GRID from the audio
# ----------------------------------------------------------------------------
def get_beats(video_path, fps, n_frames):
    """Real beat grid from the audio (see beats.py). Delegates to the robust
    ffmpeg->wav->librosa path; falls back to a fixed grid only if audio is truly
    absent. Returns (beat_video_frames, tempo) for backward compatibility."""
    from beats import get_beats as _get_beats
    frames, tempo, source = _get_beats(video_path, fps, n_frames)
    if source != "audio":
        print(f"(beat detection fell back: {source}; using fixed 120bpm grid)")
    return frames, tempo


# ----------------------------------------------------------------------------
# STEP 3 — SYNC: who is ahead/behind the group
# ----------------------------------------------------------------------------
def motion_signal(frames, track_id):
    """Per-frame motion magnitude (limb speed) for one dancer."""
    sig = []
    prev = None
    for ft in frames:
        kp = ft.get(track_id)
        if kp is None:
            sig.append(np.nan)
            prev = None
            continue
        if prev is None:
            sig.append(0.0)
        else:
            sig.append(float(np.nanmean(np.linalg.norm(kp - prev, axis=1))))
        prev = kp
    return np.array(sig)


def analyze_sync(frames, beats):
    """For each dancer, compare their motion peaks to the GROUP median timing.
    Returns per-dancer average lead/lag in frames (negative = early, positive = late)."""
    ids = sorted({tid for ft in frames for tid in ft})
    sigs = {tid: motion_signal(frames, tid) for tid in ids}

    # group reference = median motion across all dancers per frame
    stack = np.vstack([sigs[t] for t in ids]) if ids else np.zeros((1, len(frames)))
    group = np.nanmedian(stack, axis=0)

    results = {}
    win = 8  # frames to search around each beat for the local motion peak
    for tid in ids:
        s = sigs[tid]
        offsets = []
        for b in beats:
            lo, hi = max(0, b - win), min(len(s), b + win)
            seg = s[lo:hi]
            gseg = group[lo:hi]
            if np.all(np.isnan(seg)) or np.all(np.isnan(gseg)):
                continue
            d_peak = np.nanargmax(seg)
            g_peak = np.nanargmax(gseg)
            offsets.append(d_peak - g_peak)
        if offsets:
            results[tid] = {
                'avg_offset_frames': float(np.mean(offsets)),
                'worst_offset_frames': float(np.max(np.abs(offsets))),
                'consistency': float(np.std(offsets)),
                'beats_measured': len(offsets),
            }
    return results, ids


# ----------------------------------------------------------------------------
# STEP 4 — FLOOR MAP via homography (you tap the 4 floor corners once)
# ----------------------------------------------------------------------------
def get_floor_corners(video_path):
    """Show frame 1, let user click 4 floor corners (TL,TR,BR,BL). Returns homography."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    pts = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    print("\nClick the 4 corners of the dance floor: TOP-LEFT, TOP-RIGHT, "
          "BOTTOM-RIGHT, BOTTOM-LEFT. Then press any key.")
    cv2.namedWindow("Tap floor corners")
    cv2.setMouseCallback("Tap floor corners", on_click)
    while True:
        disp = frame.copy()
        for p in pts:
            cv2.circle(disp, p, 6, (0, 255, 0), -1)
        cv2.imshow("Tap floor corners", disp)
        if cv2.waitKey(20) != -1 and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    src = np.array(pts, dtype=np.float32)
    dst = np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def compute_diagnostics(frames6, lock_diag, fps):
    """Stability metrics the captain can judge directly."""
    life = defaultdict(int)
    for ft in frames6:
        for tid in ft:
            life[tid] += 1
    minutes = max(len(frames6) / max(fps, 1e-9) / 60.0, 1e-9)
    raw = lock_diag.get("raw_fragments", len(life))
    stable = lock_diag.get("stable_ids", len(life))
    return {
        "stable_ids": stable,
        "raw_fragments": raw,
        "avg_lifetime_frames": float(np.mean(list(life.values()))) if life else 0.0,
        "avg_lifetime_sec": (float(np.mean(list(life.values()))) / fps) if life else 0.0,
        # each fragment beyond the first for an identity is one continuity break
        "id_switch_per_min": (raw - stable) / minutes,
        "avg_fragments_per_id": lock_diag.get("avg_fragments_per_id", 1.0),
        "dropped_fragments": lock_diag.get("dropped_fragments", 0),
    }


def _mux_audio(silent_video: str, audio_source: str, out_path: str) -> None:
    """Copy the original soundtrack onto the annotated (silent) video. Output fps
    is chosen so duration matches the source, so audio stays in sync. Falls back
    to the silent file if there is no audio or ffmpeg is missing."""
    import os, shutil, subprocess
    from beats import _has_audio_stream
    if shutil.which("ffmpeg") and _has_audio_stream(audio_source):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", silent_video, "-i", audio_source,
                 "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                 "-shortest", out_path, "-loglevel", "error"],
                check=True)
            os.remove(silent_video)
            return
        except Exception as e:
            print(f"          (audio mux failed: {e}; leaving silent video)")
    os.replace(silent_video, out_path)


def run_pipeline(args):
    from tracking import (MediaPipePoseEstimator, TopDownPoseEstimator,
                          YoloPersonDetector, RosterLock, remap_frames,
                          extract_tracks)

    # ---- STEP 1: pose + tracking (or load a cache) ----
    if args.cache:
        print(f"STEP 1/4  Loading cached tracks from {args.cache} ...")
        with open(args.cache, "rb") as f:
            c = pickle.load(f)
        frames6, frames_full = c["frames6"], c["frames_full"]
        fps, W, H = c["fps"], c["W"], c["H"]
        sample_every = c.get("sample_every", 1)
    else:
        print(f"STEP 1/4  Tracking dancers ({args.pose}, this is the slow part)...")
        if args.pose == "topdown":
            pose = TopDownPoseEstimator(
                MODEL_PATH, detector=YoloPersonDetector(args.yolo_model))
        else:
            pose = MediaPipePoseEstimator(MODEL_PATH, max_dancers=20)
        frames6, frames_full, fps, (W, H) = extract_tracks(
            args.video, pose, max_frames=args.max_frames,
            sample_every=args.sample_every,
            tracker_kwargs={"lost_track_buffer": args.lost_buffer})
        sample_every = args.sample_every

    raw_ids = len({tid for ft in frames6 for tid in ft})
    # When sampling 1-of-N frames, everything downstream (beats, sync, render)
    # lives in PROCESSED-frame space at this effective frame rate.
    eff_fps = fps / max(1, sample_every)
    print(f"          {len(frames6)} frames @ {fps:.1f}fps "
          f"(effective {eff_fps:.1f}fps), {W}x{H}, {raw_ids} raw track ids")

    # ---- STEP 2: roster lock -> stable identities ----
    print(f"STEP 2/4  Roster lock to ~{args.dancers} dancers ...")
    lock = RosterLock(n_dancers=args.dancers, max_gap=args.max_gap)
    mapping = lock.fit(frames_full)
    frames_full, frames6 = remap_frames(frames_full, frames6, mapping)
    diag = compute_diagnostics(frames6, lock.diagnostics, eff_fps)
    print(f"          raw {diag['raw_fragments']} -> stable {diag['stable_ids']} ids "
          f"| avg life {diag['avg_lifetime_sec']:.1f}s "
          f"| ID-switch est {diag['id_switch_per_min']:.1f}/min")

    # ---- hide non-dancers (bench / side figures) ----
    exclude = set()
    if args.exclude:
        exclude |= {int(x) for x in str(args.exclude).split(",") if x.strip()}
    if not args.no_auto_exclude:
        from formation import normalized_positions as _norm_fn
        from formation_consistency import detect_events as _detect
        _ev, _ = _detect(_norm_fn(frames_full, W, H), eff_fps,
                         formation_margin_s=args.formation_margin)
        auto = {e.dancer for e in _ev if e.kind == "structural_outlier"}
        if auto:
            print(f"          auto-detected non-dancers (isolated from group): "
                  f"{sorted(auto)}")
        exclude |= auto
    if exclude:
        frames_full = [{s: d for s, d in ft.items() if s not in exclude}
                       for ft in frames_full]
        frames6 = [{s: k for s, k in ft.items() if s not in exclude}
                   for ft in frames6]
        diag = compute_diagnostics(frames6, lock.diagnostics, eff_fps)
        real = len({s for ft in frames6 for s in ft})
        print(f"          hid {sorted(exclude)} -> {real} real dancers shown")

    # ---- STEP 3: real beat grid (indexed in processed-frame space) ----
    print("STEP 3/4  Detecting beat grid from audio ...")
    from beats import get_beats as get_beats_full
    beats, tempo, beat_src = get_beats_full(args.video, eff_fps, len(frames6))
    print(f"          tempo ~{tempo:.0f} bpm, {len(beats)} beats ({beat_src})")

    # ---- sync (verified math, unchanged) ----
    sync, ids = analyze_sync(frames6, beats)

    # ---- self-referential formation consistency (no chart) ----
    self_events = []
    if not args.no_self:
        from formation import normalized_positions
        from formation_consistency import detect_events
        from make_pdf import build_self_report
        print("          checking formation consistency (vs the group) ...")
        norm = normalized_positions(frames_full, W, H)
        self_events, holds = detect_events(
            norm, eff_fps, formation_margin_s=args.formation_margin)
        sstats = build_self_report(self_events, holds, args.self_pdf,
                                   fps=eff_fps, video=args.video,
                                   n_dancers=args.dancers)
        print(f"          {len(holds)} formations; {sstats['real_events']} flags, "
              f"{sstats['structural']} side-figures -> {args.self_pdf}")

    # ---- formation key: names + position grading (optional) ----
    name_map, position_events = {}, []
    if args.key:
        from formation import (load_key, normalized_positions, assign_names,
                               grade_positions)
        from make_pdf import build_report
        print(f"          grading positions against {args.key} ...")
        key = load_key(args.key, eff_fps)
        norm = normalized_positions(frames_full, W, H)
        name_map = assign_names(key, norm)
        position_events = grade_positions(key, norm, name_map)
        pstats = build_report(key, name_map, position_events, args.pdf,
                              fps=eff_fps, video=args.video)
        print(f"          named {len(name_map)} dancers; "
              f"{pstats['flagged']} out-of-position, {pstats['missing']} missing"
              f" -> {args.pdf}")

    # ---- STEP 4: annotated video (rendered silent, then muxed with audio) ----
    if not args.skip_render:
        print(f"STEP 4/4  Rendering {args.out} ...")
        from render import render_annotated
        silent = args.out + ".silent.mp4"
        render_annotated(args.video, frames_full, frames6, beats, silent,
                         fps=fps, sample_every=sample_every, diagnostics=diag,
                         tempo=tempo, beat_source=beat_src,
                         name_map=name_map, position_events=position_events,
                         self_events=self_events, show_timing=args.timing_tags)
        _mux_audio(silent, args.video, args.out)
        print(f"          wrote {args.out} (with audio)")
    else:
        print("STEP 4/4  (skipped render)")

    report = {
        'video': args.video, 'fps': fps, 'frame_size': [W, H],
        'pose_source': args.pose if not args.cache else 'cache',
        'tempo_bpm': tempo, 'beat_source': beat_src, 'n_frames': len(frames6),
        'roster_target': args.dancers,
        'diagnostics': diag,
        'beats': len(beats),
        'sync_per_dancer': {str(k): v for k, v in sync.items()},
        'NOTES': [
            "Stable IDs come from YOLO top-down detection + ByteTrack + roster lock.",
            "Beats are detected from the real audio; sync offsets are in frames "
            "(negative=early, positive=late).",
            "Roster lock stitches fragmented tracks into the given dancer count; "
            "supply external anchors (formation answer-key) later to pin who-is-who.",
        ]
    }
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to a practice-run video")
    ap.add_argument("--dancers", type=int, default=16,
                    help="real roster size to lock identities to")
    ap.add_argument("--pose", choices=["topdown", "mediapipe"], default="topdown",
                    help="pose source: topdown=YOLO+per-crop pose (finds all "
                         "dancers), mediapipe=bottom-up (fast, few dancers)")
    ap.add_argument("--yolo-model", default="yolov8s.pt")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=1,
                    help="process 1 of every N frames (speed knob)")
    ap.add_argument("--lost-buffer", type=int, default=60,
                    help="ByteTrack frames-to-keep-a-lost-track (occlusion gap)")
    ap.add_argument("--max-gap", type=int, default=1200,
                    help="roster-lock: longest absence (frames) to bridge")
    ap.add_argument("--cache", default=None,
                    help="load tracks from a pickle (build_tracks.py) instead of "
                         "re-running pose; skips STEP 1")
    ap.add_argument("--key", default=None,
                    help="formation_key.json: names + intended positions to grade")
    ap.add_argument("--pdf", default="formation_report.pdf",
                    help="external-chart report PDF (only when --key is given)")
    ap.add_argument("--no-self", action="store_true",
                    help="skip the self-referential formation-consistency report")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated dancer IDs to hide as non-dancers, "
                         "e.g. --exclude 9,13")
    ap.add_argument("--no-auto-exclude", action="store_true",
                    help="don't auto-hide figures that are isolated from the group")
    ap.add_argument("--formation-margin", type=float, default=0.6,
                    help="timing margin of error (s): only flag EARLY/LATE to a "
                         "formation beyond this. Raise it to be more lenient.")
    ap.add_argument("--self-pdf", default="formation_consistency.pdf",
                    help="self-referential (no-chart) report PDF path")
    ap.add_argument("--timing-tags", action="store_true",
                    help="overlay per-beat EARLY/LATE tags (noisy; off by default)")
    ap.add_argument("--out", default="annotated_output.mp4")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--no-floor", action="store_true",
                    help="(floor homography is interactive; off by default here)")
    args = ap.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
