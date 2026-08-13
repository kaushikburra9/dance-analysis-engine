"""
STEP 1 TEST — detection + tracking quality.
Compares three configurations on the same clip:
  (A) MediaPipe bottom-up  + old SimpleTracker   (the original pipeline)
  (B) MediaPipe bottom-up  + ByteTrack
  (C) YOLO top-down pose   + ByteTrack           (the upgrade)
Reports poses/frame, distinct tracked IDs, and track lifetimes.
"""
import argparse, time
from collections import defaultdict
import numpy as np
import cv2

from tracking import (MediaPipePoseEstimator, TopDownPoseEstimator,
                      ByteTrackAssociator, YoloPersonDetector)
from analyze_run import SimpleTracker, MODEL_PATH


def run_pose_then_trackers(video, pose, fps, W, use_simple=False, max_frames=None):
    bt = ByteTrackAssociator(fps=fps, lost_track_buffer=60)
    simple = SimpleTracker(max_dist=max(80, W // 12)) if use_simple else None
    bt_life, s_life = defaultdict(int), defaultdict(int)
    n_det = 0
    cap = cv2.VideoCapture(video)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and fi >= max_frames):
            break
        dets = pose.detect(frame, int((fi / fps) * 1000))
        n_det += len(dets)
        for tid, _ in bt.update(dets):
            bt_life[tid] += 1
        if simple is not None:
            cen = {i: d.centroid() for i, d in enumerate(dets)}
            for sid in simple.update(cen).values():
                s_life[sid] += 1
        fi += 1
        if fi % 100 == 0:
            print(f"    ...{fi} frames", flush=True)
    cap.release()
    pose.close()
    return n_det, fi, bt_life, s_life


def stat(life):
    if not life:
        return 0, 0.0, 0
    v = np.array(list(life.values()))
    return len(life), float(v.mean()), int(v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="subclip.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--yolo-model", default="yolov8s.pt")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(">> MediaPipe bottom-up (feeds SimpleTracker + ByteTrack)...")
    t0 = time.time()
    mp_pose = MediaPipePoseEstimator(MODEL_PATH, max_dancers=20)
    mp_ndet, nf, mp_bt, mp_simple = run_pose_then_trackers(
        args.video, mp_pose, fps, W, use_simple=True, max_frames=args.max_frames)
    t_mp = time.time() - t0

    print(">> YOLO top-down pose (feeds ByteTrack)...")
    t0 = time.time()
    td_pose = TopDownPoseEstimator(
        MODEL_PATH, detector=YoloPersonDetector(args.yolo_model))
    td_ndet, _, td_bt, _ = run_pose_then_trackers(
        args.video, td_pose, fps, W, use_simple=False, max_frames=args.max_frames)
    t_td = time.time() - t0

    s_n, s_a, s_m = stat(mp_simple)
    b_n, b_a, b_m = stat(mp_bt)
    t_n, t_a, t_m = stat(td_bt)

    print("\n" + "=" * 66)
    print(f"Clip {args.video}  {nf} frames @ {fps:.1f}fps  {W}x{H}")
    print("-" * 66)
    print(f"{'config':38} {'poses/fr':>8} {'IDs':>5} {'avgLife':>8} {'maxLife':>7}")
    print(f"{'A MediaPipe + SimpleTracker (orig)':38} "
          f"{mp_ndet/nf:8.1f} {s_n:5d} {s_a:8.1f} {s_m:7d}")
    print(f"{'B MediaPipe + ByteTrack':38} "
          f"{mp_ndet/nf:8.1f} {b_n:5d} {b_a:8.1f} {b_m:7d}")
    print(f"{'C YOLO top-down + ByteTrack':38} "
          f"{td_ndet/nf:8.1f} {t_n:5d} {t_a:8.1f} {t_m:7d}")
    print("-" * 66)
    print(f"timing: MediaPipe path {t_mp:.1f}s ({nf/t_mp:.1f}fps) | "
          f"top-down path {t_td:.1f}s ({nf/t_td:.1f}fps)")
    print("=" * 66)


if __name__ == "__main__":
    main()
