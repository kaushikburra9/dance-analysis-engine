"""
Run (pose -> ByteTrack) over a clip ONCE and cache the tracks to a pickle, so
roster-lock tuning doesn't pay the pose cost every iteration.
"""
import argparse, pickle, time
from tracking import (MediaPipePoseEstimator, TopDownPoseEstimator,
                      YoloPersonDetector, extract_tracks)
from analyze_run import MODEL_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="subclip.mp4")
    ap.add_argument("--out", default="tracks_cache.pkl")
    ap.add_argument("--pose", choices=["topdown", "mediapipe"], default="topdown")
    ap.add_argument("--yolo-model", default="yolov8s.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25,
                    help="YOLO person-detection confidence (lower = find more, "
                         "e.g. 0.15 for small/dark dancers)")
    ap.add_argument("--yolo-imgsz", type=int, default=1280,
                    help="YOLO inference size (higher = better on small dancers)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--lost-buffer", type=int, default=60)
    args = ap.parse_args()

    if args.pose == "topdown":
        pose = TopDownPoseEstimator(
            MODEL_PATH,
            detector=YoloPersonDetector(args.yolo_model, conf=args.yolo_conf,
                                        imgsz=args.yolo_imgsz))
    else:
        pose = MediaPipePoseEstimator(MODEL_PATH, max_dancers=20)

    t0 = time.time()
    frames6, frames_full, fps, (W, H) = extract_tracks(
        args.video, pose, max_frames=args.max_frames,
        sample_every=args.sample_every,
        tracker_kwargs={"lost_track_buffer": args.lost_buffer})
    dt = time.time() - t0

    with open(args.out, "wb") as f:
        pickle.dump({"frames6": frames6, "frames_full": frames_full,
                     "fps": fps, "W": W, "H": H, "video": args.video,
                     "pose": args.pose, "sample_every": args.sample_every}, f)
    n_raw = len({tid for ft in frames6 for tid in ft})
    print(f"\ncached {len(frames6)} frames, {n_raw} raw ids -> {args.out} "
          f"({dt:.0f}s, {len(frames6)/max(dt,1e-9):.1f}fps)")


if __name__ == "__main__":
    main()
