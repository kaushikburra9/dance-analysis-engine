"""
TRACKING ENGINE — swappable pose source + proper multi-object tracking.
=======================================================================
This module replaces the old nearest-centroid SimpleTracker. It is split into
three clean layers so each can be swapped without touching the others:

  1. PoseEstimator  — produces per-frame pose detections (bbox + keypoints + score).
                      MediaPipe today; RTMPose+GPU later. The rest of the pipeline
                      only sees the `PoseDetection` interface, never MediaPipe.

  2. ByteTrackAssociator — turns per-frame detections into motion-consistent tracks
                      using ByteTrack (IOU + Kalman prediction + two-stage low-conf
                      recovery, which re-IDs across short occlusion gaps).

  3. RosterLock (Step 2, added below) — consolidates fragmented tracks into a fixed
                      roster of N persistent dancer identities.

The downstream sync math + floor homography in analyze_run.py are UNCHANGED; they
consume `frames6` (a list of {id: 6-joint array}) exactly as before.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# The pose interface every estimator must satisfy.
# ----------------------------------------------------------------------------
# MediaPipe Pose landmark indices (BlazePose 33-point topology).
L_WRIST, R_WRIST = 15, 16
L_ANKLE, R_ANKLE = 27, 28
L_HIP, R_HIP = 23, 24
# The 6 joints the verified sync math expects, IN THIS ORDER:
#   [L_wrist, R_wrist, L_ankle, R_ankle, L_hip, R_hip]
SYNC_JOINTS = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP]

# Skeleton edges for drawing (subset of BlazePose connections worth seeing).
POSE_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),          # torso box
    (11, 13), (13, 15), (12, 14), (14, 16),          # arms
    (23, 25), (25, 27), (24, 26), (26, 28),          # legs
    (27, 31), (28, 32),                              # feet
    (11, 0), (12, 0),                                # neck->nose
]


@dataclass
class PoseDetection:
    """One detected person in one frame, in pixel coordinates.

    keypoints : (33, 2) float32 — full BlazePose landmarks in px (for drawing).
    visibility: (33,)  float32 — per-landmark visibility in [0,1].
    bbox      : (x1, y1, x2, y2) float — tight box around visible landmarks.
    score     : float — detection confidence proxy (mean visibility), for ByteTrack.
    """
    keypoints: np.ndarray
    visibility: np.ndarray
    bbox: Tuple[float, float, float, float]
    score: float

    def joints6(self) -> np.ndarray:
        """The 6-joint subset the verified sync math consumes."""
        return self.keypoints[SYNC_JOINTS].astype(np.float32)

    def centroid(self) -> Tuple[float, float]:
        """Mid-hip centroid (same reference point the old tracker used)."""
        c = self.keypoints[[L_HIP, R_HIP]].mean(axis=0)
        return float(c[0]), float(c[1])


class PoseEstimator(ABC):
    """Swappable pose source. Implement `detect` and you can drop in any model."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[PoseDetection]:
        ...

    def close(self) -> None:  # optional cleanup hook
        pass


# ----------------------------------------------------------------------------
# MediaPipe implementation of the pose interface.
# ----------------------------------------------------------------------------
class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe PoseLandmarker (Tasks API, VIDEO mode, multi-person)."""

    def __init__(self, model_path: str, max_dancers: int = 20,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=max_dancers,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[PoseDetection]:
        import cv2
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        out: List[PoseDetection] = []
        if not res.pose_landmarks:
            return out
        for lms in res.pose_landmarks:
            pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)
            vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms],
                           dtype=np.float32)
            # bbox from reasonably-visible landmarks (fall back to all if none).
            good = pts[vis > 0.3]
            if len(good) < 4:
                good = pts
            x1, y1 = good.min(axis=0)
            x2, y2 = good.max(axis=0)
            # clamp + pad a little so the box brackets the body
            pad = 0.05 * max(x2 - x1, y2 - y1)
            bbox = (float(max(0, x1 - pad)), float(max(0, y1 - pad)),
                    float(min(W, x2 + pad)), float(min(H, y2 + pad)))
            score = float(np.clip(vis.mean(), 0.05, 1.0))
            out.append(PoseDetection(keypoints=pts, visibility=vis,
                                     bbox=bbox, score=score))
        return out

    def close(self) -> None:
        self._landmarker.close()


# ----------------------------------------------------------------------------
# Top-down pose estimation: a person DETECTOR finds every dancer, then a pose
# model runs on each crop. This is the architecture that actually finds 16 small
# dancers in a wide shot — MediaPipe's bottom-up detector only surfaces 1-4.
# It is ALSO the production shape: swap YOLO->RTMDet and the per-crop MediaPipe
# ->RTMPose later, with zero changes downstream.
# ----------------------------------------------------------------------------
class YoloPersonDetector:
    """YOLOv8 person detector. Returns [(bbox, conf), ...] per frame."""

    def __init__(self, model_path: str = "yolov8s.pt", conf: float = 0.25,
                 imgsz: int = 1280):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._conf = conf
        self._imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray
               ) -> List[Tuple[Tuple[float, float, float, float], float]]:
        r = self._model.predict(frame_bgr, classes=[0], conf=self._conf,
                                imgsz=self._imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        out = []
        for b, c in zip(xyxy, confs):
            out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        float(c)))
        return out


class TopDownPoseEstimator(PoseEstimator):
    """Person-detector + per-crop single-person pose.

    The detector (default YOLOv8) finds every dancer; a single-person pose model
    (default MediaPipe IMAGE mode) runs on each crop and its landmarks are mapped
    back to full-frame pixels. Keeps the PoseEstimator interface, so ByteTrack /
    RosterLock / the sync math never change. To go GPU later: swap the detector
    for RTMDet and `_pose_crop` for RTMPose.
    """

    def __init__(self, model_path: str, detector: Optional[YoloPersonDetector] = None,
                 crop_pad: float = 0.15, min_box_area: int = 24 * 48):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self._detector = detector or YoloPersonDetector()
        self._crop_pad = crop_pad
        self._min_box_area = min_box_area
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,  # one crop at a time
            num_poses=1,
            min_pose_detection_confidence=0.3)
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[PoseDetection]:
        import cv2
        H, W = frame_bgr.shape[:2]
        out: List[PoseDetection] = []
        for (x1, y1, x2, y2), conf in self._detector.detect(frame_bgr):
            bw, bh = x2 - x1, y2 - y1
            if bw * bh < self._min_box_area:
                continue
            # pad the crop so limbs near the box edge aren't clipped
            px, py = self._crop_pad * bw, self._crop_pad * bh
            cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            cx2, cy2 = int(min(W, x2 + px)), int(min(H, y2 + py))
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = self._landmarker.detect(
                self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_landmarks:
                continue
            lms = res.pose_landmarks[0]
            # map crop-normalized landmarks -> full-frame pixels
            pts = np.array([[cx1 + lm.x * cw, cy1 + lm.y * ch] for lm in lms],
                           dtype=np.float32)
            vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms],
                           dtype=np.float32)
            # keep the reliable YOLO box (clamped) as the track bbox + its conf
            bbox = (float(max(0, x1)), float(max(0, y1)),
                    float(min(W, x2)), float(min(H, y2)))
            out.append(PoseDetection(keypoints=pts, visibility=vis,
                                     bbox=bbox, score=float(conf)))
        return out

    def close(self) -> None:
        self._landmarker.close()


# ----------------------------------------------------------------------------
# ByteTrack association layer.
# ----------------------------------------------------------------------------
class ByteTrackAssociator:
    """Wraps supervision's ByteTrack. Feeds it boxes+scores, returns tracker_ids.

    ByteTrack gives us: IOU association, Kalman motion prediction, and a second
    association pass on low-confidence boxes — which is what bridges the brief
    occlusions that made the old centroid tracker spawn phantom IDs.
    """

    def __init__(self, fps: float, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 60, minimum_matching_threshold: float = 0.8,
                 minimum_consecutive_frames: int = 1):
        import supervision as sv
        self._sv = sv
        # `lost_track_buffer` = frames a track survives while unmatched (re-ID gap).
        # 60 @30fps = ~2s of occlusion tolerance.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the v0.30 move notice
            self._bt = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=int(round(fps)),
                minimum_consecutive_frames=minimum_consecutive_frames)

    def update(self, detections: List[PoseDetection]
               ) -> List[Tuple[int, PoseDetection]]:
        """Returns [(tracker_id, PoseDetection), ...] for this frame."""
        sv = self._sv
        if not detections:
            # Still advance the tracker's internal clock with an empty update.
            self._bt.update_with_detections(sv.Detections.empty())
            return []
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        conf = np.array([d.score for d in detections], dtype=np.float32)
        cls = np.zeros(len(detections), dtype=int)
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        # Carry an index into our PoseDetection list through ByteTrack via `data`.
        det.data["det_idx"] = np.arange(len(detections))
        tracked = self._bt.update_with_detections(det)

        results: List[Tuple[int, PoseDetection]] = []
        idxs = tracked.data.get("det_idx", np.arange(len(tracked)))
        for k in range(len(tracked)):
            tid = tracked.tracker_id[k]
            if tid is None:
                continue
            src = detections[int(idxs[k])]
            results.append((int(tid), src))
        return results


# ----------------------------------------------------------------------------
# STEP 2 — ROSTER LOCK: consolidate fragmented tracks into N persistent IDs.
# ----------------------------------------------------------------------------
# ByteTrack still fragments an identity whenever a dancer is occluded longer than
# its lost-track buffer, or leaves and re-enters. The roster lock stitches those
# fragments back together under a fixed set of N "slots" (the real dancer count),
# using two physical facts:
#   * two fragments that overlap IN TIME are different people  -> never merged;
#   * a dancer who vanishes reappears NEAR where they left      -> gap-scaled
#                                                                  spatial gate.
# Anchor seeding is the seam for the future formation answer-key: pass external
# `anchors` (one reference point per known dancer) and fragments get assigned to
# the right person instead of to data-derived seeds.

@dataclass
class _Fragment:
    raw_id: int
    frames: List[int]                       # frame indices present (sorted)
    centroids: Dict[int, Tuple[float, float]]  # frame -> (x,y)

    @property
    def start(self) -> int: return self.frames[0]
    @property
    def end(self) -> int: return self.frames[-1]
    @property
    def lifetime(self) -> int: return len(self.frames)

    def centroid_at(self, fidx: int) -> Tuple[float, float]:
        return self.centroids[fidx]


class _Slot:
    """One roster identity: an occupancy timeline assembled from fragments."""
    def __init__(self, stable_id: int):
        self.stable_id = stable_id
        self.frame_to_centroid: Dict[int, Tuple[float, float]] = {}
        self.raw_ids: List[int] = []
        self._occupied = set()

    def add(self, frag: _Fragment):
        self.raw_ids.append(frag.raw_id)
        for f in frag.frames:
            self.frame_to_centroid[f] = frag.centroids[f]
            self._occupied.add(f)

    def conflicts(self, frag: _Fragment) -> bool:
        return any(f in self._occupied for f in frag.frames)

    def nearest_centroid_in_time(self, fidx: int):
        """Slot's known centroid closest in time to `fidx`, and the frame gap."""
        if not self.frame_to_centroid:
            return None, None
        best_f = min(self.frame_to_centroid, key=lambda f: abs(f - fidx))
        return self.frame_to_centroid[best_f], abs(best_f - fidx)


class RosterLock:
    """Collapse raw ByteTrack ids into <= n_dancers persistent identities.

    Parameters
    ----------
    n_dancers : the real roster size (e.g. 16).
    max_speed_px : assumed max centroid speed (px/frame); sets how far a dancer
        may have drifted during an absence before we stop believing it's them.
    base_radius_px : slack added on top of speed*gap (pose jitter, box wobble).
    max_gap : longest absence (frames) we will bridge across.
    min_fragment_frames : fragments shorter than this are treated as noise when
        they cannot be attached to a slot.
    anchors : optional list of (x,y) reference points, one per known dancer
        (FUTURE answer-key hook). When given, slots are seeded at these points
        instead of from the longest-lived fragments.
    """

    def __init__(self, n_dancers: int, max_speed_px: float = 45.0,
                 base_radius_px: float = 80.0, max_gap: int = 1200,
                 max_match_dist: float = 320.0, min_fragment_frames: int = 3,
                 anchors: Optional[List[Tuple[float, float]]] = None):
        self.n = n_dancers
        self.max_speed_px = max_speed_px
        self.base_radius_px = base_radius_px
        self.max_gap = max_gap
        # Hard ceiling on the spatial gate: a dancer returning to formation
        # reappears near their spot, so even across a long absence we refuse
        # merges beyond this distance. Prevents over-merging distinct dancers.
        self.max_match_dist = max_match_dist
        self.min_fragment_frames = min_fragment_frames
        self.anchors = anchors
        self.diagnostics: Dict[str, float] = {}

    # -- build fragments from per-frame full detections -----------------------
    @staticmethod
    def _fragments_from_frames(frames_full: List[Dict[int, "PoseDetection"]]
                               ) -> Dict[int, _Fragment]:
        frags: Dict[int, _Fragment] = {}
        for fidx, ftracks in enumerate(frames_full):
            for raw_id, det in ftracks.items():
                fr = frags.get(raw_id)
                if fr is None:
                    fr = _Fragment(raw_id=raw_id, frames=[], centroids={})
                    frags[raw_id] = fr
                fr.frames.append(fidx)
                fr.centroids[fidx] = det.centroid()
        for fr in frags.values():
            fr.frames.sort()
        return frags

    def _allowed_dist(self, gap: int) -> float:
        grown = self.base_radius_px + self.max_speed_px * min(gap, self.max_gap)
        return min(grown, self.max_match_dist)

    def fit(self, frames_full: List[Dict[int, "PoseDetection"]]) -> Dict[int, int]:
        """Returns raw_id -> stable_id (0..n-1), or -1 for dropped noise."""
        frags = self._fragments_from_frames(frames_full)
        order = sorted(frags.values(), key=lambda f: (-f.lifetime, f.start))

        slots: List[_Slot] = [_Slot(i) for i in range(self.n)]
        mapping: Dict[int, int] = {}
        unassigned: List[_Fragment] = []

        # -- seed slots ------------------------------------------------------
        if self.anchors is not None:
            # answer-key mode: pre-place an anchor centroid in each slot
            for i, a in enumerate(self.anchors[:self.n]):
                slots[i].frame_to_centroid[-1] = (float(a[0]), float(a[1]))
            seeds = order
        else:
            # data mode: longest-lived fragments seed the slots
            seeds = order[self.n:]
            for i, frag in enumerate(order[:self.n]):
                slots[i].add(frag)
                mapping[frag.raw_id] = i

        # -- greedily attach remaining fragments to best compatible slot -----
        for frag in seeds:
            if frag.raw_id in mapping:
                continue
            best_slot, best_cost = None, None
            for slot in slots:
                if slot.conflicts(frag):
                    continue  # overlapping in time => different dancer
                # cost = how far the fragment's nearest boundary sits from the
                # slot's nearest-in-time known position, gap-gated.
                cand_costs = []
                for boundary in (frag.start, frag.end):
                    ref, gap = slot.nearest_centroid_in_time(boundary)
                    if ref is None:
                        cand_costs.append(0.0)  # empty anchored slot: accept
                        continue
                    fc = frag.centroid_at(boundary)
                    d = float(np.hypot(fc[0] - ref[0], fc[1] - ref[1]))
                    if d <= self._allowed_dist(gap):
                        cand_costs.append(d)
                if not cand_costs:
                    continue
                cost = min(cand_costs)
                if best_cost is None or cost < best_cost:
                    best_slot, best_cost = slot, cost
            if best_slot is not None:
                best_slot.add(frag)
                mapping[frag.raw_id] = best_slot.stable_id
            else:
                unassigned.append(frag)

        # -- noise / overflow handling ---------------------------------------
        dropped = 0
        for frag in unassigned:
            mapping[frag.raw_id] = -1
            dropped += 1

        used_slots = sorted({v for v in mapping.values() if v >= 0})
        frags_per_id = [len(slots[s].raw_ids) for s in used_slots]
        self.diagnostics = {
            "raw_fragments": len(frags),
            "stable_ids": len(used_slots),
            "dropped_fragments": dropped,
            "avg_fragments_per_id": float(np.mean(frags_per_id)) if frags_per_id else 0.0,
            "max_fragments_per_id": int(np.max(frags_per_id)) if frags_per_id else 0,
        }
        self._slots = slots
        return mapping


def remap_frames(frames_full: List[Dict[int, "PoseDetection"]],
                 frames6: List[Dict[int, np.ndarray]],
                 mapping: Dict[int, int]):
    """Apply raw_id->stable_id mapping. Drops -1 (noise). If two raw ids collide
    on a stable id in the same frame (shouldn't, given non-overlap), keeps the
    higher-score detection."""
    new_full, new6 = [], []
    for ffull, f6 in zip(frames_full, frames6):
        nf, n6 = {}, {}
        for raw_id, det in ffull.items():
            sid = mapping.get(raw_id, -1)
            if sid < 0:
                continue
            if sid in nf and det.score <= nf[sid].score:
                continue
            nf[sid] = det
            n6[sid] = f6[raw_id]
        new_full.append(nf)
        new6.append(n6)
    return new_full, new6


# ----------------------------------------------------------------------------
# Driver: run a video through (pose -> tracker) and collect tracks.
# ----------------------------------------------------------------------------
def extract_tracks(video_path: str, pose: PoseEstimator, fps_hint: Optional[float] = None,
                   max_frames: Optional[int] = None, sample_every: int = 1,
                   tracker_kwargs: Optional[dict] = None, progress_every: int = 100):
    """Run the full (pose -> ByteTrack) pipeline over a video.

    Returns:
      frames6     : list[dict[int, (6,2) np.ndarray]]  — feeds the verified sync math.
      frames_full : list[dict[int, PoseDetection]]      — full skeletons for drawing.
      fps, (W, H)
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = ByteTrackAssociator(fps=fps, **(tracker_kwargs or {}))

    frames6: List[Dict[int, np.ndarray]] = []
    frames_full: List[Dict[int, PoseDetection]] = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % sample_every != 0:
            fi += 1
            continue
        if max_frames and len(frames6) >= max_frames:
            break

        ts = int((fi / fps) * 1000)
        dets = pose.detect(frame, ts)
        tracked = tracker.update(dets)

        f6: Dict[int, np.ndarray] = {}
        ffull: Dict[int, PoseDetection] = {}
        for tid, d in tracked:
            f6[tid] = d.joints6()
            ffull[tid] = d
        frames6.append(f6)
        frames_full.append(ffull)

        if progress_every and len(frames6) % progress_every == 0:
            print(f"          ...{len(frames6)} frames processed", flush=True)
        fi += 1

    cap.release()
    pose.close()
    return frames6, frames_full, fps, (W, H)
