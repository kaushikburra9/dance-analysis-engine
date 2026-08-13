"""
STEP 3 — BEAT GRID from real audio.

The old code called librosa.load() straight on the .MOV. librosa's default
backend (soundfile) can't read a QuickTime container, so it raised
"PySoundFile failed" and then the tempo handling tripped on a 0-d array,
silently falling back to a fixed 120bpm grid — which makes the sync read
meaningless.

Fix: probe for an audio stream, extract it to a real WAV with ffmpeg, run
librosa.beat.beat_track on that, and coerce the tempo (which modern librosa
returns as a length-1 array) to a scalar. Only fall back to a fixed grid when
there is genuinely no audio.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np


def _has_audio_stream(video_path: str) -> bool:
    """True if ffprobe finds at least one audio stream."""
    if not shutil.which("ffprobe"):
        return True  # can't check; let extraction try and fail gracefully
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30)
        return "audio" in out.stdout
    except Exception:
        return True


def _extract_wav(video_path: str, sr: int = 22050) -> str:
    """Extract mono WAV via ffmpeg to a temp file. Returns the path."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "wav", tmp.name, "-loglevel", "error"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return tmp.name


def _tempo_scalar(tempo) -> float:
    """librosa may return tempo as a length-1 ndarray; coerce to float safely."""
    arr = np.atleast_1d(np.asarray(tempo, dtype=float))
    return float(arr.flat[0])


def get_beats(video_path: str, fps: float, n_frames: int
              ) -> Tuple[List[int], float, str]:
    """Return (beat_video_frames, tempo_bpm, source).

    source is one of: 'audio' (real detected beats), 'no-audio' / 'error:<...>'
    (fixed 120bpm fallback) so the caller can report honestly.
    """
    if not _has_audio_stream(video_path):
        step = max(1, int(fps * 0.5))  # 120 bpm
        return list(range(0, n_frames, step)), 120.0, "no-audio"

    wav = None
    try:
        import librosa
        wav = _extract_wav(video_path)
        y, sr = librosa.load(wav, sr=None, mono=True)
        if y.size == 0:
            raise RuntimeError("empty audio after extraction")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = _tempo_scalar(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_video_frames = [int(round(t * fps)) for t in beat_times
                             if 0 <= int(round(t * fps)) < n_frames]
        if not beat_video_frames:
            raise RuntimeError("no beats detected in audio")
        return beat_video_frames, tempo, "audio"
    except Exception as e:
        step = max(1, int(fps * 0.5))
        return list(range(0, n_frames, step)), 120.0, f"error:{e}"
    finally:
        if wav and os.path.exists(wav):
            os.remove(wav)


if __name__ == "__main__":
    import argparse, cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    beats, tempo, src = get_beats(args.video, fps, n)
    print(f"video={args.video} fps={fps:.2f} frames={n}")
    print(f"source={src}  tempo={tempo:.1f}bpm  beats={len(beats)}")
    if beats:
        ivs = np.diff(beats)
        print(f"first beats (frames): {beats[:12]}")
        print(f"median inter-beat: {np.median(ivs):.1f} frames "
              f"= {np.median(ivs)/fps*1000:.0f}ms "
              f"=> {60.0/(np.median(ivs)/fps):.1f} bpm from spacing")
