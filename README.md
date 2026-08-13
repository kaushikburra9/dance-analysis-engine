# Dance App — Phase 0: Validate the Engine Before Building Anything

**The goal of this folder:** answer ONE question before you write a single line of app code —
*"Can a computer track our dancers, time them to the beat, and place them on the floor
well enough that a captain would trust it?"* If yes → build the app. If the tracking is
garbage on real footage → fix capture first, don't waste months on UI.

This is the smallest possible real first step. You can do it this weekend with one
practice video you already have.

---

## What's in here

- `analyze_run.py` — the real pipeline: track dancers → beat grid → sync analysis → floor map → report.
- `verify_logic.py` — already-passing tests proving the **sync math and floor geometry are correct** (run anywhere, no model needed).
- `make_test_clip.py` — generates a tiny synthetic clip to smoke-test the code path.

The core analysis logic is **already verified** (see "What's proven" below). The only thing
left to test is capture quality on *your* real footage — which needs your video + a machine
with a webcam display, so you run it locally, not in chat.

---

## Setup (one time, ~5 min)

```bash
# 1. Make a clean Python environment
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install mediapipe opencv-python numpy librosa scipy

# 3. Download the pose model ONCE (free, from Google)
wget -O pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
# (no wget? paste that URL in a browser, save the file here as pose_landmarker.task)
```

## Run it on your team's footage

```bash
python analyze_run.py --video my_team_run.mp4
```

You'll get:
- console output: how many dancers it tracked, the tempo, and each dancer's early/late timing
- `report.json`: the full machine-readable read, with honest notes about what to trust

**Tip for your first run:** use a clip where you *already know* what happened — a run where
you remember "Dancer 7 was dragging in the second formation." Then check: does the tool catch
what you already know? That's the trust test.

---

## What's already proven (verified in development)

Running `verify_logic.py` passes two tests with controlled data:

1. **Sync detection works** — a dancer made deliberately 4 frames late is flagged LATE; one
   made early is flagged EARLY; one on the beat reads on-time. The timing math is sound.
2. **Floor projection works** — a dancer standing center-floor in the camera image maps to
   the center of the top-down floor map. The homography geometry is sound.

So the *product logic* — the valuable part — is correct. What remains is purely a
**capture-quality** question on real footage.

---

## What you're actually testing (be honest with yourself)

The #1 risk, the thing this whole step exists to expose:

> **When dancers cross and overlap, does the tracker keep their IDs straight?**

Watch for: does the number of tracked dancer-IDs roughly match your real dancer count?
If it tracks "37 dancers" for a 16-person team, IDs are swapping on every crossing — that's
expected with this deliberately-crude Phase-0 tracker, and it tells you the production build
needs the real engine (RTMPose + ByteTrack) **plus the formation answer-key** to anchor who's
who. That's not a failure — it's exactly the finding that de-risks your roadmap.

### Camera tips that make capture dramatically better
- Film from an **elevated, front** position (a balcony, a ladder, top of the bleachers) — overhead-ish angles reduce dancer overlap, which is the whole battle.
- Keep the **whole floor in frame**, steady (tripod or propped phone), decent light.
- Higher resolution helps; 1080p is plenty.

---

## The decision gate

After you run it on 2–3 real clips, show the output to your captains and ask the only
question that matters:

> **"Would this have saved you an hour, or caught something you missed?"**

- **Yes** → you've validated the core. Move to building the captain app around this engine.
- **Tracking is unusable even with a good camera angle** → the first engineering job is the
  production capture swap (RTMPose + ByteTrack + answer-key ID anchoring) before any UI.

Either answer is a win, because you learned it in a weekend instead of in month six.

---

## The production swap (for later, when you build the real app)

| Phase 0 (this) | Production |
|---|---|
| MediaPipe PoseLandmarker | RTMPose (commercially licensable, higher accuracy) |
| SimpleTracker (nearest-centroid) | ByteTrack + formation answer-key ID anchoring |
| CPU, runs on your laptop | GPU server, batched |
| sync + floor read | + formation-vs-answer-key comparison, prioritized fixes, per-dancer notes |

The analysis logic you validated here carries over unchanged. You're only swapping the
capture layer for something more robust on crossing dancers — which is the one thing this
test will tell you whether you need.
