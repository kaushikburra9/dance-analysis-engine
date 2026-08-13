"""
STEP 2 TEST — roster lock. Loads cached tracks, collapses raw ByteTrack ids into
N persistent dancer identities, and reports before/after + diagnostics.
"""
import argparse, pickle
import numpy as np
from collections import defaultdict
from tracking import RosterLock, remap_frames


def lifetimes(frames):
    life = defaultdict(int)
    for ft in frames:
        for tid in ft:
            life[tid] += 1
    return life


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="tracks_cache.pkl")
    ap.add_argument("--dancers", type=int, default=16)
    ap.add_argument("--max-gap", type=int, default=1200)
    ap.add_argument("--max-speed", type=float, default=45.0)
    ap.add_argument("--base-radius", type=float, default=80.0)
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        c = pickle.load(f)
    frames6, frames_full = c["frames6"], c["frames_full"]
    n_frames = len(frames6)

    raw_life = lifetimes(frames6)
    raw_ids = len(raw_life)

    lock = RosterLock(n_dancers=args.dancers, max_gap=args.max_gap,
                      max_speed_px=args.max_speed, base_radius_px=args.base_radius)
    mapping = lock.fit(frames_full)
    new_full, new6 = remap_frames(frames_full, frames6, mapping)

    new_life = lifetimes(new6)
    d = lock.diagnostics

    print("=" * 64)
    print(f"cache: {c['video']}  {n_frames} frames, pose={c['pose']}")
    print(f"roster target: {args.dancers} dancers "
          f"(max_gap={args.max_gap}, max_speed={args.max_speed}, "
          f"base_radius={args.base_radius})")
    print("-" * 64)
    print(f"raw ByteTrack ids         : {raw_ids}")
    print(f"  raw avg lifetime        : {np.mean(list(raw_life.values())):.1f} frames")
    print(f"stable ids after lock     : {d['stable_ids']}")
    print(f"  stable avg lifetime     : {np.mean(list(new_life.values())):.1f} frames")
    print(f"  fragments merged per id : avg {d['avg_fragments_per_id']:.1f}, "
          f"max {d['max_fragments_per_id']}")
    print(f"  dropped (noise) frags   : {d['dropped_fragments']}")
    print("-" * 64)
    # presence: how many stable dancers visible per frame, on average
    present = [len(ft) for ft in new6]
    print(f"avg dancers tracked/frame : {np.mean(present):.1f} "
          f"(min {min(present)}, max {max(present)})")
    # per-stable-id coverage
    cov = sorted(new_life.items())
    print("per-id frame coverage     : " +
          ", ".join(f"#{k}:{v}" for k, v in cov))
    print("=" * 64)


if __name__ == "__main__":
    main()
