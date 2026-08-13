"""
End-to-end test of the formation-key pipeline on cached tracks, using a
SYNTHETIC key derived from the video (so we can verify the math without the
real chart). Two dancers are deliberately shifted to prove out-of-position
detection + the PDF.
"""
import json, pickle
import numpy as np
from tracking import RosterLock, remap_frames
from formation import (load_key, normalized_positions, avg_positions_in_window,
                       assign_names, grade_positions)
from make_pdf import build_report


def main():
    c = pickle.load(open("full_tracks.pkl", "rb"))
    frames6, frames_full = c["frames6"], c["frames_full"]
    fps = c["fps"]; W, H = c["W"], c["H"]
    n = len(frames6)
    lock = RosterLock(16); mp = lock.fit(frames_full)
    frames_full, frames6 = remap_frames(frames_full, frames6, mp)
    norm = normalized_positions(frames_full, W, H)

    # 3 formation windows across the video
    wins = [(int(n*0.05), int(n*0.20), "formation_1"),
            (int(n*0.40), int(n*0.55), "formation_2"),
            (int(n*0.75), int(n*0.90), "formation_3")]
    roster = [f"P{i}" for i in range(16)]
    formations = []
    ids_sorted = sorted({sid for ft in frames6 for sid in ft})
    name_by_id = {sid: f"P{i}" for i, sid in enumerate(ids_sorted[:16])}
    for s, e, fname in wins:
        actual = avg_positions_in_window(norm, s, e)
        positions = {}
        for sid, nm in name_by_id.items():
            if sid in actual:
                positions[nm] = [round(actual[sid][0], 3), round(actual[sid][1], 3)]
        formations.append({"name": fname,
                           "start": round(s / fps, 1), "end": round(e / fps, 1),
                           "positions": positions})
    # deliberately move P3 in formation_2 and P7 in formation_3 out of spot
    if "P3" in formations[1]["positions"]:
        formations[1]["positions"]["P3"][0] += 0.20
    if "P7" in formations[2]["positions"]:
        formations[2]["positions"]["P7"][1] += 0.18

    key_dict = {"roster": roster, "name_anchor_formation": "formation_1",
                "out_of_position_threshold": 0.07, "formations": formations}
    json.dump(key_dict, open("synthetic_key.json", "w"), indent=2)

    key = load_key("synthetic_key.json", fps)
    name_map = assign_names(key, norm)
    events = grade_positions(key, norm, name_map)
    stats = build_report(key, name_map, events, "formation_report.pdf",
                         fps=fps, video=c["video"])

    print(f"named {len(name_map)} ids; flagged {stats['flagged']} out-of-position, "
          f"{stats['missing']} missing; {stats['pages']} PDF pages")
    for e in events:
        if e.status != "ok":
            print(f"  {e.formation:12} {e.dancer:4} {e.status:16} err={e.error:.3f}")


if __name__ == "__main__":
    main()
