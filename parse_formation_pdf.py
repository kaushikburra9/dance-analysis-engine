"""
Parse the formation-chart PDF into a formation_key.json.

Each page = one formation. Dancers render as a numbered dot with a NAME below it;
the gray stage markers (3-18) are numbers with NO name. The pink stage rectangle
is bounded by the marker dots, which we use to normalize every dancer's position
to [0,1] x [0,1]  (x: stage-left->right, y: backstage(top)->audience(bottom)),
matching the front-camera video's orientation.
"""
import argparse, json, re
import fitz  # PyMuPDF


def parse_page(page):
    from collections import defaultdict
    words = page.get_text("words")  # x0,y0,x1,y1,word,...
    nums, names = [], []
    for x0, y0, x1, y1, w, *_ in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if re.fullmatch(r"\d+", w):
            nums.append((cx, cy, w))
        elif re.fullmatch(r"[A-Za-z]+", w):
            names.append((cx, cy, w))
    # assign each NAME token to the single nearest number directly above it
    groups = defaultdict(list)   # number index -> [(x, word), ...]
    for nx, ny, nw in names:
        best, bestd = None, 1e9
        for i, (cx, cy, _) in enumerate(nums):
            dy = ny - cy
            if 6 < dy < 26 and abs(nx - cx) < 22:
                d = abs(nx - cx) + dy
                if d < bestd:
                    bestd, best = d, i
        if best is not None:
            groups[best].append((nx, nw))
    dancers = []
    for i, parts in groups.items():
        parts.sort()
        cx, cy, num = nums[i]
        dancers.append({"num": num, "name": " ".join(p[1] for p in parts),
                        "x": cx, "y": cy})
    return dancers, nums


def stage_box(all_marker_nums):
    xs = [x for x, y, n in all_marker_nums]
    ys = [y for x, y, n in all_marker_nums]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="formation_key_mov2.json")
    ap.add_argument("--threshold", type=float, default=0.07)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    # stage box from ALL bare-number markers across page 1
    p0_dancers, p0_nums = parse_page(doc[0])
    dancer_nums = {d["num"] for d in p0_dancers}
    markers = [(x, y, n) for x, y, n in p0_nums if n not in dancer_nums]
    x0, y0, x1, y1 = stage_box(markers)
    W, H = (x1 - x0) or 1, (y1 - y0) or 1

    formations, roster = [], {}
    for pi in range(doc.page_count):
        dancers, _ = parse_page(doc[pi])
        positions = {}
        for d in dancers:
            nx = round((d["x"] - x0) / W, 3)
            ny = round((d["y"] - y0) / H, 3)
            positions[d["name"]] = [nx, ny]
            roster[d["name"]] = d["num"]
        formations.append({"name": f"Formation {pi+1}", "start": "0:00",
                           "end": "0:00", "positions": positions})

    key = {"roster": sorted(roster, key=lambda n: int(roster[n])),
           "name_anchor_formation": "Formation 1",
           "out_of_position_threshold": args.threshold,
           "_note": "positions normalized to stage box; start/end filled by "
                    "auto-formation-matching at run time",
           "formations": formations}
    json.dump(key, open(args.out, "w"), indent=2)
    print(f"{doc.page_count} formations, {len(roster)} dancers -> {args.out}")
    print("roster:", ", ".join(key["roster"]))
    print(f"dancers per formation: {[len(f['positions']) for f in formations]}")


if __name__ == "__main__":
    main()
