"""Given (formation, timestamp) pairs the user identified, fit a homography from
the detected dancers onto that KNOWN formation's charted spots (ICP), read off
names, and vote across formations for a stable identity per tracked dancer."""
import pickle, json, numpy as np, cv2
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
from formation_match_topdown import foot_point, match_hold

d = pickle.load(open("sr_boxtracks.pkl", "rb"))
ff = d["frames_full"]; eff = d["eff_fps"]; W, H = d["W"], d["H"]
key = json.load(open("formation_key_mov2.json"))
forms = {int(f["name"].split()[1]): f["positions"] for f in key["formations"]}

# user-provided (formation number, seconds)
STAMPS = [(2, 8.0), (3, 11.57), (4, 13.33), (7, 16.45), (12, 24.53)]


def positions_at(t, win=0.3):
    fc = int(round(t * eff)); lo, hi = fc - int(win * eff) - 1, fc + int(win * eff) + 1
    acc = defaultdict(list)
    for fi in range(max(0, lo), min(len(ff), hi + 1)):
        for sid, det in ff[fi].items():
            acc[sid].append(foot_point(det, W, H))
    return {sid: np.median(v, axis=0) for sid, v in acc.items() if v}


def match_known(pos, chart, iters=10):
    """ICP with a homography (handles the camera perspective) onto a known chart."""
    m = match_hold(pos, chart)               # similarity init -> assignment
    if m is None:
        return None
    sids = list(pos); names = list(chart)
    P = np.array([pos[s] for s in sids], np.float32)
    Q = np.array([chart[n] for n in names], np.float32)
    nidx = {n: j for j, n in enumerate(names)}
    assign = m[1]
    for _ in range(iters):
        pi = [sids.index(s) for s in assign]; qi = [nidx[assign[s]] for s in assign]
        if len(pi) < 4:
            break
        Hm, _ = cv2.findHomography(P[pi], Q[qi], 0)
        if Hm is None:
            break
        Pt = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        C = np.linalg.norm(Pt[:, None, :] - Q[None, :, :], axis=2)
        r, c = linear_sum_assignment(C)
        assign = {sids[i]: names[j] for i, j in zip(r, c)}
        cost = float(C[r, c].mean())
    return cost, assign


votes = defaultdict(Counter)
print("per-formation match quality:")
for fnum, t in STAMPS:
    pos = positions_at(t)
    res = match_known(pos, forms[fnum])
    if res is None:
        print(f"  F{fnum} @ {t}s: too few dancers"); continue
    cost, mapping = res
    for sid, name in mapping.items():
        votes[sid][name] += 1
    print(f"  F{fnum} @ {t:5.1f}s: {len(pos)} dancers, fit cost={cost:.3f}")

# stable identity = majority vote, no name used twice
name_map, taken = {}, {}
for sid, ctr in sorted(votes.items(), key=lambda kv: -sum(kv[1].values())):
    for name, _ in ctr.most_common():
        if name not in taken:
            name_map[sid] = name; taken[name] = sid; break
# consistency: fraction of a dancer's votes that went to their final name
consist = []
for sid, name in name_map.items():
    tot = sum(votes[sid].values()); consist.append(votes[sid][name] / tot)
print(f"\nnamed {len(name_map)} tracked dancers")
print(f"naming consistency across formations: mean {np.mean(consist)*100:.0f}% "
      f"(1.0 = same name every formation)")
print("stable names:", {f"D{s}": n for s, n in sorted(name_map.items())})
pickle.dump(name_map, open("sr_name_map.pkl", "wb"))
