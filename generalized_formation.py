"""
GENERALIZED (no-names) formation grading.

For each (formation, timestamp) the user gives, fit the detected dancers to that
formation's intended SHAPE (homography ICP), then report — spatially, not by name:
  * an overall 'tightness' score (how close the group was to the shape),
  * intended spots that had NO dancer (gaps),
  * dancers sitting well off any intended spot (out of place).
No per-dancer identity or cross-time tracking, so the unreliable parts vanish.
"""
import pickle, json, numpy as np, cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from formation_match_topdown import foot_point, match_hold

OFF_THRESH = 0.09   # normalized stage distance beyond which a dancer is "off"


def positions_at(ff, eff, W, H, t, win=0.3):
    fc = int(round(t * eff)); lo, hi = fc - int(win * eff) - 1, fc + int(win * eff) + 1
    acc = defaultdict(list)
    for fi in range(max(0, lo), min(len(ff), hi + 1)):
        for sid, det in ff[fi].items():
            acc[sid].append(foot_point(det, W, H))
    return [np.median(v, axis=0) for v in acc.values() if v]


def fit_to_formation(P, Q, iters=12):
    """ICP homography of detected points P onto intended spots Q. Returns the
    projected points and the assignment (dancer idx -> spot idx)."""
    m = match_hold({i: P[i] for i in range(len(P))},
                   {j: list(Q[j]) for j in range(len(Q))})
    if m is None:
        return None
    assign = {int(k): int(v) for k, v in m[1].items()}
    Pf = P.astype(np.float32)
    for _ in range(iters):
        pi = list(assign); qi = [assign[i] for i in pi]
        if len(pi) < 4:
            break
        Hm, _ = cv2.findHomography(Pf[pi], Q[qi].astype(np.float32), 0)
        if Hm is None:
            break
        Pt = cv2.perspectiveTransform(Pf.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        C = np.linalg.norm(Pt[:, None, :] - Q[None, :, :], axis=2)
        r, c = linear_sum_assignment(C)
        assign = {int(i): int(j) for i, j in zip(r, c)}
    return Pt, assign


def grade(P, Q):
    Pt, assign = fit_to_formation(P, Q)
    dists = {i: float(np.linalg.norm(Pt[i] - Q[assign[i]])) for i in assign}
    off = [i for i, d in dists.items() if d > OFF_THRESH]
    filled = set(assign.values())
    empty = [j for j in range(len(Q)) if j not in filled]
    avg_dev = float(np.mean(list(dists.values()))) if dists else 1.0
    score = max(0, 100 * (1 - avg_dev / 0.15))   # 0.15 dev ~ score 0
    return Pt, assign, dists, off, empty, avg_dev, score


def main():
    d = pickle.load(open("sr_boxtracks.pkl", "rb"))
    ff, eff, W, H = d["frames_full"], d["eff_fps"], d["W"], d["H"]
    key = json.load(open("formation_key_mov2.json"))
    forms = {int(f["name"].split()[1]): f["positions"] for f in key["formations"]}
    STAMPS = [(2, 8.0), (3, 11.57), (4, 13.33), (7, 16.45), (12, 24.53)]

    with PdfPages("formation_shape_report.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11)); fig.clf()
        fig.text(0.5, 0.93, "Formation Shape Report (generalized)", ha="center",
                 fontsize=18, weight="bold")
        fig.text(0.5, 0.90, "sr.MP4  -  how close the group was to each formation "
                 "(no per-dancer names)", ha="center", fontsize=10, color="gray")
        summ = []
        for fnum, t in STAMPS:
            Q = np.array([forms[fnum][n] for n in forms[fnum]], float)
            P = np.array(positions_at(ff, eff, W, H, t), float)
            Pt, assign, dists, off, empty, dev, score = grade(P, Q)
            summ.append(f"Formation {fnum:>2} @ {t:5.1f}s : score {score:3.0f}/100  "
                        f"| {len(off)} dancers off-spot | {len(empty)} empty spots "
                        f"| avg dev {dev*100:.0f}% of stage")
        fig.text(0.08, 0.83, "\n".join(summ), va="top", fontsize=11,
                 family="monospace")
        fig.text(0.08, 0.83 - 0.03 * len(summ) - 0.03,
                 "Score = how tightly the group matched the intended shape.\n"
                 "'Off-spot' = a dancer well away from any intended position.\n"
                 "'Empty spots' = intended positions with nobody near them.\n"
                 "Generalized shape match - not exact identities.",
                 va="top", fontsize=9, color="#444")
        pdf.savefig(fig); plt.close(fig)

        for fnum, t in STAMPS:
            Q = np.array([forms[fnum][n] for n in forms[fnum]], float)
            P = np.array(positions_at(ff, eff, W, H, t), float)
            Pt, assign, dists, off, empty, dev, score = grade(P, Q)
            fig, ax = plt.subplots(figsize=(8.5, 7))
            ax.set_title(f"Formation {fnum} @ {t:.1f}s   -   score {score:.0f}/100",
                         fontsize=14, weight="bold")
            ax.scatter(Q[:, 0], Q[:, 1], s=280, facecolors="none",
                       edgecolors="#999", linewidths=1.5, label="intended spot")
            for j in empty:
                ax.scatter(Q[j, 0], Q[j, 1], marker="x", s=160, c="#ff7f0e")
            for i, j in assign.items():
                col = "#d62728" if i in off else "#2ca02c"
                ax.scatter(Pt[i, 0], Pt[i, 1], s=70, c=col, zorder=3)
                ax.plot([Q[j, 0], Pt[i, 0]], [Q[j, 1], Pt[i, 1]], c=col,
                        lw=0.8, alpha=0.6)
            ax.set_xlim(-.1, 1.1); ax.set_ylim(1.1, -.1); ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.set_xlabel("stage L -> R      (o=spot  green=on  red=off  x=empty)")
            ax.set_ylabel("back -> front")
            pdf.savefig(fig); plt.close(fig)
    print("wrote formation_shape_report.pdf")
    for s in summ:
        print("  " + s)


if __name__ == "__main__":
    main()
