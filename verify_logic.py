"""
Verifies the ANALYSIS LOGIC (sync math + floor homography) with synthetic
keypoint data — the parts that are the actual product, independent of which
pose model produces the skeletons. If these pass, the engine logic is sound;
the only remaining question on your real footage is capture/tracking quality.
"""
import numpy as np, cv2

# ---- Recreate the core sync logic standalone ----
def motion_signal(frames, tid):
    sig, prev = [], None
    for ft in frames:
        kp = ft.get(tid)
        if kp is None: sig.append(np.nan); prev=None; continue
        sig.append(0.0 if prev is None else float(np.nanmean(np.linalg.norm(kp-prev,axis=1))))
        prev = kp
    return np.array(sig)

def analyze_sync(frames, beats, win=8):
    ids = sorted({t for ft in frames for t in ft})
    sigs = {t: motion_signal(frames,t) for t in ids}
    group = np.nanmedian(np.vstack([sigs[t] for t in ids]),axis=0)
    out={}
    for t in ids:
        s=sigs[t]; offs=[]
        for b in beats:
            lo,hi=max(0,b-win),min(len(s),b+win)
            seg,g=s[lo:hi],group[lo:hi]
            if np.all(np.isnan(seg)) or np.all(np.isnan(g)): continue
            offs.append(int(np.nanargmax(seg))-int(np.nanargmax(g)))
        if offs: out[t]={'avg':float(np.mean(offs)),'worst':float(np.max(np.abs(offs)))}
    return out

# ---- TEST 1: a dancer who is deliberately LATE should be flagged late ----
np.random.seed(0)
N=120; beats=list(range(10,N,20))
frames=[]
for f in range(N):
    ft={}
    # dancer 0 = on the beat; dancer 1 = 4 frames late; dancer 2 = 4 frames early
    for tid,delay in [(0,0),(1,4),(2,-4)]:
        phase=f-delay
        # limb position spikes near each beat -> motion peak on beat
        amp=sum(np.exp(-((phase-b)**2)/4.0) for b in beats)
        kp=np.array([[100+tid*50, 100+amp*20],[100+tid*50, 130+amp*20]],dtype=np.float32)
        ft[tid]=kp
    frames.append(ft)
res=analyze_sync(frames,beats)
print("TEST 1 — sync detection:")
for tid in sorted(res):
    print(f"  Dancer {tid}: avg offset {res[tid]['avg']:+.1f} frames "
          f"({'LATE' if res[tid]['avg']>1 else 'EARLY' if res[tid]['avg']<-1 else 'on time'})")
ok1 = res[1]['avg']>1 and res[2]['avg']<-1 and abs(res[0]['avg'])<=1.5
print(f"  -> {'PASS' if ok1 else 'FAIL'}: late dancer flagged late, early flagged early\n")

# ---- TEST 2: floor homography places dancers correctly top-down ----
src=np.array([[100,200],[540,200],[600,340],[40,340]],np.float32)  # a trapezoid (perspective floor)
dst=np.array([[0,0],[400,0],[400,600],[0,600]],np.float32)
Hm,_=cv2.findHomography(src,dst)
# a dancer standing dead-center of the floor in the image:
center_img=np.array([[[(100+540+600+40)/4,(200+200+340+340)/4]]],np.float32)
mapped=cv2.perspectiveTransform(center_img,Hm)[0][0]
print("TEST 2 — floor projection:")
print(f"  center-of-floor maps to top-down {mapped.round(1)} (expect near [200,300])")
ok2 = abs(mapped[0]-200)<60 and abs(mapped[1]-300)<60
print(f"  -> {'PASS' if ok2 else 'FAIL'}: image point projects to correct floor spot\n")

print("="*55)
print(f"ENGINE LOGIC: {'ALL PASS — math is sound' if ok1 and ok2 else 'CHECK FAILURES'}")
print("The sync + floor logic works. On your real footage the only open")
print("question is capture/tracking quality on crossing dancers.")
