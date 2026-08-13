# Generates a tiny synthetic clip so we can prove the pipeline runs end-to-end.
# (Stick figures, not real dancers — MediaPipe may not detect them, which is FINE;
# the point is to confirm the code path executes without crashing.)
import cv2, numpy as np
W,H,fps,secs = 640,360,15,3
vw = cv2.VideoWriter('test_clip.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
for f in range(fps*secs):
    img = np.full((H,W,3), 30, np.uint8)
    for i in range(3):
        x = 120 + i*180 + int(20*np.sin(f*0.4 + i))
        y = 180 + int(15*np.sin(f*0.4))
        cv2.circle(img,(x,y-40),18,(200,200,200),-1)      # head
        cv2.line(img,(x,y-22),(x,y+30),(200,200,200),4)    # torso
        cv2.line(img,(x,y),(x-25,y+15+int(10*np.sin(f*0.5))),(200,200,200),4)  # arm
        cv2.line(img,(x,y),(x+25,y+15-int(10*np.sin(f*0.5))),(200,200,200),4)
        cv2.line(img,(x,y+30),(x-15,y+60),(200,200,200),4) # legs
        cv2.line(img,(x,y+30),(x+15,y+60),(200,200,200),4)
    vw.write(img)
vw.release()
print("wrote test_clip.mp4")
