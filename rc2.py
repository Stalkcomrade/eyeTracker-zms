import numpy as np
import cv2
from mss import mss
import time

sct = mss()
mon = sct.monitors[0]

previous_time = 0
sct_img_0 = sct.grab(mon)
scaling = 3
h, w = sct_img_0.height, sct_img_0.width
h = int(h / scaling)
w = int(w / scaling)
# mon_cust = mon
# mon_cust["height"] = h
# mon_cust["width"] = w
# print("df", " ", (h,w))


fps = 30               # fps should be the minimum constant rate at which the camera can
fourcc = "mp4v"       # capture images (with no decrease in speed over time; testing is required)

frameSize = (w, h) # video formats and sizes also depend and vary according to the camera used

video_filename = "temp_video.avi"
video_writer = cv2.VideoWriter_fourcc(*fourcc)
video_out = cv2.VideoWriter(video_filename, video_writer, 6, frameSize)

# print(sct_img_0)
# print("df", " ", (h,w))

timer_start = time.time()
timer_current = 0
frame_counts = 1

while True :
    try:
        if True:
            # sct_img = sct.grab(mon_cust)
            sct_img = sct.grab(mon)
            
            # Convert the screenshot to a numpy array
            # Convert it from BGR(Blue, Green, Red) to RGB(Red, Green, Blue)
            fps_custom = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
            previous_time = time.time()
            frame_counts =+ 1
            print(fps_custom)
            
            # frame = np.array(sct_img)
            frame = np.asarray(sct_img)
            frame = cv2.resize(frame, (int(frame.shape[1]/scaling), int(frame.shape[0]/scaling)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow('frame', frame)

            video_out.write(frame)
            timer_current = time.time() - timer_start
            # time.sleep(0.16)
            
            # if (cv2.waitKey(1) & 0xff) == ord( 'q' ) :
            #     video_out.release()
            #     cv2.destroyAllWindows()
            #     break
        
    except KeyboardInterrupt:
        print("kbd interrupt")
        video_out.release()
        cv2.destroyAllWindows()
        break

        