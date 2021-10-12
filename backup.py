import numpy as np
import cv2
import glob
# from moviepy.editor import VideoFileClip
from mss import mss
from PIL import Image
import time
# from imutils import resize
import imutils

# img = cv2.imread(file , 0)
# img = imutils.resize(img, width=1280)
# cv2.imshow('image' , img)


sct = mss()
mon = sct.monitors[0]

previous_time = 0
sct_img_0 = sct.grab(mon)
h, w = sct_img_0.height, sct_img_0.width


fps = 30               # fps should be the minimum constant rate at which the camera can
# fourcc = "MJPG"       # capture images (with no decrease in speed over time; testing is required)
fourcc = "mp4v"       # capture images (with no decrease in speed over time; testing is required)
# frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used


frameSize = (w, h) # video formats and sizes also depend and vary according to the camera used
# bounding_box = {'top': 0, 'left': 0, 'width': 400, 'height': 300}
# frameSize = (bounding_box["width"], bounding_box["height"]) # video formats and sizes also depend and vary according to the camera used

video_filename = "temp_video.avi"
video_writer = cv2.VideoWriter_fourcc(*fourcc)
video_out = cv2.VideoWriter(video_filename, video_writer, 4, frameSize)

while True :
    try:
        sct_img = sct.grab(mon)
        # sct_img = sct.grab(bounding_box)
        img = sct_img
        # sct_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        # frame = Image.frombytes( 'RGB', (w, h), sct_img.rgb )
        # frame = sct_img
        # frame = np.array(frame)
        # cv2.imshow ('frame', frame)
        # frame = frame.reshape(frame.shape[1], frame.shape[0], frame.shape[2])

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        # frame = cv2.resize(img, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        cv2.imshow ('frame', frame)
        
        # fm_out = imutils.resize(sct_img, width=1280)
        video_out.write(frame)
        
        if (cv2.waitKey(1) & 0xff) == ord( 'q' ) :
            video_out.release()
            cv2.destroyAllWindows()
            break

        fps_custom = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
        previous_time = time.time()
        print(fps_custom)
        
    except KeyboardInterrupt:
        print("kbd interrupt")
        video_out.release()
        cv2.destroyAllWindows()
        break

        