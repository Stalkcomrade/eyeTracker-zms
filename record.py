import time

import cv2
import numpy as np
from mss import mss


def record(name):
    with mss() as sct:
        # mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
        mon = sct.monitors[0]
        # print(sct.monitors)
        name = name + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        desired_fps = 30.0
        out = cv2.VideoWriter(name, fourcc, desired_fps,
                              (mon['width'], mon['height']))
        last_time = 0
        while True:
            img = sct.grab(mon)
            cv2.imshow('test', np.array(img))
            if time.time() - last_time > 1./desired_fps:
                last_time = time.time()
                destRGB = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
                out.write(destRGB)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


record("Video")

