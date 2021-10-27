import math
import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import platform
import glob 
import numpy as np
import logging
import sys
import cv2
import statistics

import atexit

from mss.screenshot import ScreenShot

if sys.platform == 'win32':
    import win32gui, win32ui, win32con, win32api
else:
    logging.warning(f"Screen capture is not supported on platform: `{sys.platform}`")

from collections import namedtuple

import logging
import sys
import re
import traceback

from subprocess import Popen, PIPE, STDOUT

log = logging.getLogger('stdxxx')

class StreamLogger(object):

    def __init__(self, stream, prefix=''):
        self.stream = stream
        self.prefix = prefix
        self.data = ''

    def flush(self):
        pass

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

        if (re.search("FPS\: ", data) is None):
            self.data += data
        # filter 
        re.search("FPS\: ", data)
        tmp = str(self.data)
        if '\x0a' in tmp or '\x0d' in tmp:
            tmp = tmp.rstrip('\x0a\x0d')
            log.info('%s%s' % (self.prefix, tmp))
            self.data = ''


logging.basicConfig(level=logging.DEBUG,
                    filename='text.log',
                    filemode='a')

sys.stdout = StreamLogger(sys.stdout, '[stdout] ')
sys.stderr = StreamLogger(sys.stderr, '[stderr] ')


wrk_path = os.getcwd()
global real_fps
real_fps = list()

class ScreenCapture:
    """
        Captures a fixed  region of the total screen. If no region is given
        it will take the full screen size.
        region_ltrb: Tuple[int, int, int, int]
            Specific region that has to be taken from the screen using
            the top left `x` and `y`,  bottom right `x` and `y` (ltrb coordinates).
    """
    __region = namedtuple('region', ('x', 'y', 'width', 'height'))

    def __init__(self, region_ltrb=None):
        self.region = region_ltrb
        self.hwin = win32gui.GetDesktopWindow()

        # Time management
        self._time_start = time.time()
        self._time_taken = 0
        self._time_average = 0.04

    def __getitem__(self, item):
        return self.screenshot()

    def __next__(self):
        return self.screenshot()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type and isinstance(exc_val, StopIteration):
            return True
        return False

    @staticmethod
    def screen_dimensions():
        """ Retrieve total screen dimensions.  """
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        return left, top, height, width

    @property
    def fps(self):
        return int(1 / self._time_average) * (self._time_average > 0)

    @property
    def region(self):
        return self._region

    @property
    def size(self):
        return self._region.width, self._region.height

    @region.setter
    def region(self, value):
        if value is None:
            self._region = self.__region(*self.screen_dimensions())
        else:
            assert len(value) == 4, f"Region requires 4 input, x, y of left top, and x, y of right bottom."
            left, top, x2, y2 = value
            width = x2 - left + 1
            height = y2 - top + 1
            self._region = self.__region(*list(map(int, (left, top, width, height))))

    def screenshot(self, color=None):
        """
            Takes a  part of the screen, defined by the region.
            :param color: cv2.COLOR_....2...
                Converts the created BGRA image to the requested image output.
            :return: np.ndarray
                An image of the region in BGRA values.
        """
        left, top, width, height = self._region
        hwindc = win32gui.GetWindowDC(self.hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()

        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signed_ints_array = bmp.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype='uint8')
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(self.hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        # This makes sure that the FPS are taken in comparison to screenshots rates and vary only slightly.
        self._time_taken, self._time_start = time.time() - self._time_start, time.time()
        self._time_average = self._time_average * 0.95 + self._time_taken * 0.05

        if color is not None:
            return cv2.cvtColor(img, color)
        return img

    def show(self, screenshot=None):
        """ Displays an image to the screen. """
        image = screenshot if screenshot is not None else self.screenshot()
        cv2.imshow('Screenshot', image)

        if cv2.waitKey(1) & 0xff == ord('q'):
            raise StopIteration
        return image

    def close(self):
        """ Needs to be called before exiting when `show` is used, otherwise an error will occur.  """
        cv2.destroyWindow('Screenshot')

    def scale(self, src: np.ndarray, size: tuple):
        return cv2.resize(src, size, interpolation=cv2.INTER_LINEAR_EXACT)

    def save(self, path, screenshot=None):
        """ Store the current screenshot in the provided path. Full path, with img name is required.) """
        image = screenshot if screenshot is not None else self.screenshot
        cv2.imwrite(filename=path, img=image)


        




class VideoRecorder():


        # Video class based on openCV
        def __init__(self, filename):
            self.videoFilename = wrk_path + "\\video_data\\" + filename + ".avi"
            self.SCREEN_SIZE = 1920, 1080 
            # self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v") # occupies less space
            self.rescale_flag = True
            self.scale_percent = 60 # percent of original size

            if self.rescale_flag:
                width = int(self.SCREEN_SIZE[0] * self.scale_percent / 100)
                height = int(self.SCREEN_SIZE[1] * self.scale_percent / 100)
                self.scaledScreenSize = width, height
                self.out = cv2.VideoWriter(self.videoFilename,self.fourcc, 30.0, (self.scaledScreenSize))
            else:
                self.out = cv2.VideoWriter(self.videoFilename,self.fourcc, 30.0, (self.SCREEN_SIZE))
            
            self.open = True

            
        # Video starts being recorded
        def record(self):

                while(self.open==True):
                    start_time = time.perf_counter()
                    for frame, screenshot in enumerate(ScreenCapture((0, 0, 1920, 1080)), start=1):
                        print(f"\rFPS: {frame / (time.perf_counter() - start_time):3.0f}", end='')
                        real_fps.append(frame / (time.perf_counter() - start_time))
                        # the dimensions are right, since width goes second in an np.array
                        # resize image

                        if self.rescale_flag:
                            width = int(screenshot.shape[1] * self.scale_percent / 100)
                            height = int(screenshot.shape[0] * self.scale_percent / 100)
                            dim = (width, height)

                            # resize image
                            resized = cv2.resize(screenshot, dim, interpolation = cv2.INTER_LINEAR_EXACT)
                            # self.out.write(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                            # cv2.COLOR_BGRA2RGB
                            self.out.write(cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR))
                            # cv2.color_BGR
                            # self.out.write(resized)
                        else:
                            self.out.write(cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR))

        # Finishes the video recording therefore the thread too
        def stop(self):

                if self.open==True:
                        self.open=False
                        self.out.release()
                        print("Stopped video thread")
                        self.videoThreadActive=False
                else: 
                        pass


        # Launches the video recording function using a thread
        def start(self):
                video_thread = threading.Thread(target=self.record, daemon=True)
                video_thread.start()





class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self, filename):
        
        # self.filename = filename
        self.open = True
        self.audioThreadActive=True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = wrk_path + "\\audio_data\\" + filename + ".wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):
        
        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break
        
            
    # Finishes the audio recording therefore the thread too    
    def stop(self):
       
        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
               
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()
            self.audioThreadActive=False
        
        pass
    
    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record, daemon=True)
        audio_thread.start()



def start_AVrecording(filename):

        global video_thread
        global audio_thread

        print("Active threads before start ", threading.active_count())
        video_thread = VideoRecorder(filename)
        audio_thread = AudioRecorder(filename)

        audio_thread.start()
        video_thread.start()

        return filename



def stop_AVrecording(filename):

        audio_thread.stop()
        video_thread.stop()

        print("Active threads AFTER stopped ", threading.active_count())
        # Makes sure the threads have finished
        while (video_thread.videoThreadActive & audio_thread.audioThreadActive):
                time.sleep(1)

        if platform.system() == "Windows":
                ffmpeg_path = wrk_path + "\\ffmpeg-4.4-essentials_build\\ffmpeg-4.4-essentials_build\\bin\\ffmpeg.exe"
        else:
                ffmpeg_path = "ffmpeg"

        merged_filename = wrk_path + "\\merged_data\\" + filename + ".avi"
        # Merging audio and video signal
        print("Before ABS")
        print("Normal recording\nMuxing")


        # mean fps
        if len(real_fps) != 0:
            mean_real_fps = math.ceil(statistics.mean(real_fps))
            print("Real fps*: ", mean_real_fps)
        
        print("Participant: ", merged_filename)

        # cmd = ffmpeg_path + " -ac 1 -channel_layout mono -i " + audio_thread.audio_filename + " -i " +  video_thread.videoFilename +  " -pix_fmt yuv420p " + merged_filename + ".avi" 
        # cmd1 = ffmpeg_path + " -y -i " + video_thread.videoFilename + " -c copy -f h264 " + video_thread.videoFilename + "_output_raw_bitstream.h264"
        # cmd2 = ffmpeg_path + " -y -r " + str(mean_real_fps) + " -i " + video_thread.videoFilename + "_output_raw_bitstream.h264" + " -c copy " + video_thread.videoFilename + "_reframed.avi"
        # cmd3 = ffmpeg_path + " -ac 1 -channel_layout mono -i " + audio_thread.audio_filename + " -i " +  video_thread.videoFilename + "_reframed.avi" " -c:v copy -c:a copy " + merged_filename
        # subprocess.call(cmd, shell=True)

        cmd1_args = " -y -i " + video_thread.videoFilename + " -c copy -f h264 " + video_thread.videoFilename + "_output_raw_bitstream.h264"
        cmd2_args = " -y -r " + str(mean_real_fps) + " -i " + video_thread.videoFilename + "_output_raw_bitstream.h264" + " -c copy " + video_thread.videoFilename + "_reframed.avi"
        cmd3_args = " -ac 1 -channel_layout mono -i " + audio_thread.audio_filename + " -i " +  video_thread.videoFilename + "_reframed.avi" " -c:v copy -c:a copy " + merged_filename
        

        call_and_log_subprocess(ffmpeg_path, cmd1_args)
        call_and_log_subprocess(ffmpeg_path, cmd2_args)
        call_and_log_subprocess(ffmpeg_path, cmd3_args)

        # subprocess.call(cmd1, shell=True)
        # print("\n\ncmd1 is finished\n\n")
        # subprocess.call(cmd2, shell=True)
        # print("\n\ncmd2 is finished\n\n")
        # subprocess.call(cmd3, shell=True)
        # print("\n\ncmd3 is finished\n\n")
        print("..")


# DONE: log subprocess
def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.info('got line from subprocess: %r', line)

def call_and_log_subprocess(ffmpeg_path, cmd_args):
    process = Popen(executable=ffmpeg_path, args=cmd_args, stdout=PIPE, stderr=STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout)
    exitcode = process.wait() # 0 means success
    print("CMD: ", ffmpeg_path + cmd_args)
    print("EXIT CODE: ", exitcode)


# DONE: ensure systemExit is performed only after stop_recording is finished
def on_exit(signal_type):
    print('caught signal:', str(signal_type))
    try:
        stop_AVrecording(filename)
    except BaseException:
        print(traceback.format_exc())

win32api.SetConsoleCtrlHandler(on_exit, True)

# SOLVED: fix color issues (BRGA to BRG)

if __name__== "__main__":

    try:
        ts = int(round(time.time() * 1000))
        filename = str(ts) + "_Default_user"
        start_AVrecording(filename) 
        time.sleep(3600*3)
    except KeyboardInterrupt:
        print("KeyboardInterrupt") 
    except SystemExit:
        print("SystemExit") 
    except BaseException:
        print(traceback.format_exc())
    finally: # dirty fix
        try:
            stop_AVrecording(filename)
        except BaseException:
            print(traceback.format_exc())
        
    # DONE: run it anyway
    # # important to catch stop_recording issues
    # try:
    #     stop_AVrecording(filename)
    # except BaseException:
    #     print(traceback.format_exc())
    # finally:

  
