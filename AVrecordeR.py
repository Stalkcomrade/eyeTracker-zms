import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os

import numpy as np
from mss import mss

########################
## JRF
## VideoRecorder and AudioRecorder are two classes based on openCV and pyaudio, respectively. 
## By using multithreading these two classes allow to record simultaneously video and audio.
## ffmpeg is used for muxing the two signals
## A timer loop is used to control the frame rate of the video recording. This timer as well as
## the final encoding rate can be adjusted according to camera capabilities
##

########################
## Usage:
## 
## numpy, PyAudio and Wave need to be installed
## install openCV, make sure the file cv2.pyd is located in the same folder as the other libraries
## install ffmpeg and make sure the ffmpeg .exe is in the working directory
##
## 
## start_AVrecording(filename) # function to start the recording
## stop_AVrecording(filename)  # "" ... to stop it
##
##
########################

sct = mss()
mon = sct.monitors[0]

previous_time = 0
sct_img_0 = sct.grab(mon)
scaling = 2
h, w = sct_img_0.height, sct_img_0.width
h = int(h / scaling)
w = int(w / scaling)


fps = 30               # fps should be the minimum constant rate at which the camera can
fourcc = "mp4v"       # capture images (with no decrease in speed over time; testing is required)

frameSize = (w, h) # video formats and sizes also depend and vary according to the camera used

video_filename = "temp_video.avi"
video_writer = cv2.VideoWriter_fourcc(*fourcc)
video_out = cv2.VideoWriter(video_filename, video_writer, 6, frameSize)


class VideoRecorder():


        # Video class based on openCV
        def __init__(self):

                self.sct = mss()
                self.videoThreadActive=True
                self.mon = self.sct.monitors[0]
                self.sct_img_0 = self.sct.grab(mon)
                self.scaling = 2
                self.h = int(self.sct_img_0.height / self.scaling)
                self.w = int(self.sct_img_0.width / self.scaling)
                self.frameSize = (self.w, self.h) 
                self.open = True
                # self.device_index = 0
                self.fps = 6               # fps should be the minimum constant rate at which the camera can
                self.fourcc = "mp4v"       # capture images (with no decrease in speed over time; testing is required)
                # self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
                self.video_filename = "temp_video.avi"
                # self.video_cap = cv2.VideoCapture(self.device_index)
                self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
                self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
                self.frame_counts = 1
                self.start_time = time.time()


        # Video starts being recorded
        def record(self):

#               counter = 1
                timer_start = time.time()
                timer_current = 0

                while(self.open==True):
                        # ret, video_frame = self.video_cap.read()
                        # if (ret==True):
                        sct_img = self.sct.grab(mon)
                        # Convert the screenshot to a numpy array
                        # Convert it from BGR(Blue, Green, Red) to RGB(Red, Green, Blue)
                        # fps_custom = 'fps: %.1f' % ( 1./( time.time() - timer_current ))
                        # print(fps_custom)
                        # previous_time = time.time()
                        
                        frame = np.array(sct_img)
                        frame = cv2.resize(frame, (int(frame.shape[1]/scaling), int(frame.shape[0]/scaling)))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        # cv2.imshow ('frame', frame)
                        # video_out.write(frame)
                        self.video_out.write(frame)
                        self.frame_counts += 1
                        print("frame ", self.frame_counts)
                        timer_current = time.time() - timer_start
                        time.sleep(0.16)
                        
                        # gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow('video_frame', gray)
                        # cv2.waitKey(1)
                        # 0.16 delay -> 6 fps


        # Finishes the video recording therefore the thread too
        def stop(self):

                if self.open==True:
                        self.open=False
                        self.video_out.release()
                        print("Stopped video thread")
                        self.videoThreadActive=False
                        # self.video_cap.release()
                        # cv2.destroyAllWindows()
                else: 
                        pass


        # Launches the video recording function using a thread
        def start(self):
                video_thread = threading.Thread(target=self.record, daemon=True)
                video_thread.start()





class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self):
        
        self.open = True
        self.audioThreadActive=True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
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
        video_thread = VideoRecorder()
        audio_thread = AudioRecorder()

        audio_thread.start()
        video_thread.start()

        return filename




def start_video_recording(filename):

        global video_thread

        video_thread = VideoRecorder()
        video_thread.start()

        return filename


def start_audio_recording(filename):

        global audio_thread

        audio_thread = AudioRecorder()
        audio_thread.start()

        return filename




def stop_AVrecording(filename):

        audio_thread.stop()
        frame_counts = video_thread.frame_counts
        elapsed_time = time.time() - video_thread.start_time
        recorded_fps = frame_counts / elapsed_time
        print("total frames " + str(frame_counts))
        print("elapsed time " + str(elapsed_time))
        print("recorded fps " + str(recorded_fps))
        video_thread.stop()

        print("Active threads AFTER stopped ", threading.active_count())
        # Makes sure the threads have finished
        # while threading.active_count() > 1:
        #         time.sleep(1)
        while (video_thread.videoThreadActive & audio_thread.audioThreadActive):
                time.sleep(1)


#        Merging audio and video signal
        print("Before ABS")
        if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected

                print("in ABS")
                print("Re-encoding")
                cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
                subprocess.call(cmd, shell=True)

                print("Muxing")
                cmd = "ffmpeg -ac 1 -channel_layout mono -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
                subprocess.call(cmd, shell=True)

        else:

                print("Normal recording\nMuxing")
                cmd = "ffmpeg -ac 1 -channel_layout mono -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
                subprocess.call(cmd, shell=True)

                print("..")




# Required and wanted processing of final files
def file_manager(filename):

        local_path = os.getcwd()

        if os.path.exists(str(local_path) + "/temp_audio.wav"):
                os.remove(str(local_path) + "/temp_audio.wav")

        if os.path.exists(str(local_path) + "/temp_video.avi"):
                os.remove(str(local_path) + "/temp_video.avi")

        if os.path.exists(str(local_path) + "/temp_video2.avi"):
                os.remove(str(local_path) + "/temp_video2.avi")

        if os.path.exists(str(local_path) + "/" + filename + ".avi"):
                os.remove(str(local_path) + "/" + filename + ".avi")




if __name__== "__main__":
# try:

        filename = "Default_user"
        file_manager(filename)
        start_AVrecording(filename)     

# except KeyboardInterrupt:

        time.sleep(5)
        print("kbd interrupt")
        stop_AVrecording(filename)
        print("Done")
