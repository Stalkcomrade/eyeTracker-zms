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



if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected

                print("in ABS")
                print("Re-encoding")
                cmd = ffmpeg_path + " -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
                subprocess.call(cmd, shell=True)

                print("Muxing")
                cmd = ffmpeg_path + " -ac 1 -channel_layout mono -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
                subprocess.call(cmd, shell=True)

        else:

                print("Normal recording\nMuxing")
                cmd = ffmpeg_path + " -ac 1 -channel_layout mono -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
                subprocess.call(cmd, shell=True)

                print("..")



# def start_video_recording(filename):

#         global video_thread

#         video_thread = VideoRecorder()
#         video_thread.start()

#         return filename


# def start_audio_recording(filename):

#         global audio_thread

#         audio_thread = AudioRecorder()
#         audio_thread.start()

#         return filename

