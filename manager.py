import os
import subprocess
import glob
import tobii_research as tr


def call_eyetracker_manager_example(device_addres):
    try:
        ETM_PATH = ''
        DEVICE_ADDRESS = device_addres
        # FIXME: locate more reliably
        ETM_PATH = glob.glob(os.environ["LocalAppData"] + "/Programs" + "/TobiiTechConfigurationApplication/TobiiProEyeTrackerManager.exe")[0]


        eyetracker = tr.EyeTracker(DEVICE_ADDRESS)

        mode = "displayarea"

        #etm_p = subprocess.Popen([ETM_PATH,
        #                          "--device-address=" + eyetracker.address,
        #                          "--mode=" + mode],
        #                         stdout=subprocess.PIPE,
        #                         stderr=subprocess.PIPE,
        #                         shell=False)

        etm_p = subprocess.Popen([ETM_PATH,
                                  "--device-address=" + eyetracker.address,
                                  "--mode=" + "usercalibration"],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 shell=False)

        stdout, stderr = etm_p.communicate()  # Returns a tuple with (stdout, stderr)

        if etm_p.returncode == 0:
            print("Eye Tracker Manager was called successfully!")
        else:
            print("Eye Tracker Manager call returned the error code: " + str(etm_p.returncode)) 
            errlog = stdout  # On Windows ETM error messages are logged to stdout
           

            for line in errlog.splitlines():
                if line.startswith("ETM Error:"):
                    print(line)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    call_eyetracker_manager_example()