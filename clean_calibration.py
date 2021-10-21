import time
import tobii_research as tr
from tobii_research_addons import ScreenBasedCalibrationValidation, Point2
from manager import call_eyetracker_manager_example
from calibration import execute

# find device
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
eyetracker = my_eyetracker
print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)


# eyetracker.retrieve_calibration_data()
filename = "saved_calibration.bin"
 
# Save the calibration to file.
with open(filename, "wb") as f:
    calibration_data = eyetracker.retrieve_calibration_data()

    # None is returned on empty calibration.
    if calibration_data is not None:
        print("Saving calibration to file for eye tracker with serial number {0}.".format(eyetracker.serial_number))
        f.write(eyetracker.retrieve_calibration_data())
    else:
        print("No calibration available for eye tracker with serial number {0}.".format(eyetracker.serial_number))

# Read the calibration from file.
with open(filename, "rb") as f:
    calibration_data = f.read()
    # Don't apply empty calibrations.
    # if len(calibration_data) > 0:
    print("Applying calibration on eye tracker with serial number {0}.".format(eyetracker.serial_number))
    eyetracker.apply_calibration_data(calibration_data)