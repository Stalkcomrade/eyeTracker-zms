import time
import tobii_research as tr
from manager import call_eyetracker_manager_example
import pandas as pd

# find device
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)

# calibration
# call_eyetracker_manager_example(my_eyetracker.address)

all_gaze_data = []

def timestamp():
    return int(round(time.time() * 1000))


# get data
def gaze_data_callback(gaze_data):
    gaze_data['timestamp'] = timestamp()
    all_gaze_data.append(gaze_data)
    print(gaze_data)

# # TODO: get all needed data logged
def start_gaze_collection():
    my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)       

def stop_gaze_collection(): 
    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


start_gaze_collection()
first_timestamp = str(timestamp())

time.sleep(5)
stop_gaze_collection()

df = pd.DataFrame.from_records(all_gaze_data)
df['mean_pupil_diameter'] = (df['left_pupil_diameter'] + df['right_pupil_diameter']) / 2
df.to_csv('data/all_gaze_data-' + first_timestamp + '.csv', index=False)