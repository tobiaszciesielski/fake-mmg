import json
import pickle
from typing import Counter
import numpy as np
from sklearn import svm, model_selection, metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import cross_val_score


with open('calibration_params.json', 'r') as file:
    config = json.load(file)

ZERO_PARAM = np.transpose(np.array([config['X_0'], config['Y_0'], config['Z_0']]))
K_PARAM = np.transpose(np.array([config['X_k'], config['Y_k'], config['Z_k']]))

SAMPLES_PER_WINDOW = 150
SAMPLES_PER_STRIDE = 150

GESTURES = [
    0,  # 'idle',                 # 0 -> 0
    1,  # 'fist',                 # 1 -> 1
    2,  # 'flexion',              # 2 -> 2
    3,  # 'extension',            # 3 -> 3
    4,  # 'pinch_thumb-index',    # 4 -> 6
    5,  # 'pinch_thumb-middle',   # 5 -> 7
    6,  # 'pinch_thumb-ring',     # 6 -> 8
    7,  # 'pinch_thumb-small'     # 7 -> 9
] 

def to_MMG_band_format(arrays):
    arrays = np.reshape(arrays, (arrays.shape[0], 8, 9))
    arrays[:, :,[0,1,2,3,4,5]] = arrays[:, :,[3,4,5,0,1,2]]
    return arrays

def calibrate_accelerometer(packet) -> np.ndarray:
  packet[:,0:3] = (packet[:,0:3]-ZERO_PARAM)/K_PARAM
  return packet

def get_accelerometer_features(samples) -> np.ndarray:
    squared = np.square(samples[:,:,0:3])
    sums = np.sum(squared, axis=2, keepdims=True)
    rooted = np.sqrt(sums) - 1 # minus gravity for each feature
    std = np.std(rooted, axis = 0)
    return np.reshape(std, 8)

def split_data_by_gesture(arrays, labels):
    data_dict = {key: [] for key in set(labels)}
    for i in range(len(labels)):
        data_dict[labels[i]].append(arrays[i])
    
    for label, data in data_dict.items():
        data_dict[label] = np.array(data)
    return data_dict

def change_labels_name(dict_to_change, new_shape):
    new_dict = {label: None for label in range(8)}
    new_dict[0] = dict_to_change[0][:new_shape]
    new_dict[1]  = dict_to_change[1][:new_shape]
    new_dict[2]  = dict_to_change[2][:new_shape]
    new_dict[3]  = dict_to_change[3][:new_shape]
    new_dict[4]  = dict_to_change[6][:new_shape]
    new_dict[5]  = dict_to_change[7][:new_shape]
    new_dict[6]  = dict_to_change[8][:new_shape]
    new_dict[7]  = dict_to_change[9][:new_shape]
    return new_dict

def process_data(file_name):
    # Read hdf5 format
    df = pd.read_hdf("./data/{}".format(file_name))

    # Get only valid data
    df = df[df["TRAJ_GT"].notna()]

    # Convert to numpy
    numpy_array = df.to_numpy()

    # Taking specified columns of mmg data 
    arrays = np.array(numpy_array[:,:-16], dtype=float) 
    labels_gt = np.array(numpy_array[:,-1], dtype=int)

    current_gesture = 0
    all_gestures = []
    gesture_data = []
    for i in range(0, arrays.shape[0]-1):
        gesture_data.append(arrays[i])
        current_gesture = labels_gt[i] 

        if current_gesture != labels_gt[i+1]:   # if gesture will change
            if len(gesture_data) > 200:         # and it will be not classification error 
                before_cutting = len(gesture_data)
                to_cut = gesture_data  
                to_cut = to_cut[100:]   # cut first 100 samples
                cutted = to_cut[:-100]  # cut last 100 samples
                all_gestures.append((current_gesture, cutted))    # save current gesture data
            gesture_data=[]     # and clear list for next data


    # Flatten array and tag every sample
    labels = []
    samples = []
    for gesture in all_gestures:
        label, data = gesture
        for sample in data:
            labels.append(label)
            samples.append(sample)
    arrays = np.array(samples)

    # To MMG Band format
    arrays = to_MMG_band_format(arrays)

    # Accelerometer calibration
    arrays = [calibrate_accelerometer(packet) for packet in arrays]

    # Group data by gesture
    splitted_data = split_data_by_gesture(arrays, labels)

    # Find the smallest amount of data for the gesture reshape every gesture data
    new_shape = min([value.shape[0] for key, value in splitted_data.items()])
    
    # Change labels name
    correct_data = change_labels_name(splitted_data, new_shape)
    for key, value in correct_data.items():
        print("Gesture:", key, "Features:", len(value))

    buff = {'features': [], 'labels': []}
    for gesture, data in correct_data.items():
        for i in range(0, data.shape[0], SAMPLES_PER_STRIDE):
            features = get_accelerometer_features(data[i:i+SAMPLES_PER_WINDOW])
            buff['features'].append(features)
            buff['labels'].append(gesture)

    return buff['labels'], buff['features']

def main():
    DATA_SETS = [
        "mmg_gestures-03-sequential-2018-05-11-10-47-44-485.hdf5", #<- this is used for sending test
        "mmg_gestures-03-sequential-2018-06-14-12-16-35-837.hdf5",
        "mmg_gestures-04-sequential-2018-03-28-12-20-57-890.hdf5",
        "mmg_gestures-04-sequential-2018-06-18-14-40-48-463.hdf5",
        "mmg_gestures-04-repeats_long-2018-03-28-12-14-49-698.hdf5",
        "mmg_gestures-04-repeats_long-2018-06-18-14-25-59-114.hdf5",
        "mmg_gestures-04-repeats_long-2018-06-26-12-22-52-038.hdf5",
        "mmg_gestures-04-repeats_short-2018-06-18-14-46-35-634.hdf5",
        "mmg_gestures-04-repeats_short-2018-07-02-11-34-26-385.hdf5",
        "mmg_gestures-04-repeats_short-2018-07-03-14-02-31-148.hdf5",
    ]


    buffer = {'features': [], 'labels': []}
    for i, file in enumerate(DATA_SETS):
        print(f"({i+1}/{len(DATA_SETS)}) Reading data from {file}")
        labels, features = process_data(file)
        buffer['labels'].extend(labels)
        buffer['features'].extend(features)

    arr = [buffer['features'], buffer['labels']]
    print(len(buffer['features']))
    np.save("data", np.array(arr, dtype=object))

    print("Data Saved properly." )
    return
    

if __name__ == "__main__":
    main()
