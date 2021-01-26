import json
import pickle
from typing import Counter
import numpy as np
from sklearn import svm, model_selection, metrics
import pandas as pd


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

def main():
    # train:
    # mmg_gestures-03-sequential-2018-05-11-10-47-44-485.hdf5
    # mmg_gestures-03-sequential-2018-06-14-12-16-35-837.hdf5
    # mmg_gestures-04-sequential-2018-03-28-12-20-57-890.hdf5
    # mmg_gestures-04-repeats_long-2018-03-28-12-14-49-698.hdf5
    # mmg_gestures-04-repeats_short-2018-06-18-14-46-35-634.hdf5

    # test:
    # mmg_gestures-04-sequential-2018-06-18-14-40-48-463.hdf5
    
    df = pd.read_hdf("./data/mmg_gestures-04-sequential-2018-06-18-14-40-48-463.hdf5")
    # print(df.columns)


    # Get only valid data
    df = df[df["TRAJ_GT"].notna()]

    # Convert to numpy
    numpy_array = df.to_numpy()

    # print(Counter(numpy_array[:,-1]))
    # Counter({0.0: 53354, -1.0: 15312, 1.0: 5469, 7.0: 5202, 3.0: 5135, 6.0: 5122, 8.0: 5122, 2.0: 5110, 9.0: 5110})

    # print("\n\n===TAKING MMG DATA COLUMNS===")
    arrays = np.array(numpy_array[:,:-16], dtype=float) 
    labels_gt = np.array(numpy_array[:,-1], dtype=int)
    # print(arrays[0], labels[0], labels_gt[0])

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
                # print(current_gesture, before_cutting, len(cutted))
            gesture_data=[]     # and clear list for next data
    # return

    # flatten array and tag every sample
    labels = []
    samples = []
    for gesture in all_gestures:
        label, data = gesture
        for sample in data:
            labels.append(label)
            samples.append(sample)
    arrays = np.array(samples)

    # print("\n\n===CORRECT DATA COUNT===")
    # print(len(labels))

    # print("\n\n===TO MMG FORMAT===")
    arrays = to_MMG_band_format(arrays)
    # print(arrays[0], labels[0])
    
    # print("\n\n===ACCELEROMETER CALIBRATION===")
    arrays = [calibrate_accelerometer(packet) for packet in arrays]
    # print(arrays[0])
    
    # print("\n\n===GROUP DATA BY GESTURE===")
    splitted_data = split_data_by_gesture(arrays, labels)
    # for key, value in splitted_data.items():
    #     print(key, value.shape)

    # Find the smallest amount of data for the gesture
    # print("\n\n===NEW SHAPE OF EACH GESTURE DATA===")
    new_shape = min([value.shape[0] for key, value in splitted_data.items()])
    # print(new_shape)
    
    # print("\n\n===CHANGE LABEL NAME===")
    correct_data = change_labels_name(splitted_data, new_shape)
    # for key, value in correct_data.items():
    #     print(key, value.shape)

    buffer = {'features': [], 'labels': []}
    for gesture, data in correct_data.items():
        for i in range(0, data.shape[0], SAMPLES_PER_STRIDE):
            features = get_accelerometer_features(data[i:i+SAMPLES_PER_WINDOW])
            buffer['features'].append(features)
            buffer['labels'].append(gesture)
            # print(features)

    for i in range(0, len(buffer["features"]), 10):
        print(buffer["labels"][i], buffer["features"][i])

    # TRAIN WITH ALL DATA

    # loaded_model = pickle.load(open("mmg_model.pkl", 'rb'))
    
    # loaded_model.fit(buffer['features'], buffer['labels'])

    # pickle.dump(loaded_model, open("mmg_model.pkl", 'wb'))


    # # # TRAIN/TEST

    # print("\n\n===TRAIN TEST SPLIT===")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(buffer['features'], buffer['labels'], test_size=0.2)
    # print(len(buffer['labels']), "splited to", len(X_train), "train and", len(X_test), "test")
    
    # # # TRAIN
    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)

    # # # # TEST1 # need probability=true
    # # # print(buffer['features'][724])
    # # # print(buffer['labels'][724])
    # # # y_pred = clf.predict_proba([X_test[724]])[0]
    # # # prob_dist={label: round(prob, 3) for (label, prob) in zip(GESTURES, y_pred)}
    # # # for key, value in prob_dist.items():
    # # #     print(key, value)

    # # TEST2
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)

    # print("\n\n===ACCURACY===")
    print(acc)

if __name__ == "__main__":
    main()
    # acc = [main() for i in range(100)]
    # print(min(acc), max(acc))       

    # 0.5333333333333333
    # 0.06666666666666667