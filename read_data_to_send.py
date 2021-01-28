from numpy.core.arrayprint import array2string
import pandas as pd
import numpy as np

def to_MMG_band_format(arrays):
    arrays = np.reshape(arrays, (arrays.shape[0], 8, 9))
    arrays[:, :,[0,1,2,3,4,5]] = arrays[:, :,[3,4,5,0,1,2]]
    return arrays


def format_and_split_sets():        
    df = pd.read_hdf("./data/mmg_gestures-03-sequential-2018-05-11-10-47-44-485.hdf5")

    numpy_array = df.to_numpy()
    arrays, labels = numpy_array[:,:-16], numpy_array[:,-1]

    arrays = to_MMG_band_format(arrays)
    
    arrays = arrays.tolist()
    labels = labels.tolist()
    
    data_dict = {key: [] for key in set(labels)}
    for i in range(len(labels)):
        data_dict[labels[i]].append(arrays[i])
        
    for label, data in data_dict.items():
        data_dict[label] = np.array(data)

    # GESTURES = [
    #     'idle',                 # 0 -> 0
    #     'fist',                 # 1 -> 1
    #     'flexion',              # 2 -> 2
    #     'extension',            # 3 -> 3
    #     'pinch_thumb-index',    # 4 -> 6
    #     'pinch_thumb-middle',   # 5 -> 7
    #     'pinch_thumb-ring',     # 6 -> 8
    #     'pinch_thumb-small'     # 7 -> 9
    # ] 

    np.save('./data/gestures/idle_0.npy', data_dict[0])
    np.save('./data/gestures/fist_1.npy', data_dict[1])
    np.save('./data/gestures/flexion_2.npy', data_dict[2])
    np.save('./data/gestures/extension_3.npy', data_dict[3])
    np.save('./data/gestures/pinch_thumb-index_4.npy', data_dict[6])
    np.save('./data/gestures/pinch_thumb-middle_5.npy', data_dict[7])
    np.save('./data/gestures/pinch_thumb-ring_6.npy', data_dict[8])
    np.save('./data/gestures/pinch_thumb-small_7.npy', data_dict[9])

    # Counter({0.0: 57603, -1.0: 14002, 3.0: 4801, 1.0: 4800, 2.0: 4800, 6.0: 4800, 7.0: 4800, 8.0: 4800, 9.0: 4800})

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(arrays, labels, test_size=0.3)
    # print(X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0])

    # train_set = create_json(X_train, y_train)
    # test_set = create_json(X_test, y_test)

    # with open('train_set.json', "w") as file:
    #     file.write(train_set)

    # with open('test_set.json', "w") as file:
    #     file.write(test_set)

if __name__ == "__main__":
    format_and_split_sets()