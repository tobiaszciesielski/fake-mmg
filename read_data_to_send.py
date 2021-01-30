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

    np.save('./data/gestures/idle_0.npy', data_dict[0])
    np.save('./data/gestures/fist_1.npy', data_dict[1])
    np.save('./data/gestures/flexion_2.npy', data_dict[2])
    np.save('./data/gestures/extension_3.npy', data_dict[3])
    np.save('./data/gestures/pinch_thumb-index_4.npy', data_dict[6])
    np.save('./data/gestures/pinch_thumb-middle_5.npy', data_dict[7])
    np.save('./data/gestures/pinch_thumb-ring_6.npy', data_dict[8])
    np.save('./data/gestures/pinch_thumb-small_7.npy', data_dict[9])

if __name__ == "__main__":
    format_and_split_sets()
