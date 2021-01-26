import json
import numpy as np


FILES = [
    "MMG_calib_X_down.csv",
    "MMG_calib_X_up.csv",
    "MMG_calib_Y_down.csv",
    "MMG_calib_Y_up.csv",
    "MMG_calib_Z_down.csv",
    "MMG_calib_Z_up.csv",
]

SAMPLES_PER_AXIS = 1000


def compute_calibration_params(axes_data):
    def find_zero(a, b):
        return (a + b) / 2 

    def count_k(a, b):
        return (b - a) / 2

    extreme_values = np.mean(axes_data, axis=1, dtype=float)
    Xmin = extreme_values[0,:,0]
    Xmax = extreme_values[1,:,0]
    Ymin = extreme_values[2,:,1]
    Ymax = extreme_values[3,:,1]
    Zmin = extreme_values[4,:,2]
    Zmax = extreme_values[5,:,2]

    X_zero = find_zero(Xmin, Xmax)
    X_k = count_k(Xmin, Xmax)

    Y_zero = find_zero(Ymin, Ymax)
    Y_k = count_k(Ymin, Ymax)

    Z_zero = find_zero(Zmin, Zmax)
    Z_k = count_k(Zmin, Zmax)

    params = {
        "X_0":X_zero.tolist(),
        "Y_k":Y_k.tolist(),
        "Y_0":Y_zero.tolist(),
        "X_k":X_k.tolist(),
        "Z_0":Z_zero.tolist(),
        "Z_k":Z_k.tolist(),
    }

    with open("calibration_params.json", 'w') as file:
        file.write(json.dumps(params))

def to_MMG_band_format(arrays):
    arrays = np.reshape(arrays, (arrays.shape[0], 8, 9))
    arrays[:, :,[0,1,2,3,4,5]] = arrays[:, :,[3,4,5,0,1,2]]
    return arrays

def load_data() -> np.ndarray:
    axes = np.empty((6,SAMPLES_PER_AXIS,8,9))
    for i in range(len(FILES)):
        axis_data = []
        with open("./data/calibration/"+FILES[i]) as file:
            j = 0
            for line in file.readlines():
                if line[0] == '0': # take only mmg band device
                    if j == SAMPLES_PER_AXIS: break
                    axis_data.append([float(x) for x in line.split(',')[2:]])
                    j+=1

        loaded = np.array(axis_data)
        formatted = to_MMG_band_format(loaded)
        axes[i] = formatted
    
    return np.array(axes, dtype=float)

if __name__ == "__main__":
    axes_data = load_data()
    compute_calibration_params(axes_data)
