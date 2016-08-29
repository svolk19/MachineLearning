import pandas as pd
import numpy as np


def csv_reader(filepath, label_index=0):
    # read data into numpy array
    data = pd.read_csv(filepath)
    data.replace('?', -99999, inplace=True)
    for i, data_array in enumerate(data):
        for j, datapoint in enumerate(data_array):
            data[i] = int(i)

    data = np.array(data.as_matrix())

    labels = np.empty(len(data))

    for i, row in enumerate(data):
        labels[i] = row[label_index]

    data = np.delete(data, label_index, axis=1)
    return labels, data

