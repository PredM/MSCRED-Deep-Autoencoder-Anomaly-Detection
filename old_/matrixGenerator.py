from datetime import datetime
from builtins import range, len, str
import numpy as np
import pandas as pd
import os
import sys

# from configuration.Configuration import Configuration
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))
print(os.path.abspath("."))
print(sys.path)

from configuration.Configuration import Configuration

config = Configuration()

step_max = config.step_max
gap_time = config.gap_time
win_size = config.win_size

train_start = config.train_start_point
train_end = config.train_end_point
test_start = config.test_start_point
test_end = config.test_end_point


# scale_n = len(win_size) * len(value_colnames)


# import the normalised and fused data
def unpickle_data(path_to_file):
    print('Unpickle dataset')
    # read the imported dataframe from the saved file
    df: pd.DataFrame = pd.read_pickle(path_to_file)
    return df


# import of normalised data, transposition of data and creation of signature matrix
def generate_signature_matrix_node(path_to_file):
    unpickled_data = unpickle_data(path_to_file + config.filename_pkl_normalised)
    data = unpickled_data.values
    lenght = data.shape[0]
    sensor_n = data.shape[1]

    data = np.transpose(data)

    # multi-scale signature matrix generation for every window size of each segment
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        print("generating signature with window " + str(win) + "...")

        # range starts with min_tine until max_times in steps of gap_time
        for t in range(0, lenght, gap_time):
            matrix_t = np.zeros((sensor_n, sensor_n))
            # t have to be bigger than the win_size to create a time series from t-win to t
            if t >= win_size[len(win_size) - 1]:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t]) / (win)  # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            matrix_all.append(matrix_t)
        matrix_data_path = path_to_file+ config.directoryname_matrix_data
        if not os.path.exists(matrix_data_path):
            os.makedirs(matrix_data_path)
        np.save(matrix_data_path + config.filename_matrix_data + str(win), matrix_all)
        del matrix_all[:]
    print("matrix generation finish!")


# import every pkl file of the normalised data
def main():
    # need one more dimension to manage mulitple "features" for each node or link in each time point, this multiple features can be simply added as extra channels
    nbr_datasets = len(config.datasets)

    for i in range(0, nbr_datasets):
        print('-----------------------------------')
        print('import of dataset: ' + str(i))
        if i == config.no_failure:
            path_to_file = os.path.abspath(".") + config.datasets[i][0][2::] + config.directoryname_training_data
            generate_signature_matrix_node(path_to_file)
            path_to_file = os.path.abspath(".") + config.datasets[i][0][2::] + config.directoryname_eval_data
            generate_signature_matrix_node(path_to_file)
        else:
            path_to_file = os.path.abspath(".") + config.datasets[i][0][2::]
            generate_signature_matrix_node(path_to_file)
        print('-----------------------------------')
        print()


if __name__ == '__main__':
    main()
