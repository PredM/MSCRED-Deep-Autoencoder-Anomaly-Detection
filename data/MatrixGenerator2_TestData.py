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

#step_max = config.step_max
gap_time = config.gap_time # time step between each signature matrix
win_size = config.win_size # length of time steps considered for each dimension (depth) of the signature matrix

count_number_of_entries = 0


# scale_n = len(win_size) * len(value_colnames)

# creation of signature matrix
# input: run: a numpy array of sesnor data, numOfRun is the index used in the file name
def generate_signature_matrix_node(run, numOfRun):
    #unpickled_data = unpickle_data(path_to_file + config.filename_pkl_normalised)
    #data = unpickled_data.values
    data = run
    length = data.shape[0]
    sensor_n = data.shape[1]

    data = np.transpose(data)
    max_win_size = max(win_size)
    print("max_win_size: ", max_win_size)
    # Replace zeros with epsilon (value near zero)
    if config.replace_zeros_with_epsilion_in_matrix_generation:
        data = np.where(data == 0, np.finfo(np.float32).eps, data)

    # multi-scale signature matrix generation for every window size of each segment
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        print("generating signature with window " + str(win) + "..." )

        # range starts with min_tine until max_times in steps of gap_time
        for t in range(0, length, gap_time):
            matrix_t = np.zeros((sensor_n, sensor_n))
            # t have to be bigger than the MAX win_size to create a time series from t-win to t
            # so that all sig matrices use the same t !!!

            if t >= max_win_size:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        # Calculate correlation between sensor i and j for a window of size w from t-win to t
                        matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t]) / (win)  # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            matrix_all.append(matrix_t)
        #matrix_data_path = config.directoryname_matrix_data + "test"
        matrix_data_path = config.path + config.directoryname_matrix_data + "_epsi_test/"
        if not os.path.exists(matrix_data_path):
            os.makedirs(matrix_data_path)
        np.save(matrix_data_path + config.filename_matrix_data + str(win)+"_"+str(numOfRun), matrix_all)
        del matrix_all[:]
    print("matrix generation finish!")


# import every pkl file of the normalised data
def main():

    # Import runs
    npzfile = np.load(config.path + "Test_runs.npz", allow_pickle=True)
    #print("npzfile: ", npzfile.files)
    TestRuns = npzfile['arr_0']
    print("TestRuns shape: ", TestRuns.shape)
    #TestRuns.shape[0]
    for i in range(TestRuns.shape[0]): #TestRuns.shape[0]-50
        print('-----------------------------------')
        #i = 49#i+50
        print("TestRuns[i].shape:", TestRuns[i].shape)
        print('import of run: ' + str(i))
        generate_signature_matrix_node(run=TestRuns[i], numOfRun=i)

        print('-----------------------------------')
        print()


if __name__ == '__main__':
    main()
