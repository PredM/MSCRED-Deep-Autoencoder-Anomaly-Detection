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
    #print("data shape: ", data.shape)
    max_win_size = max(win_size)
    #print("max_win_size: ", max_win_size)
    # Replace zeros with epsilon (value near zero)
    if config.replace_zeros_with_epsilion_in_matrix_generation:
        data = np.where(data == 0, np.finfo(np.float32).eps, data)

    examples = length / max_win_size
    #print("examples per given example: ", examples)
    matrix_t_ = np.zeros((int(examples),len(win_size), sensor_n, sensor_n))
    #print("matrix_t_ shape: ", matrix_t_.shape)
    # multi-scale signature matrix generation for every window size of each segment
    for w in range(len(win_size)):
        #print("w: ",w,"in:", len(win_size))
        matrix_all = []
        win = win_size[w]
        #print("generating signature with window " + str(win) + "..." )

        # range starts with min_tine until max_times in steps of gap_time
        count_curr_example = 0
        for t in range(0, length, gap_time):
            #print("Generating at time point",t,"with win size:",win)
            matrix_t = np.zeros((sensor_n, sensor_n))
            # t have to be bigger than the MAX win_size to create a time series from t-win to t
            # so that all sig matrices use the same t !!!
            t_ = t + gap_time # to start from zero ...
            #print("data[i, t_ - win:t_]: ",data[60, t_ - win:t_][:3])
            if t_ >= max_win_size:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        # Calculate correlation between sensor i and j for a window of size w from t-win to t
                        matrix_t[i][j] = np.inner(data[i, t_ - win:t_], data[j, t_ - win:t_]) / (win)  # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
                #matrix_all.append(matrix_t)
                #print("matrix_t: ", matrix_t)

            matrix_t_[count_curr_example, w, :, :] = matrix_t
            count_curr_example = count_curr_example +1
        #matrix_data_path = config.directoryname_matrix_data + "test"
        #matrix_data_path = config.path + config.directoryname_matrix_data + "_epsi_test/"
        #matrix_data_path = config.directoryname_matrix_data + "_epsi_test/"
        #if not os.path.exists(matrix_data_path):
        #    os.makedirs(matrix_data_path)
        #np.save(matrix_data_path + config.filename_matrix_data + str(win)+"_"+str(numOfRun), matrix_all)
        #del matrix_all[:]
    return matrix_t_




    print("matrix generation finish!")


# import every pkl file of the normalised data
def main():
    import pickle

    a_file = open("../../../../data/pklein/Datensatz2/training_data/raw_data/aux_df_test.pkl", "rb")
    y_labels = pickle.load(a_file)
    a_file.close()
    print("y_labels: ", y_labels['label'].values)

    y_labels = y_labels['label'].values
    np.save(config.path + '/test_labels.npy', y_labels, )

    y_labels = np.load(config.path + '/test_labels.npy',allow_pickle=True)
    print("y_labels: ", y_labels)
    print("where no_failure: ", len(np.argwhere(y_labels != 'no_failure')))
    print(sds)
    # Import examples
    #y_labels = np.load("../../../../data/pklein/PredMSiamNN/data/training_data/valid_labels_new2.npy")  # labels of the training data
    #y_labels = np.load("../../../../data/pklein/Datensatz2/training_data/raw_data/x_train_val.npy")  # labels of the training data

    #x_features = np.load("../../../../data/pklein/PredMSiamNN/data/training_data/valid_features_new2.npy")  # data streams to train a machine learning model
    x_features = np.load("../../../../data/pklein/Datensatz2/training_data/raw_data/x_test.npy")  # data streams to train a machine learning model
    print("y_labels shape: ", y_labels.shape)
    print("x_features shape: ", x_features.shape)

    example_dim = x_features.shape[0]

    generated_example = np.zeros((example_dim,4,len(win_size), x_features.shape[2], x_features.shape[2]))
    #Shape (training samples, #sigmatrixes, windows, sensors, sensors)
    cnt=0
    for i in range(x_features.shape[0]): #TestRuns.shape[0]-50
        print(i,"-", y_labels[i])
        print('-----------------------------------')
        #if not y_labels[i] == "no_failure":
        sig_mat = generate_signature_matrix_node(run=x_features[i], numOfRun=i)
        print(i,"- sig_mat: ", sig_mat.shape)
        generated_example[cnt, :,:,:,:] = sig_mat
        cnt = cnt+1
        print('-----------------------------------')
        print()
    np.save(config.path + 'sig_mat_test.npy', generated_example)

if __name__ == '__main__':
    main()
