import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from configuration.Configuration import Configuration

config = Configuration()
num_of_run_to_import = 67


def importSignatureMatrix2(path):
    print('Import Signature Matrix 2')
    curr_data_set = None # One single run
    full_data_set = [] # contains all runs (curr_data_sets)
    for run in range(num_of_run_to_import): #1
        print("imported:",run," of ", num_of_run_to_import-1)
        for w in range(0, len(config.win_size)):
            #curr_data_set_w = np.expand_dims(np.load(path + "matrix_win_"+str(config.win_size[w])+"_51.npy"), axis=3)
            curr_data_set_w = np.expand_dims(np.load(path + "matrix_win_" + str(config.win_size[w]) + "_"+ str(run) + ".npy"),axis=3)
            if w == 0:
                curr_data_set = curr_data_set_w
            else:
                curr_data_set = np.concatenate((curr_data_set, curr_data_set_w), axis=3)
        full_data_set.append(curr_data_set)
    return full_data_set

def main():
    config = Configuration()
    print("Start to import ", num_of_run_to_import)
    #full_data_set = importSignatureMatrix2('../data/matrix_data/')
    # config.path + config.directoryname_matrix_data + "_epsi_train_Failure/"
    full_data_set = importSignatureMatrix2(config.path + config.directoryname_matrix_data + "_epsi_train_Failure/")
    with open('../data/runFailureLabels.txt', 'rb') as f:
        labels = [x.decode('utf8').strip() for x in f.readlines()]
    labels_new = []
    #print("curr_data_set shape: ", full_data_set[0].shape)

    # Generate training sequences of size s
    s = config.step_max # step size
    curr_example = None #np.zeros((s, full_data_set[0].shape[1], full_data_set[0].shape[2], full_data_set[0].shape[3]))
    curr_xTrainData = np.zeros((1, s, full_data_set[0].shape[1], full_data_set[0].shape[2], full_data_set[0].shape[3]))
    for run in range(num_of_run_to_import):
        curr_run = full_data_set[run]
        print("run: ", run)
        for i in range(curr_run.shape[0]-s):  # num_of_examples - s
            if run == i:
                curr_example = curr_run[i:i + s, :, :, :]
                if run == 0:
                    curr_xTrainData[0, :, :, :, :] = curr_example
                else:
                    curr_example = np.expand_dims(curr_example, axis=0)
                    curr_xTrainData = np.concatenate((curr_xTrainData, curr_example), axis=0)
                print("curr_xTrainData 1: ", curr_xTrainData.shape)
                labels_new.append(labels[run])
            else:
                # curr_example = np.concatenate((curr_example, curr_data_set[i:i+s,:,:,:]), axis=0)
                curr_example = curr_run[i:i + s, :, :, :]
                curr_example = np.expand_dims(curr_example, axis=0)
                # print("curr_example: ", curr_example.shape)
                print("curr_xTrainData 2: ", curr_xTrainData.shape)
                curr_xTrainData = np.concatenate((curr_xTrainData, curr_example), axis=0)
                print("curr_xTrainData 3: ", curr_xTrainData.shape)
                labels_new.append(labels[run])
            print(i, " curr_xTrainData shape: ", curr_xTrainData.shape)
            print(i, " Labels new shape: ", len(labels_new))
        #np.save('training_data_set_failure.npy', curr_xTrainData)
        #print("curr_xTrainData shape: ", curr_xTrainData.shape)
        #print("Labels new shape: ", len(labels_new))

    np.save(config.path +'training_data_set_4_epsi_trainWFailure.npy', curr_xTrainData)
    labels_new_arr = np.asarray(labels_new)
    print("labels_new_arr: ",labels_new_arr)
    np.save(config.path +'training_data_set_4_epsi_failure_labels.npy', labels_new_arr)

    print("Final Data Set generation finished!")

if __name__ == '__main__':
    main()
