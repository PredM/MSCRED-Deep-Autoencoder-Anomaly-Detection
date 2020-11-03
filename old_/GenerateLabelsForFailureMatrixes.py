import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from configuration.Configuration import Configuration

config = Configuration()
num_of_run_to_import = 66


def importSignatureMatrix2_GenerateLabels(path, labels):
    count_num_of_label_repeatments_per_run = np.zeros(len(labels))
    print('Import Signature Matrix 2')
    curr_data_set = None # One single run
    full_data_set = [] # contains all runs (curr_data_sets)
    new_labels = []
    for run in range(num_of_run_to_import): #num_of_run_to_import
        print("new_labels: ", len(new_labels))
        print("imported:",run," of ", num_of_run_to_import-1)
        for w in range(0, 1): #len(config.win_size)
            #Get first dimension
            curr_data_set_w = np.expand_dims(np.load(path + "matrix_win_" + str(config.win_size[w]) + "_"+ str(run) + ".npy"),axis=3)
            for example_in_current_run in range(curr_data_set_w.shape[0]):  # len(config.win_size)
                print("Label: ", labels[run], ": ",example_in_current_run, "of repeatments: ", curr_data_set_w.shape[0])
                new_labels.append(labels[run])
                #print(np.repeat(labels[run],curr_data_set_w.shape[0]))

    print("new_labels: ", len(new_labels))
    return new_labels

def main():
    ### Generates the labels for the example of the failure data
    config = Configuration()
    print("Start to import ", num_of_run_to_import)
    #labels = np.load('../data/runFailureLabels.txt')
    labels = None
    with open('../data/runFailureLabels.txt', 'rb') as f:
        labels = [x.decode('utf8').strip() for x in f.readlines()]
    new_labels = importSignatureMatrix2_GenerateLabels('../data/matrix_data_train_failure/', labels)
    new_labels = np.asarray(new_labels, dtype=np.str)
    print("new_labels: ", new_labels.shape)
    np.save('runFailureLabels_forMatrixData.npy', new_labels)
    #print("curr_data_set shape: ", full_data_set[0].shape)

if __name__ == '__main__':
    main()
