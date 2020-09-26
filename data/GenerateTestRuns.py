import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))




def main():

    # import data sets
    # 4 sec windows (overlapping) with 4ms sampling

    test_window_times = np.load("../../PredMSiamNN2/data/training_data/test_window_times.npy")  # labels of the training data
    y_test = np.load("../../PredMSiamNN2/data/training_data/test_labels.npy")  # labels of the training data
    x_test_features = np.load("../../PredMSiamNN2/data/training_data/test_features.npy")  # data streams to train a machine learning model

    print("Test window times shape: ", test_window_times.shape)
    print("Feature Window shape: ", x_test_features.shape)
    print("test_window_times: ", test_window_times[0,:])
    for entry in range(test_window_times.shape[0]):
        # Convert start date from string to datetime
        startDatum = str(test_window_times[entry, 0])
        startDatum_startPos = startDatum.find("(")
        startDatum_endPos = startDatum.find(")")
        startDatum = startDatum[(startDatum_startPos + 1):startDatum_endPos]
        endDatum = str(test_window_times[entry, 2])
        endDatum_startPos = endDatum.find("(")
        endDatum_endPos = endDatum.find(")")
        endDatum = endDatum[(endDatum_startPos + 1):endDatum_endPos]
        try:
            startDatum_dt = datetime.strptime(startDatum, '%Y%m%d %H:%M:%S')
            test_window_times[entry, 0] = startDatum_dt
            endDatum_dt = datetime.strptime(endDatum, '%Y%m%d %H:%M:%S')
            test_window_times[entry, 2] = endDatum_dt
            #print(entry, x)
        except ValueError:
            print("An exception occurred")


    # Concat labels and window times
    y_test = np.reshape(y_test,(len(y_test),1))
    print(y_test.shape)
    print(test_window_times.shape)
    train_data = np.concatenate((test_window_times, y_test), axis=1)
    print(train_data.shape)
    train_data = np.delete(train_data, 1, 1)
    print(train_data[0,:])

    # remove entries with failure
    #train_data = np.delete(train_data, np.argwhere(train_data[:,2] != 'no_failure'),0)
    print("Test", train_data.shape)

    # Find trajectories for training
    tractory_counter = 0
    curr_tractory_instances = 0
    curr_trjactory_indices = []
    runs = []
    runFailureLabels = []
    runFailureTimes = []
    for entry in range(train_data.shape[0]):
        #print(train_data[entry,:])
        currEndDate = train_data[entry,1]
        found = np.argwhere(train_data[:,0] == currEndDate)
        #print("Found:",found, "enddate:", currEndDate)
        if found.size == 0 and curr_tractory_instances > 4:
            #print("new trajectory: ", curr_tractory_instances)
            #print("indicies: ", curr_trjactory_indices)
            runs.append(curr_trjactory_indices)
            tractory_counter = tractory_counter +1
            curr_tractory_instances = 0
            curr_trjactory_indices = []
            runFailureLabels.append(train_data[entry,2])
            runFailureTimes.append(train_data[entry,1])
        else:
            #print(entry)
            curr_trjactory_indices.append(entry)
            curr_tractory_instances = curr_tractory_instances +1

    print("tractory_counter: ", tractory_counter)
    print("runs: ", runs)
    print("runs length: ", len(runs))

    with open('runTestLabels.txt', 'w') as filehandle:
        for listitem in runFailureLabels:
            filehandle.write('%s\n' % listitem)
    with open('runTestTimes.txt', 'w') as filehandle:
        for listitem in runFailureTimes:
            filehandle.write('%s\n' % listitem)

    # Extract windows or better say cutting out overlapping windows in order to create a run
    runs_x_features = []
    for run in range(len(runs)):
        len_curr_run = len(runs[run])
        curr_run_data_np = np.zeros((1000,61))
        firstentry = runs[run][0]
        print("firstentry:", firstentry, "lenght: ", len_curr_run)

        print("curr_run_data_np: ", curr_run_data_np.shape)
        #for window_idx in runs[run]:
        for index, window_index in enumerate(runs[run]):
            #print(index, window_index)
            if index % 4 == 0:
                if index == 0:
                    curr_run_data_np = x_test_features[window_index, :, :]
                else:
                    curr_run_data_np = np.concatenate((curr_run_data_np, x_test_features[window_index, :, :]), axis=0)

        print("curr_run_data_np shape: ", curr_run_data_np.shape)
        runs_x_features.append(curr_run_data_np)


    # Print Save:
    np.savez("Test_runs.npz", runs_x_features)
    npzfile = np.load("Test_runs.npz")


if __name__ == '__main__':
    main()
