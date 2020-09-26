import datetime
import json

import numpy as np
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath("."))
from configuration.Configuration import Configuration

config = Configuration()

# split sensors in two groups depending on the summed reconstruction error of columns
# border between the groups is the maximal distance between two neighboured summed reconstruction errors which
def find_max_distance(resid_sorted):
    max_dif = 0
    value_max_dif = 0
    for i in range(1, resid_sorted.shape[0]):
        dif = abs(resid_sorted[i] - resid_sorted[i - 1])
        if dif > max_dif:
            max_dif = dif
            value_max_dif = resid_sorted[i]

    return value_max_dif

# determines the threshold for anomaly identification
# values which are higher as this value increment the anomaly Score
def determinethreshold():
    threshold_max = []
    threshold_min = []
    path = os.path.abspath(".") + config.datasets[config.validation_data][0][2::] + config.directoryname_eval_data + config.directoryname_NNresults

    print('Start threshold calculation for dataset ')
    # determine minimum and maximum of datasets
    for files in os.listdir(path):
        if files.find(config.reconstruction_error) != (-1):
            resid = np.load(path + files)
            resid_sorted = np.sort(resid, axis=None)
            length = len(resid_sorted) - 1
            threshold_max.append(resid_sorted[length])
            threshold_min.append(resid_sorted[0])
    threshold_min_avg = 0
    threshold_max_avg = 0

    #calculate average of minimum and maximum
    for i in range(0,len(threshold_min)):
        threshold_min_avg += threshold_min[i]
        threshold_max_avg += threshold_max[i]

    threshold_min_avg = float(threshold_min_avg) / float(len(threshold_min))
    threshold_max_avg = float(threshold_max_avg) / float(len(threshold_max))

    return threshold_max_avg, threshold_min_avg


# determine average of reconstruction error per sensor
def min_max_SensorError():
    path = os.path.abspath(".") + config.datasets[config.validation_data][0][2::] + config.directoryname_NNresults
    minError = []
    maxError = []
    count = 0
    for files in os.listdir(path):
        if files.find(config.reconstruction_error) != (-1):
            resid = np.load(path + files)
            if minError == []:
                minError = np.zeros(resid.shape[2])
            if maxError == []:
                maxError = np.zeros(resid.shape[2])
            for i in range(0, resid.shape[0]):
                for j in range(0, resid.shape[3]):
                    for k in range(0, resid.shape[2]):
                        count += 1
                        for l in range(0, resid.shape[1]):
                            if resid[i, l, k, j] < 0 and minError[l] > (resid[i, l, k, j]):
                                minError[l] = (resid[i, l, k, j])
                            elif maxError[l] < resid[i, l, k, j]:
                                maxError[l] = resid[i, l, k, j]
    return maxError, minError

# identify the root cause for every anomaly event
def anomalyDiagnosis(anomalyEvents, resid):
    m = 0
    root_cause = []
    for anomalyEvents_Matrix in anomalyEvents:
        root_cause_matrix = []
        for anomalyEvent in anomalyEvents_Matrix:
            matrix = resid[anomalyEvent, :, :, m]
            sum = np.zeros((matrix.shape[0]))
            for i in range(0, matrix.shape[0]):
                for j in range(0, matrix.shape[1]):
                    if matrix[i, j] < 0:
                        sum[i] += (matrix[i, j] * (-1))
                    else:
                        sum[i] += matrix[i, j]
            sum_sorted = np.sort(sum)
            max_dif_value = find_max_distance(sum_sorted)
            root_cause_time = []
            for i in range(0, matrix.shape[0]):
                if sum[i] >= max_dif_value:
                    root_cause_time.append(i)
            root_cause_matrix.append(root_cause_time)
        m += 1
        root_cause.append(root_cause_matrix)

    return root_cause

# load pickled data
def unpickle_data(path):
    # read the imported dataframe from the saved file
    df: pd.DataFrame = pd.read_pickle(path)
    return df

# get the end of an anomaly
def getEnd(t, path):
    #t = t*10
    df = unpickle_data(path)
    start_time = pd.to_datetime(df.index.min())
    batch_size = config.batch_size
    seconds = int(config.win_size[len(config.win_size) - 1]) * int(config.fusion_interval) + int(config.gap_time) * \
              batch_size * int(config.fusion_interval) + int(config.gap_time) * t * int(config.fusion_interval)
    time = start_time + datetime.timedelta(milliseconds=seconds)
    return time

# get the start of an anomaly
def getStart(t, path):
    #t = t * 10
    df = unpickle_data(path)
    start_time = pd.to_datetime(df.index.min())
    seconds = int(config.win_size[len(config.win_size) - 1]) * int(config.fusion_interval) + int(config.gap_time) \
              * t * int(config.fusion_interval)
    time = start_time + datetime.timedelta(milliseconds=seconds)
    return time

# calculates the length of an anomaly in seconds
def getDif(start, end):
    return (end - start).total_seconds()

# defines matrix where the anomaly first and last appears
def getDuration(anomalyEvent, rootCause, resid, avgMax, avgMin):
    m = 0
    time_final = []
    for matrix in anomalyEvent:
        t = 0
        times_list = []
        for time in matrix:
            roots = rootCause[m][t]
            time_begin = time
            time_end = time
            for root in roots:
                contin = 1
                #decrement begin time until there exists no reconstruction error of a root cause sensor which is bigger than maxavg[root]
                while contin == 1 and time_begin > 0:
                    if np.max(resid[time_begin-1, root, :, m]) > avgMax[root] or np.min(resid[time_begin-1, root, :, m]) < avgMin[root]:
                        time_begin = time_begin-1
                    elif np.max(resid[time_begin-1, :, root, m]) > avgMax[root] or np.min(resid[time_begin-1, :, root, m]) < avgMin[root]:
                        time_begin = time_begin-1
                    else:
                        contin = 0
                contin = 1
                # increment end until average reconstruction error of root cause sensors decreases
                while contin == 1 and time_end < (resid.shape[0]-1):
                    if np.average(np.absolute(resid[time_end + 1, root, :, m])) >= np.average(np.absolute(resid[time_end, root, :,m])):
                        time_end = time_end + 1
                    elif np.average(np.absolute(resid[time_end + 1, :, root, m])) >= np.average(np.absolute(resid[time_end, :, root, m])):
                        time_end = time_end + 1
                    else:
                        contin = 0
            t = t+1
            ti=[]
            ti.append(time_begin)
            ti.append(time_end)
            times_list.append(ti)
        m = m+1
        time_final.append(times_list)
    return time_final

# fuse the start and end times with same length of time series
def fuseRootTime (root_cause, times, anomalyScore):
    time = []
    roots = []
    for i in range(0, len(root_cause)):
        ti = []
        ro = []
        j = 0
        while j < len(root_cause[i]):
            tmp_j = j
            start = times[i][j][0]
            end = times[i][j][1]
            if (j < len(times[i]) - 1):
                while end > times[i][j+1][0]:
                    j = j+1
                    for tmp in range(tmp_j+1, j+1):
                        if times[i][tmp][1] > end:
                            end = times[i][tmp][1]
                    if (j >= len(times[i])-1):
                        break
            t = []
            t.append(start)
            t.append(end)
            ti.append(t)
            r = []

            for d in range(tmp_j, j+1):
                for elem in root_cause[i][d]:
                    r.append(elem)
            ro.append(list(set(r)))
            j = j+1
        time.append(ti)
        roots.append(ro)
    anomalyScoreFused = []
    print(anomalyScore)
    for m in range(0, len(time)):
        score = []
        for t in range(0, len(time[m])):
            max = anomalyScore[time[m][t][0]][m]
            for h in range(time[m][t][0]+1, time[m][t][1]):
             if max < anomalyScore[h][m]:
                 max = anomalyScore[h][m]
            score.append(max)
        anomalyScoreFused.append(score)
    return roots, time, anomalyScoreFused

# fuse start and end anomalies for different time series length
def fuseMatrices(roots, times, anomalyScore):
    time = []
    score = []
    ro = []
    min_j = 0
    while times != []:
        list = []
        tmp = 0

        # find the minimal start value in times until times is empty then return
        while len(times[tmp]) == 0:
            if tmp < len(times)-1:
                tmp =tmp+1
            else:
                return ro, time, score
        print('search min')
        min = times[tmp][min_j][0]
        min_i = tmp
        for i in range(tmp+1, len(times)):
            if (min_j< len(times[i])):
                if min > times[i][min_j][0]:
                    min_i = i
                    min = times[min_i][min_j][0]
        min_time = times[min_i][min_j][0]
        max_time = times[min_i][min_j][1]
        root_cause = []
        for e in roots[min_i][min_j]:
            root_cause.append(e)
        anomaly_score = anomalyScore[min_i][min_j]
        list.append([min_i, min_j])
        print('merge start and end')
        #find detected anomaly events which starts between start and end of minimal anomaly
        for i in range(0, len(times)):
            for j in range(0, len(times[i])):
                if (min_j < len(times[i])):
                    if times[min_i][min_j][0] == times[i][j][0] and (min_i != i or min_j != j):
                        if max_time < times[i][j][1]:
                            max_time = times[i][j][1]
                            for e in roots[min_i][min_j]:
                                if not root_cause.__contains__(e):
                                    root_cause.append(e)
                            if anomalyScore[i][j] > anomaly_score:
                                anomaly_score = anomalyScore[i][j]
                        list.append([i, j])
                    elif times[min_i][min_j][1] > times[i][j][0]:
                        if i != min_i or j != min_j:
                            print(times[min_i][min_j][1], times[i][j][0])
                            list.append([i,j])
                            if times[min_i][min_j][1] <= times[i][j][1]:
                                max_time = times[i][j][1]
                                for e in roots[min_i][min_j]:
                                    if not root_cause.__contains__(e):
                                        root_cause.append(e)
                                if anomalyScore[i][j] > anomaly_score:
                                    anomaly_score = anomalyScore[i][j]
        time.append([min_time, max_time])
        print(root_cause)
        ro.append(root_cause)
        score.append(anomaly_score)
        for elem in list:
            times[elem[0]].pop(elem[1])
            roots[elem[0]].pop(elem[1])
            anomalyScore[elem[0]].pop(elem[1])
    return ro, time, score


# gets the reconstruction error and identifies and analyse anomalys
def main():
    print('-------------------------------')
    print('Calculate thresholds based on the validation data set')
    # calculate minimal and maximal threshold for anomaly detection based on the validation data set
    threshold_max, threshold_min = determinethreshold()

    print(threshold_min, threshold_max)

    # calculate the average reconstruction error for every sensor based on the validation data set
    maxError, minError = min_max_SensorError()

    print(maxError, minError)
    print('-------------------------------')

    nbr_datasets = len(config.datasets)

    # start anomaly detection and diagnosis for every dataset
    for i in range(0, nbr_datasets):
        if (i != config.no_failure):
            print('-------------------------------')
            print('Start Anomaly Detection and Diagnosis for dataset', i)

            path = os.path.abspath(".") + config.datasets[i][0][2::]

            # Anomaly Detection and Diagnosis for every file with the reconstruction error
            for file in os.listdir(path + config.directoryname_NNresults):
                if file.find(config.filename_reconstruction_error) != -1 and file[len(file)-4:len(file)] == '.npy':

                    # import reconstruction error
                    resid = np.load(path + config.directoryname_NNresults + file)

                    # creation of a matrix with size timesteps x sensors
                    # to save the anomaly score of every sensor in every timestep
                    anomalyScore = np.zeros((resid.shape[0], resid.shape[3]))

                    # identify reconstruction errors which are higher than the calculated threshold_max oder
                    # smaler thal threshold_min; increment the anomaly threshold of the timestep for every correlation
                    # which is higher than threshold_max or lower than threshold_min
                    print('Calculate Anomaly Score')
                    for l in range(0, resid.shape[0]):
                        for h in range(0, resid.shape[1]):
                            for j in range(0, resid.shape[2]):
                                for k in range(0, resid.shape[3]):
                                    if resid[l][h][j][k] > threshold_max:
                                        anomalyScore[l][k] += 1
                                    if resid[l][h][j][k] < threshold_min:
                                        anomalyScore[l][k] += 1

                    # identify timestamps with a anomaly score which is higher than 2 as anomaly event
                    # create a list of anomaly events for ever signature matrix with different length
                    print('Identify Anomaly Events')
                    anomalyEvents = []
                    for h in range(0, resid.shape[3]):
                        anomalytmp = []
                        for j in range(0, resid.shape[0]):
                            if anomalyScore[j][h] >= 2:
                                anomalytmp.append(j)
                        anomalyEvents.append(anomalytmp)

                    # determine the root cause for every anomaly event
                    print('Root Cause Detection for every Anomaly Event')
                    root_cause = anomalyDiagnosis(anomalyEvents, resid)

                    #calculate start, end for every anomaly event
                    print('Determine Anomaly start and end')
                    times = getDuration(anomalyEvents, root_cause, resid, maxError, minError)

                    # fuse Anomaly Events with overlapping start and end time
                    print('Fuse Anomaly Events')
                    roots, time, score = fuseRootTime(root_cause, times, anomalyScore)
                    rootfinal, timefinal, scorefinal = fuseMatrices(roots, time, score)

                    # create JSON to save information about Anomlies
                    data = {}
                    data['anomalies'] = []
                    for h in range(0,len(rootfinal)):
                        start = getStart(timefinal[h][0], path + config.filename_pkl)
                        end = getEnd(timefinal[h][1], path + config.filename_pkl)
                        dif = getDif(start, end)
                        data_export = unpickle_data(path + config.filename_pkl)
                        roots_names = []
                        for j in rootfinal[h]:
                            roots_names.append(data_export.columns[j])
                        score = scorefinal[h]
                        data['anomalies'].append({
                            'start': str(start),
                            'end': str(end),
                            'duration': str(dif),
                            'roots': str(roots_names),
                            'anomaly score': str(score)

                        })

                    with open(path + config.filename_diagnosis, 'w') as outfile:
                        json.dump(data, outfile)


if __name__ == '__main__':
    main()

