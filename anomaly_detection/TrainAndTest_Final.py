import os
import sys
import tensorflow as tf
import pandas as pd
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
import numpy as np
import collections
from MscredModel import MSCRED
from MscredModel import Memory
from MscredModel import Memory2
from MscredModel import MSGCRED
#from mscred import MemoryInstanceBased
from configuration.Configuration import Configuration
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# TF and background relevant settings
#tf.disable_v2_behavior()

# import logging
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load configuration
config = Configuration()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# This method allows to compute thresholds on a hold-out validation data set that can be used for detecting anomalies on the test data set
# Returns
# thresholds: np matrix (dim, 2*attributes) with thresholds for each dimension and attribute (used in MSCRED-Framework)
# mse_threshold: mse threshold value considering the full example
def calculateThreshold(reconstructed_input, recon_err_perAttrib_valid, threshold_selection_criterium='99%', mse_per_example=None, print_pandas_statistics_for_validation=False, feature_names=None):

    thresholds = np.zeros((reconstructed_input.shape[3], reconstructed_input.shape[1] + reconstructed_input.shape[2]))
    mse_threshold = None
    for i_dim in range(reconstructed_input.shape[3]):
        data_curr_dim = recon_err_perAttrib_valid[:, :, i_dim]
        # print("data_curr_dim shape: ", data_curr_dim.shape)
        data_curr_dim = np.squeeze(data_curr_dim)
        # print("data_curr_dim shape: ", data_curr_dim.shape)
        df_curr_dim = pd.DataFrame(data_curr_dim)
        # print(df_curr_dim.head())
        df_curr_dim_described = df_curr_dim.describe(percentiles=[.25, .5, .75, 0.8, 0.9, 0.95, 0.97, 0.99])
        if print_pandas_statistics_for_validation:
            print("Statistics of dim: ", i_dim, " from healthy data of the validation data set: ")
            print(df_curr_dim_described)
        # get max value for each sensor
        # max = df_curr_dim_described.loc['max', :]
        # print("max: ", max)
        thresholds[i_dim, :] = df_curr_dim_described.loc[threshold_selection_criterium, :].values  # 97%
        # d1 = dict(zip(df_curr_dim_described.loc[threshold_selection_criterium, :].values, feature_names))
        # print(i_dim, ": ", df_curr_dim_described.loc[threshold_selection_criterium, :].values)
        # print("d1: ", d1)
        # arr_index = np.where(feature_names == 'txt16_i4')
        # print("INDEX txt16_i4: ", arr_index)
    # MSE threshold
    if mse_per_example is not None:
        df_mse = pd.DataFrame(mse_per_example)
        df_mse_described = df_mse.describe(percentiles=[.25, .5, .75, 0.8, 0.9, 0.95, 0.97, 0.99])
        if print_pandas_statistics_for_validation:
            print("Statistics for mse from healthy data of the validation data set: ")
            print(df_mse_described)
        mse_threshold = df_mse_described.loc[threshold_selection_criterium].values
        print("Selected mse threshold: ", mse_threshold)

    return thresholds, mse_threshold

def calculateThresholdWithLabelsAsMSCRED(residual_matrix, labels_valid, residual_matrix_test, residual_matrix_wo_FaF=None, attr_names=None,curr_run_identifier="", dict_results={}):
    # Treshold calculation

    # Determine threshold θ empirically  Paper only states the following: (" ... the number of elements whose value is
    # larger than a given threshold θ in the residual signature matrices and θ is detemined empirically over different
    # datasets." (p. 1413)
    # Idea: iterate over all residual values of the validation set and then use the threshold with the highest ROC-AUC
    # since to optimize the anomaly score s(t) w.r.t. to an f1-score (p. 1414) labeled data is needed anyway

    ### Supervised
    # 381/61/61/
    highest_roc_auc_attri_dims_count_w = 0
    highest_dict = {}
    num_examples = residual_matrix.shape[0]
    num_data_streams = residual_matrix.shape[1]
    anomaly_score_per_example_w_highest_roc_auc = np.zeros((num_examples))
    y_true = np.where(labels_valid == 'no_failure', 0, 1)
    for dimension in range(residual_matrix.shape[3]):
        data_curr_dim = residual_matrix[:, :, :, dimension]
        # Reshape to (examples, residual elements) --> (381,61*61)
        data_curr_dim =np.reshape(data_curr_dim,(num_examples, num_data_streams * num_data_streams))
        #Normalize the data
        #scaler = scaler_dict[dimension]
        #data_curr_dim = scaler.transform(data_curr_dim)
        # Iterate over all occured residual values, using every 1000th entry
        for curr_threshold in np.sort(data_curr_dim.flatten()[0::1000]):
            # Define broken elements for every example:
            anomaly_score_per_example = np.zeros((num_examples))
            for example_idx in range(num_examples):
                res_mat_example = data_curr_dim[example_idx,:]
                num_over_thrshld = len(res_mat_example[res_mat_example > curr_threshold])
                anomaly_score_per_example[example_idx] = num_over_thrshld
            # All data collected for current threshold ... do some evaluation with roc auc
            roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels_valid, anomaly_score_per_example)
            avgpr_w, avgpr_m, pr_auc_valid = calculate_PRCurve(labels_valid, anomaly_score_per_example)
            #print("Dim:",dimension,"Thrs:", curr_threshold,"| roc-auc:",roc_auc_attri_dims_count_w,"| avgpr:",avgpr_w,"| pr_auc:",pr_auc_valid_knn)
            if highest_roc_auc_attri_dims_count_w < roc_auc_attri_dims_count_w:
                highest_roc_auc_attri_dims_count_w = roc_auc_attri_dims_count_w
                highest_dict["dim"] = dimension
                highest_dict["threshold"] = curr_threshold
                anomaly_score_per_example_w_highest_roc_auc = anomaly_score_per_example
                # Store relevant metrics
                dict_results["threshold"] = curr_threshold
                dict_results['roc_auc_valid'] = roc_auc_attri_dims_count_w
                dict_results['avgpr_valid'] = avgpr_w
                dict_results['pr_auc_valid'] = pr_auc_valid

    print("Highest ROC AUC: ", highest_roc_auc_attri_dims_count_w, "at dim:",highest_dict["dim"],"with threshold:",highest_dict["threshold"] )

    # anomaly_score_per_example_w_highest_roc_auc contains anomaly scores (i.e. s(t), thus the number of elements
    # from the residual matrix that are higher than a threshold θ) based on the highest roc_auc (i.e. best decicable between normal and abnormal)
    # Aim:  τ = β · max {s(t)valid} with β ∈ [1, 2] is set to maximize the F1 Score over the validation period. (p. 1414)
    beta = 0
    tau = 0

    # Get s(t)_valid max on (failure free?) valid  data (p. 1414)
    '''
    example_idx_of_curr_label = np.where(labels_valid == "no_failure")
    as_no_failure = anomaly_score_per_example_w_highest_roc_auc[example_idx_of_curr_label]
    st_valid_max = np.max(as_no_failure)
    print("max(s(t)_valid)", st_valid_max)
    
    
    for curr_beta in arange(1,2,0.01):
        curr_threshold = st_valid_max*curr_beta
        y_pred = np.where(anomaly_score_per_example_w_highest_roc_auc >= curr_threshold, 1, 0)
        roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels_valid,  y_pred)
        prec_rec_fscore_support = precision_recall_fscore_support(labels_valid, y_pred, average=average)

        print("Thrs:", curr_threshold, "| roc-auc:", roc_auc_attri_dims_count_w, "| f1:", prec_rec_fscore_support[2])
    '''

    best_threshold_tau = 0
    highest_f1_score = 0
    # Iterate over the number of elements of the residual matrix and look which value maximize the f1-score:
    for curr_threshold in range(num_data_streams * num_data_streams): #np.sort(anomaly_score_per_example_w_highest_roc_auc):
        y_pred = np.where(anomaly_score_per_example_w_highest_roc_auc >= curr_threshold, 1, 0)
        #roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels_valid,  y_pred)
        prec_rec_fscore_support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        curr_weighted_f1 = prec_rec_fscore_support[2]
        #print("Thrs:", curr_threshold, "| roc-auc:", 0, "| f1:", curr_weighted_f1)

        # Store highest f1 score and the threshold tau
        if highest_f1_score < curr_weighted_f1:
            best_threshold_tau = curr_threshold
            highest_f1_score = curr_weighted_f1

            # Store relevant metrics
            dict_results["precision_valid"] = prec_rec_fscore_support[0]
            dict_results["recall_valid"] = prec_rec_fscore_support[1]
            dict_results["f1_valid"] = curr_weighted_f1

    print("Best f1-score on valid data set: ", highest_f1_score,"with threshold tau=",best_threshold_tau)
    y_pred = np.where(anomaly_score_per_example_w_highest_roc_auc >= best_threshold_tau, 1, 0)
    plotHistogram(anomaly_score_per_example_w_highest_roc_auc, labels_valid, filename="Plot_MSCRED_AnoScore_per_Example_Valid.png", min=np.min(anomaly_score_per_example_w_highest_roc_auc),
                  max=np.max(anomaly_score_per_example_w_highest_roc_auc), num_of_bins=20)
    print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly'], digits=4))

    ### UnSupervised
    # Using the max value and 99th percentil value from failure-free data ...
    threshold_unsupervised_max = {}
    threshold_unsupervised_percentil = {}
    if not residual_matrix_wo_FaF is None:
        for dimension in range(residual_matrix.shape[3]):
            data_curr_dim = residual_matrix_wo_FaF[:, :, :, dimension]
            # Max value (i.e. distance) from all residual matrices
            threshold_unsupervised_max[dimension] = np.max(data_curr_dim.flatten())
            # 99 percentil value over all residual matrices
            percentil = 99
            threshold_unsupervised_percentil[dimension] = np.percentile(data_curr_dim.flatten(), percentil)

            print("Unsupvervised max thrs:",threshold_unsupervised_max[dimension],"| 99th percentil:", threshold_unsupervised_percentil[dimension])

    # Classify test data
    #if not residual_matrix_test is None:
    # 1. Calculate anomaly scores on test data (.i.e sum of elements over a threshold in the residual matrix)
    residual_matrix_best_dim = residual_matrix_test[:,:,:,highest_dict["dim"]]
    residual_matrix_best_dim_flat = np.reshape(residual_matrix_best_dim, (residual_matrix_best_dim.shape[0], num_data_streams * num_data_streams))
    anomaly_score_per_example_test = np.zeros((residual_matrix_best_dim_flat.shape[0]))
    for example_idx in range(residual_matrix_best_dim_flat.shape[0]):
        res_mat_example = residual_matrix_best_dim_flat[example_idx, :]
        num_over_thrshld = len(res_mat_example[res_mat_example > highest_dict["threshold"]])
        anomaly_score_per_example_test[example_idx] = num_over_thrshld

    # 2. Classify scores as abnormal (i.e. anomaly) or normal (i.e. no anomaly)
    y_pred = np.where(anomaly_score_per_example_test >= best_threshold_tau, 1, 0)

    # 3. Get residuals per data stream
    residual_matrix_best_dim_marked = np.where(residual_matrix_best_dim > highest_dict["threshold"], 1, 0)
    residual_matrix_best_dim_marked_axis1 = np.sum(residual_matrix_best_dim_marked, axis=1)
    residual_matrix_best_dim_marked_axis2 = np.sum(residual_matrix_best_dim_marked, axis=2)
    #residual_matrix_best_dim_marked_axis1 = np.sum(residual_matrix_best_dim, axis=1)
    #residual_matrix_best_dim_marked_axis2 = np.sum(residual_matrix_best_dim, axis=2)
    print("residual_matrix_best_dim_marked_axis1 shape:", residual_matrix_best_dim_marked_axis1.shape, "residual_matrix_best_dim_marked_axis2 shape:", residual_matrix_best_dim_marked_axis2.shape)
    residuals_per_data_stream = residual_matrix_best_dim_marked_axis1 + residual_matrix_best_dim_marked_axis2
    print("residuals_per_data_stream shape", residuals_per_data_stream.shape)

    # Get ordered and combined dictionary of anomalous data streams
    residuals_per_data_stream_idx = np.argsort(-residuals_per_data_stream)

    # store information as dictonary
    store_relevant_attribut_idx = {}
    store_relevant_attribut_dis = {}
    store_relevant_attribut_name = {}
    for i in range(residuals_per_data_stream_idx.shape[0]):
        store_relevant_attribut_idx[i] = residuals_per_data_stream_idx[i,:]                 # indexes of data streams sorted by anomaly score
        store_relevant_attribut_dis[i] = residuals_per_data_stream[i, :]                    # anomaly scores in original order
        store_relevant_attribut_name[i] = attr_names[residuals_per_data_stream_idx[i, :]]   # names of data streams sorted by anomaly score
        if config.print_all_examples:
            print("residuals_per_data_stream["+str(i)+",:]", residuals_per_data_stream[i, :])
            print("residuals_per_data_stream_idx["+str(i)+",:]", residuals_per_data_stream_idx[i, :])
            print("attr_names[residuals_per_data_stream_idx["+str(i)+", :]]", attr_names[residuals_per_data_stream_idx[i, :]])



    import pickle
    a_file = open('store_relevant_attribut_idx_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_idx, a_file)
    a_file.close()
    a_file = open('store_relevant_attribut_dis_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_dis, a_file)
    a_file.close()
    a_file = open('store_relevant_attribut_name_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_name, a_file)
    a_file.close()
    np.save('predicted_anomalies' + curr_run_identifier + '.npy', y_pred)


    return anomaly_score_per_example_test, best_threshold_tau, dict_results #, y_pred


def calculateThresholdWithLabelsAsMSCRED_Scaled(residual_matrix_scaled, residual_matrix_scaled_axis1,residual_matrix_scaled_axis2, labels_valid, residual_matrix_test,residual_matrix_test_axis1,residual_matrix_test_axis2,
                                         residual_matrix_wo_FaF=None, attr_names=None,curr_run_identifier=""):
    # Treshold calculation

    # Determine threshold θ empirically  Paper only states the following: (" ... the number of elements whose value is
    # larger than a given threshold θ in the residual signature matrices and θ is detemined empirically over different
    # datasets." (p. 1413)
    # Idea: iterate over all residual values of the validation set and then use the threshold with the highest ROC-AUC
    # since to optimize the anomaly score s(t) w.r.t. to an f1-score (p. 1414) labeled data is needed anyway

    ### Supervised
    # 381/61/61/
    highest_roc_auc_attri_dims_count_w = 0
    highest_dict = {}
    num_examples = residual_matrix_scaled.shape[0]
    num_data_streams = residual_matrix_scaled.shape[1]
    anomaly_score_per_example_w_highest_roc_auc = np.zeros((num_examples))
    y_true = np.where(labels_valid == 'no_failure', 0, 1)
    for dimension in range(residual_matrix_scaled.shape[3]):
        data_curr_dim = residual_matrix_scaled[:, :, :, dimension]
        # Reshape to (examples, residual elements) --> (381,61*61)
        data_curr_dim = np.reshape(data_curr_dim, (num_examples, num_data_streams * num_data_streams))
        # Normalize the data
        # scaler = scaler_dict[dimension]
        # data_curr_dim = scaler.transform(data_curr_dim)
        # Iterate over all occured residual values, using every 1000th entry
        for curr_threshold in np.sort(data_curr_dim.flatten()[0::1000]):
            # Define broken elements for every example:
            anomaly_score_per_example = np.zeros((num_examples))
            for example_idx in range(num_examples):
                res_mat_example = data_curr_dim[example_idx, :]
                num_over_thrshld = len(res_mat_example[res_mat_example > curr_threshold])
                anomaly_score_per_example[example_idx] = num_over_thrshld
            # All data collected for current threshold ... do some evaluation with roc auc
            roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels_valid,
                                                                                      anomaly_score_per_example)
            avgpr_w, avgpr_m, pr_auc_valid_knn = calculate_PRCurve(labels_valid, anomaly_score_per_example)
            # print("Dim:",dimension,"Thrs:", curr_threshold,"| roc-auc:",roc_auc_attri_dims_count_w,"| avgpr:",avgpr_w,"| pr_auc:",pr_auc_valid_knn)
            if highest_roc_auc_attri_dims_count_w < roc_auc_attri_dims_count_w:
                highest_roc_auc_attri_dims_count_w = roc_auc_attri_dims_count_w
                highest_dict["dim"] = dimension
                highest_dict["threshold"] = curr_threshold
                anomaly_score_per_example_w_highest_roc_auc = anomaly_score_per_example
    print("Highest ROC AUC: ", highest_roc_auc_attri_dims_count_w, "at dim:", highest_dict["dim"], "with threshold:",
          highest_dict["threshold"])

    # anomaly_score_per_example_w_highest_roc_auc contains anomaly scores (i.e. s(t), thus the number of elements
    # from the residual matrix that are higher than a threshold θ) based on the highest roc_auc (i.e. best decicable between normal and abnormal)
    # Aim:  τ = β · max {s(t)valid} with β ∈ [1, 2] is set to maximize the F1 Score over the validation period. (p. 1414)
    beta = 0
    tau = 0

    # Get s(t)_valid max on (failure free?) valid  data (p. 1414)
    '''
    example_idx_of_curr_label = np.where(labels_valid == "no_failure")
    as_no_failure = anomaly_score_per_example_w_highest_roc_auc[example_idx_of_curr_label]
    st_valid_max = np.max(as_no_failure)
    print("max(s(t)_valid)", st_valid_max)


    for curr_beta in arange(1,2,0.01):
        curr_threshold = st_valid_max*curr_beta
        y_pred = np.where(anomaly_score_per_example_w_highest_roc_auc >= curr_threshold, 1, 0)
        roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels_valid,  y_pred)
        prec_rec_fscore_support = precision_recall_fscore_support(labels_valid, y_pred, average=average)

        print("Thrs:", curr_threshold, "| roc-auc:", roc_auc_attri_dims_count_w, "| f1:", prec_rec_fscore_support[2])
    '''

    best_threshold_tau = 0
    highest_f1_score = 0
    # Iterate over the number of elements of the residual matrix and look which value maximize the f1-score:
    for curr_threshold in range(
            num_data_streams * num_data_streams):  # np.sort(anomaly_score_per_example_w_highest_roc_auc):
        y_pred = np.where(anomaly_score_per_example_w_highest_roc_auc >= curr_threshold, 1, 0)
        # roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels_valid,  y_pred)
        prec_rec_fscore_support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        curr_weighted_f1 = prec_rec_fscore_support[2]
        # print("Thrs:", curr_threshold, "| roc-auc:", 0, "| f1:", curr_weighted_f1)

        # Store highest f1 score and the threshold tau
        if highest_f1_score < curr_weighted_f1:
            best_threshold_tau = curr_threshold
            highest_f1_score = curr_weighted_f1

    print("Best f1-score on valid data set: ", highest_f1_score, "with threshold tau=", best_threshold_tau)
    y_pred = np.where(anomaly_score_per_example_w_highest_roc_auc >= best_threshold_tau, 1, 0)
    plotHistogram(anomaly_score_per_example_w_highest_roc_auc, labels_valid,
                  filename="Plot_MSCRED_AnoScore_per_Example_Valid.png",
                  min=np.min(anomaly_score_per_example_w_highest_roc_auc),
                  max=np.max(anomaly_score_per_example_w_highest_roc_auc), num_of_bins=20)
    print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly'], digits=4))

    ### UnSupervised
    # Using the max value and 99th percentil value from failure-free data ...
    threshold_unsupervised_max = {}
    threshold_unsupervised_percentil = {}
    if not residual_matrix_wo_FaF is None:
        for dimension in range(residual_matrix_scaled.shape[3]):
            data_curr_dim = residual_matrix_wo_FaF[:, :, :, dimension]
            # Max value (i.e. distance) from all residual matrices
            threshold_unsupervised_max[dimension] = np.max(data_curr_dim.flatten())
            # 99 percentil value over all residual matrices
            percentil = 99
            threshold_unsupervised_percentil[dimension] = np.percentile(data_curr_dim.flatten(), percentil)

            print("Unsupvervised max thrs:", threshold_unsupervised_max[dimension], "| 99th percentil:",
                  threshold_unsupervised_percentil[dimension])

    # Classify test data
    # if not residual_matrix_test is None:
    # 1. Calculate anomaly scores on test data (.i.e sum of elements over a threshold in the residual matrix)
    residual_matrix_best_dim = residual_matrix_test[:, :, :, highest_dict["dim"]]
    residual_matrix_best_dim_flat = np.reshape(residual_matrix_best_dim,
                                               (residual_matrix_best_dim.shape[0], num_data_streams * num_data_streams))
    anomaly_score_per_example_test = np.zeros((residual_matrix_best_dim_flat.shape[0]))
    for example_idx in range(residual_matrix_best_dim_flat.shape[0]):
        res_mat_example = residual_matrix_best_dim_flat[example_idx, :]
        num_over_thrshld = len(res_mat_example[res_mat_example > highest_dict["threshold"]])
        anomaly_score_per_example_test[example_idx] = num_over_thrshld

    # 2. Classify scores as abnormal (i.e. anomaly) or normal (i.e. no anomaly)
    y_pred = np.where(anomaly_score_per_example_test >= best_threshold_tau, 1, 0)

    # 3. Get residuals per data stream
    # residual_matrix_best_dim_marked = np.where(residual_matrix_best_dim > highest_dict["threshold"], 1, 0)
    # residual_matrix_best_dim_marked_axis1 = np.sum(residual_matrix_best_dim_marked, axis=1)
    # residual_matrix_best_dim_marked_axis2 = np.sum(residual_matrix_best_dim_marked, axis=2)
    residual_matrix_best_dim_scaled_axis1 = residual_matrix_test_axis1[:, :, :, highest_dict["dim"]]
    residual_matrix_best_dim_scaled_axis2 = residual_matrix_test_axis2[:, :, :, highest_dict["dim"]]
    residual_matrix_best_dim_marked_axis1 = np.sum(residual_matrix_best_dim_scaled_axis1, axis=1)
    residual_matrix_best_dim_marked_axis2 = np.sum(residual_matrix_best_dim_scaled_axis2, axis=2)
    print("residual_matrix_best_dim_marked_axis1 shape:", residual_matrix_best_dim_marked_axis1.shape,
          "residual_matrix_best_dim_marked_axis2 shape:", residual_matrix_best_dim_marked_axis2.shape)
    residuals_per_data_stream = residual_matrix_best_dim_marked_axis1 + residual_matrix_best_dim_marked_axis2
    print("residuals_per_data_stream shape", residuals_per_data_stream.shape)

    # Get ordered and combined dictonary of anomalous data streams
    residuals_per_data_stream_idx = np.argsort(-residuals_per_data_stream)

    # Store information as dictonary
    store_relevant_attribut_idx = {}
    store_relevant_attribut_dis = {}
    store_relevant_attribut_name = {}
    for i in range(residuals_per_data_stream_idx.shape[0]):
        store_relevant_attribut_idx[i] = residuals_per_data_stream_idx[i, :]
        store_relevant_attribut_dis[i] = residuals_per_data_stream[i, :]
        store_relevant_attribut_name[i] = attr_names[residuals_per_data_stream_idx[i, :]]
        if config.print_all_examples:
            print("residuals_per_data_stream[" + str(i) + ",:]", residuals_per_data_stream[i, :])
            print("residuals_per_data_stream_idx[" + str(i) + ",:]", residuals_per_data_stream_idx[i, :])
            print("attr_names[residuals_per_data_stream_idx[" + str(i) + ", :]]",
                  attr_names[residuals_per_data_stream_idx[i, :]])

    # store_relevant_attribut_idx = residuals_per_data_stream_idx
    # store_relevant_attribut_dis = residuals_per_data_stream
    # store_relevant_attribut_name = attr_names[residuals_per_data_stream_idx]

    import pickle
    a_file = open('store_relevant_attribut_idx_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_idx, a_file)
    a_file.close()
    a_file = open('store_relevant_attribut_dis_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_dis, a_file)
    a_file.close()
    a_file = open('store_relevant_attribut_name_' + curr_run_identifier + '.pkl', "wb")
    pickle.dump(store_relevant_attribut_name, a_file)
    a_file.close()
    np.save('predicted_anomalies' + curr_run_identifier + '.npy', y_pred)

    return anomaly_score_per_example_test, best_threshold_tau  # , y_pred

def calculateThresholdWithLabels(reconstructed_input, recon_err_perAttrib_valid, labels_valid, mse_per_example=None, print_pandas_statistics_for_validation=False, feature_names=None,  mse_per_example_per_dims_wf=None, scaler_dict=None):
    # We have four possibilities to find a threshold with reasonable effort.
    # i) calculating the threshold as MSE over all dimension
    # ii) calculating the threshold as MSE for each dimension
    # iii) calculating on threshold and calculating the number of attribute violations
    # iv) similar to three, but for each dimension
    y_true = np.where(labels_valid == 'no_failure', 0, 1)
    #print("___y_true: ",y_true)
    #y_true = np.reshape(y_true, y_true.shape[0])
    roc_auc_mse_w, roc_auc_mse_m = calculate_RocAuc(y_true, mse_per_example)
    avgpr_w, avgpr_m, pr_auc = calculate_PRCurve(y_true, mse_per_example)

    plotHistogram(mse_per_example, labels_valid, filename="Plot_MSE_per_Example_Valid.png", min=np.min(mse_per_example), max=np.max(mse_per_example),num_of_bins=10)

    #sort all anomaly scores and iterate over them for finding the highest score
    f1_weighted_max_threshold   = 0
    f1_weighted_max_value       = 0
    f1_macro_max_threshold      = 0
    f1_macro_max_value          = 0
    f1_weighted_max_threshold_per_dim   = 0
    f1_weighted_max_value_per_dim       = 0
    f1_macro_max_threshold_per_dim      = 0
    f1_macro_max_value_per_dim          = 0
    best_dim = 0
    f1_weighted_max_threshold_per_dim_attr   = 0
    f1_weighted_max_value_per_dim_attr       = 0
    f1_macro_max_threshold_per_dim_attr      = 0
    f1_macro_max_value_per_dim_attr          = 0
    best_dim_attr = 0
    f1_weighted_max_threshold_per_dim_attr_percentil   = 0
    f1_weighted_max_value_per_dim_attr_percentil       = 0
    f1_macro_max_threshold_per_dim_attr_percentil      = 0
    f1_macro_max_value_per_dim_attr_percentil          = 0
    best_dim_attr_percentil = 0

    # i) mse over all different correlation length dimensions
    for curr_threshold in np.sort(mse_per_example):
        y_pred = np.where(mse_per_example >= curr_threshold, 1, 0)
        #print("y_pred shape:", y_pred.shape)
        #print("y_true shape:", y_true.shape)
        # print(" ---- ")
        # print("Threshold: ", curr_threshold)
        # print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        # print("precision_recall_fscore_support: ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
        # print(" ---- ")
        if f1_weighted_max_value < p_r_f_s_weighted[2]:
            f1_weighted_max_value = p_r_f_s_weighted[2]
            f1_weighted_max_threshold = curr_threshold
        if f1_macro_max_value < p_r_f_s_weighted[2]:
            f1_macro_max_value = p_r_f_s_macro[2]
            f1_macro_max_threshold = curr_threshold
    # ii) mse over all different correlation length dimensions
    for dimension in range(mse_per_example_per_dims_wf.shape[1]):
        for curr_threshold in np.sort(mse_per_example_per_dims_wf[:,dimension]):
            y_pred = np.where(mse_per_example_per_dims_wf[:,dimension] >= curr_threshold, 1, 0)
            #print("y_pred shape:", y_pred.shape)
            #print("y_true shape:", y_true.shape)
            # print(" ---- ")
            # print("Threshold: ", curr_threshold)
            # print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))
            TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
            p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            # Precision or positive predictive value
            PPV = TP / (TP + FP)
            # Negative predictive value
            NPV = TN / (TN + FN)
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            # False negative rate
            FNR = FN / (TP + FN)
            # False discovery rate
            FDR = FP / (TP + FP)

            # Overall accuracy
            ACC = (TP + TN) / (TP + FP + FN + TN)
            # print("precision_recall_fscore_support: ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
            # print(" ---- ")
            if f1_weighted_max_value_per_dim < p_r_f_s_weighted[2]:
                f1_weighted_max_value_per_dim = p_r_f_s_weighted[2]
                f1_weighted_max_threshold_per_dim = curr_threshold
                best_dim = dimension
                #print("Dimension: ", dimension,"f1_weighted_max_value_per_dim: ", f1_weighted_max_value_per_dim)
            if f1_macro_max_value_per_dim < p_r_f_s_weighted[2]:
                f1_macro_max_value_per_dim = p_r_f_s_macro[2]
                f1_macro_max_threshold_per_dim = curr_threshold
                best_dim = dimension
    # iii) attribute over threshold with same threshold (single one) for every attribute
    for dimension in range(recon_err_perAttrib_valid.shape[2]):
        data_curr_dim = recon_err_perAttrib_valid[:, :, dimension]
        # TODO: Normalisierung testen
        #Normalize the data
        #scaler = scaler_dict[dimension]
        #data_curr_dim = scaler.transform(data_curr_dim)
        for curr_threshold in np.sort(data_curr_dim.flatten()[0::25]):
            #print("curr_threshold: ", curr_threshold)
            anomalous_attributes_found = np.where(data_curr_dim >= curr_threshold, 1, 0)
            #print("anomalous_attributes_found.shape:",anomalous_attributes_found.shape)
            per_axis = np.sum(anomalous_attributes_found, axis=1)
            #print("per_axis.shape:", per_axis.shape)
            y_pred = np.where(per_axis >= 1, 1, 0)
            #print("y_pred.shape:", y_pred.shape)
            TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
            p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            # Precision or positive predictive value
            PPV = TP / (TP + FP)
            # Negative predictive value
            NPV = TN / (TN + FN)
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            # False negative rate
            FNR = FN / (TP + FN)
            # False discovery rate
            FDR = FP / (TP + FP)

            # Overall accuracy
            ACC = (TP + TN) / (TP + FP + FN + TN)
            '''
            print("NAN found y_pred: ", np.where(np.isnan(y_pred)))
            y_pred = np.nan_to_num(y_pred)
            y_pred[y_pred > 1e308] = 0
            y_pred[y_pred < -1e308] = 0
            y_pred.astype('float64')
            print("y_test_pred_df: ", y_pred.shape)
            #print("precision_recall_fscore_support: ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
            # print(" ---- ")
            roc_auc_iii_w, roc_auc_iii_m = calculate_RocAuc(y_true, y_pred)
            print("roc_auc_iii_w: ", roc_auc_iii_w)
            avgpr_w_iii, avgpr_m_iii, pr_auc = calculate_PRCurve(y_true, mse_per_example)
            '''
            if f1_weighted_max_value_per_dim_attr < p_r_f_s_weighted[2]:
                f1_weighted_max_value_per_dim_attr = p_r_f_s_weighted[2]
                f1_weighted_max_threshold_per_dim_attr = curr_threshold
                best_dim_attr= dimension
                # print("Dimension: ", dimension,"f1_weighted_max_value_per_dim: ", f1_weighted_max_value_per_dim)
            if f1_macro_max_value_per_dim_attr < p_r_f_s_weighted[2]:
                f1_macro_max_value_per_dim_attr = p_r_f_s_macro[2]
                f1_macro_max_threshold_per_dim_attr = curr_threshold
    # iv) attribute over threshold for with different thresholds based on the best percentile over all attributes
    thresholds = np.zeros((recon_err_perAttrib_valid.shape[2], recon_err_perAttrib_valid.shape[1]))
    for i_dim in range(reconstructed_input.shape[3]):
        data_curr_dim = recon_err_perAttrib_valid[:, :, i_dim]
        # TODO: Normalisierung testen
        # Normalize the data
        #scaler = scaler_dict[i_dim]
        #data_curr_dim = scaler.transform(data_curr_dim)
        # print("data_curr_dim shape: ", data_curr_dim.shape)
        data_curr_dim = np.squeeze(data_curr_dim)
        # print("data_curr_dim shape: ", data_curr_dim.shape)
        df_curr_dim = pd.DataFrame(data_curr_dim)
        # print(df_curr_dim.head())
        for percentil in [.25, .55, .75, 0.85, 0.92, 0.95, 0.97, 0.99]:
            df_curr_dim_described= df_curr_dim.describe(percentiles=[percentil])
            #print("df_curr_dim_described", df_curr_dim_described.tail())
            # required since different from given percentil values as used in pandas data framse
            key = (str(percentil)+"%").replace("0.","")
            if len(key) >3:
                key = key[:2] + '.' + key[2:]
                #print("key: ", key)
            #print("df_curr_dim_described.loc[key, :].values: ", df_curr_dim_described.loc[key, :].values)
            thresholds[i_dim, :] = df_curr_dim_described.loc[key, :].values
            #print(df_curr_dim_described.loc['mean', :].values)
            #print(df_curr_dim_described.loc['std', :].values)
            #print("curr_threshold shape: ", curr_threshold.shape)
            #print("data_curr_dim shape: ", data_curr_dim.shape)
            anomalous_attributes_found = np.where(data_curr_dim >= thresholds[i_dim, :], 1, 0)
            #print("anomalous_attributes_found.shape:",anomalous_attributes_found.shape)
            per_axis = np.sum(anomalous_attributes_found, axis=1)
            #print("per_axis.shape:", per_axis.shape)
            y_pred = np.where(per_axis >= 1, 1, 0)
            #print("y_pred.shape:", y_pred.shape)
            TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
            p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')

            #print("precision_recall_fscore_support: ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
            # print(" ---- ")
            if f1_weighted_max_value_per_dim_attr_percentil < p_r_f_s_weighted[2]:
                f1_weighted_max_value_per_dim_attr_percentil = p_r_f_s_weighted[2]
                f1_weighted_max_threshold_per_dim_attr_percentil = percentil
                best_dim_attr_percentil = dimension
                # print("Dimension: ", dimension,"f1_weighted_max_value_per_dim: ", f1_weighted_max_value_per_dim)
            if f1_macro_max_value_per_dim_attr_percentil < p_r_f_s_weighted[2]:
                f1_macro_max_value_per_dim_attr_percentil = p_r_f_s_macro[2]
                f1_macro_max_threshold_per_dim_attr_percentil = percentil
                best_dim_attr_percentil = dimension

    print(" ++++ VALID DATA with MSE per example ++++ ")
    print("ROCAUC:\t", roc_auc_mse_w)
    print("AvgPR:\t", avgpr_w)
    print("PRAUC:\t", pr_auc)
    print(" ++++ ")
    print(" Best Threshold on Validation Split Found for dimension:")
    print(" F1 Score weighted:\t", f1_weighted_max_value, "\t\t Threshold: ", f1_weighted_max_threshold)
    print(" F1 Score macro:\t", f1_macro_max_value, "\t\t Threshold: ", f1_macro_max_threshold)
    print(" F1 Score weighted:\t", f1_weighted_max_value_per_dim, "\t\t Threshold: ", f1_weighted_max_threshold_per_dim,"for dimension:",best_dim)
    print(" F1 Score macro:\t", f1_macro_max_value_per_dim, "\t\t Threshold: ", f1_macro_max_threshold_per_dim,"for dimension:",best_dim)
    print(" F1 Score weighted:\t", f1_weighted_max_value_per_dim_attr, "\t\t Threshold: ", f1_weighted_max_threshold_per_dim_attr,"for dimension:",best_dim_attr)
    print(" F1 Score macro:\t", f1_macro_max_value_per_dim_attr, "\t\t Threshold: ", f1_macro_max_threshold_per_dim_attr,"for dimension:",best_dim_attr)
    print(" F1 Score weighted:\t", f1_weighted_max_value_per_dim_attr_percentil, "\t\t Threshold: ", f1_weighted_max_threshold_per_dim_attr_percentil,"for dimension:",best_dim_attr_percentil)
    print(" F1 Score macro:\t", f1_macro_max_value_per_dim_attr_percentil, "\t\t Threshold: ", f1_macro_max_threshold_per_dim_attr_percentil,"for dimension:",best_dim_attr_percentil)
    print(" ++++ ")

    return f1_weighted_max_threshold, [f1_weighted_max_threshold_per_dim, best_dim], [f1_weighted_max_threshold_per_dim_attr, best_dim_attr], thresholds

def evaluate(anomaly_score, labels_test, anomaly_threshold, average='weighted',curr_run_identifier="", dict_results={}):
        # prepare the labels according sklearn
        y_true = np.where(labels_test == 'no_failure', 0, 1)
        #np.reshape(y_true, y_true.shape[0])
        plotHistogram(anomaly_score, labels_test, filename="Plot_MSE_per_Example_Test"+str(curr_run_identifier)+".png",
                      min=np.min(anomaly_score), max=np.max(anomaly_score), num_of_bins=50)
        if config.use_memory_restriction:
            #remove the last element since problem with batch handling ...
            y_true = y_true[:-1]

        # apply threshold for anomaly decision
        y_pred = np.where(anomaly_score >= anomaly_threshold, 1, 0)

        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        # p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        roc_auc_mse_w, roc_auc_mse_m = calculate_RocAuc(y_true, anomaly_score)
        avgpr_w, avgpr_m, pr_auc = calculate_PRCurve(y_true, anomaly_score)
        print("")
        print(" +++ +++ +++ +++ +++ FINAL EVAL TEST +++ +++ +++ +++ +++ +++ +++")
        print("")
        print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly'], digits=4))
        print("")
        print("FPR:\t", FPR)
        print("FNR:\t", FNR)
        print("ROCAUC:\t", roc_auc_mse_w)
        print("AvgPR:\t", avgpr_w)
        print("PRAUC:\t", pr_auc)
        print("")
        print(" +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++")
        print("")
        prec_rec_fscore_support = precision_recall_fscore_support(y_true, y_pred, average=average)

        # fill dictonary
        dict_results['FPR_test'] = FPR
        dict_results['FNR_test'] = FNR
        dict_results['roc_auc_test'] = roc_auc_mse_w
        dict_results['avgpr_w_test'] = avgpr_w
        dict_results['pr_auc_test'] = pr_auc
        dict_results['precision_test'] = prec_rec_fscore_support[0]
        dict_results['recall_test'] = prec_rec_fscore_support[1]
        dict_results['f1_test'] = prec_rec_fscore_support[2]

        return dict_results

# This method computes reconstruction errors
def calculateReconstructionError(real_input, reconstructed_input, plot_heatmaps, use_corr_rel_matrix=False, corr_rel_matrix=None, y_labels=None):
    reconstruction_error_matrixes = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1], reconstructed_input.shape[2], reconstructed_input.shape[3]))
    reconstruction_errors_perAttribute = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1] + reconstructed_input.shape[2], reconstructed_input.shape[3]))
    mse_per_example_over_all_dims = np.zeros((reconstructed_input.shape[0]))
    mse_per_example_per_dims = np.zeros((reconstructed_input.shape[0],reconstructed_input.shape[3]))

    print("reconstructed_input.shape[0]: ", reconstructed_input.shape[0])
    for i_example in range(reconstructed_input.shape[0]):  # Iterate over all examples
        for i_dim in range(reconstructed_input.shape[3]):  # Iterate over all "time-dimensions"
            # Get reconstructed and real input data
            curr_matrix_input_recon = reconstructed_input[i_example, :, :, i_dim]
            curr_matrix_input_real = real_input[i_example, :, :, i_dim]

            # Calculate the reconstruction error
            diff = curr_matrix_input_recon - curr_matrix_input_real
            # Mask out any contribution from irrelevant correlations (if activated)
            if use_corr_rel_matrix:
                diff = diff * corr_rel_matrix
            # Apply Frobenius norm and square (acc. Eq.6), output is a single scalar
            mse = np.square(np.square(np.linalg.norm(diff, ord='fro')))

            # According https://github.com/numpy/numpy/blob/v1.22.0/numpy/linalg/linalg.py (frobenius norm)
            diff_sqrt_squared = np.sqrt(np.square(diff))

            # This is later used to compute the "broken" elements/correlation and is done as in the official implementation
            # provided along the paper: https://github.com/7fantasysz/MSCRED/blob/4bdfcacf3d92b7f1b83a10708251453b7cf68075/code/evaluate.py#L64
            reconstruction_error_matrixes[i_example, :, :, i_dim] = diff_sqrt_squared
            '''
            plot_heatmap_of_reconstruction_error(id=i_example, dim=i_dim, input=curr_matrix_input_real,
                                                 output=curr_matrix_input_recon,
                                                 rec_error_matrix=reconstruction_error_matrixes[i_example,:,:,i_dim])
            '''

            # Mean of reconstruction error per attribute / data stream horizontally and vertical
            mse_axis0 = np.mean(diff_sqrt_squared, axis=0)
            mse_axis1 = np.mean(diff_sqrt_squared, axis=1)
            reconstruction_errors_perAttribute[i_example, :reconstructed_input.shape[1], i_dim] = mse_axis0
            reconstruction_errors_perAttribute[i_example, reconstructed_input.shape[1]:, i_dim] = mse_axis1
            mse_per_example_over_all_dims[i_example] = mse_per_example_over_all_dims[i_example] + mse
            mse_per_example_per_dims[i_example, i_dim] = mse

        if plot_heatmaps:
            if not y_labels is None:
                if i_example%50 == 0 or not y_labels[i_example] == "no_failure":
                    plot_heatmap_of_reconstruction_error2(id=i_example, input=real_input[i_example, :, :, :],
                                                  output=reconstructed_input[i_example, :, :, :],
                                                  rec_error_matrix=reconstruction_error_matrixes[i_example, :, :, :],y_label=y_labels[i_example])
            else:
                if i_example%50 == 0:
                    plot_heatmap_of_reconstruction_error2(id=i_example, input=real_input[i_example, :, :, :],
                                                  output=reconstructed_input[i_example, :, :, :],
                                                  rec_error_matrix=reconstruction_error_matrixes[i_example, :, :, :])

    # Durch Anzahl an Dimensionen teilen
    mse_per_example_over_all_dims = mse_per_example_over_all_dims / reconstructed_input.shape[3]

    # Please note that reconstruction_error_matrixes correspond to residual matrices
    return reconstruction_error_matrixes, reconstruction_errors_perAttribute, mse_per_example_over_all_dims, mse_per_example_per_dims

# This method compares threshold values with the reconstruction error and marks anomalous events
# Returns
# eval_results (#Examples, 2*attributes, dim) where each with 1 indicates an anomaly
# eval_results_over_all_dimensions (#Examples, 2*attributes) sums the number of dimensions with anomalies
# eval_results_over_all_dimensions_for_each_example (#Examples, dim) sums the number of anomalies per dimension
def calculateAnomalies(reconstructed_input, recon_err_perAttrib, thresholds, print_att_dim_statistics = True, use_dim_for_anomaly_detection = 1):
    eval_results = np.zeros((reconstructed_input.shape[0], reconstructed_input.shape[1] + reconstructed_input.shape[2], reconstructed_input.shape[3]))

    for i_example in range(recon_err_perAttrib.shape[0]):
        rec_error_curr_example = recon_err_perAttrib[i_example, :, :]
        for i_dim in use_dim_for_anomaly_detection:
            # Compare reconstruction error if it exceeds threshold to detect an anomaly
            eval = rec_error_curr_example[:, i_dim] > thresholds[i_dim, :]
            eval_results[i_example, :, i_dim] = eval
            '''
            # USED FOR DEBUGGING:
            if i_example > recon_err_perAttrib.shape[0]-10:
                d1 = dict(zip(rec_error_curr_example[:, i_dim], eval))
                print(i_example," T- ",i_dim,": ", thresholds[i_dim, :])
                print(i_example," R- ", i_dim, ": ", rec_error_curr_example[:, i_dim])
                print(i_example," E- ", i_dim, ": ", d1)
            '''
    #print("eval_results shape: ", eval_results.shape)
    eval_results_over_all_dimensions = np.sum(eval_results, axis=2)
    eval_results_over_all_dimensions_for_each_example = np.sum(eval_results, axis=1)
    #print("eval_results_over_all_dimensions shape: ", eval_results_over_all_dimensions.shape)
    #print("Last Example:")
    #print(eval_results_over_all_dimensions[recon_err_perAttrib.shape[0]-1,:])
    #print("eval_results_over_all_dimensions_for_each_example shape: ", eval_results_over_all_dimensions_for_each_example.shape)
    if print_att_dim_statistics:
        for i_dim in range(reconstructed_input.shape[3]):
            print("Investigation of Dimension: ", i_dim, " for the number of attributes of an anomaly")
            for i in range(reconstructed_input.shape[1]+reconstructed_input.shape[1]):
                print("Num of examples with ", i ," anomalous attributes (rows&columns) in dimension", i_dim, " anomalies: ",(eval_results_over_all_dimensions_for_each_example[:, i_dim] == i).sum())
            # num_ones = (y == 1).sum()

    return eval_results, eval_results_over_all_dimensions, eval_results_over_all_dimensions_for_each_example

# This method conducts the evaluation process and print out relevant results. Validation data sets prodies no labels (since all examples are from the class no-failure)
def printEvaluation2(reconstructed_input, eval_results_over_all_dimensions, feature_names, labels=None,
                     num_of_dim_under_threshold=0, num_of_dim_over_threshold=1000, mse_threshold=None, mse_values=None,
                     use_attribute_anomaly_as_condition=True, print_all_examples=True):
    # Counter variables for no_failure and failure class
    TP_NF = TN_NF = FP_NF = FN_NF = 0
    TP_F = TN_F = FP_F = FN_F = 0

    # Required to store the information for further processing and evaluation
    store_relevant_attribut_idx = {}
    store_relevant_attribut_dis = {}
    store_relevant_attribut_name = {}
    y_pred_ano = np.zeros((reconstructed_input.shape[0]))

    # Save number of attribute anomalies to calculate roc-auc values on this
    anomalous_attributes_per_example = np.zeros((reconstructed_input.shape[0]))
    anomalous_attributes_dims_per_example = np.zeros((reconstructed_input.shape[0]))

    for example_idx in range(reconstructed_input.shape[0]):
        # Adding 1 if the number of reconstruction errors over different dimension is reached, otherwise 0
        idx_with_Anomaly_1 = np.where(np.logical_and(
            num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]],
            eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]] > num_of_dim_over_threshold))
        count_dim_anomalies_1 = eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][idx_with_Anomaly_1]
        # print("eval_results_over_all_dimensions_f: ", eval_results_over_all_dimensions_f[example_idx,:pred.shape[1]])
        # idx_with_Anomaly_2 = np.where(eval_results_over_all_dimensions_f[example_idx,pred.shape[1]:] > num_of_dim_over_threshold)
        idx_with_Anomaly_2 = np.where(np.logical_and(
            num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:],
            eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:] > num_of_dim_over_threshold))
        count_dim_anomalies_2 = eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][idx_with_Anomaly_2]

        anomalous_attributes_per_example[example_idx] = np.sum(idx_with_Anomaly_1) + np.sum(idx_with_Anomaly_2)
        anomalous_attributes_dims_per_example[example_idx] = np.sum(count_dim_anomalies_1) + np.sum(count_dim_anomalies_2)

        if example_idx >= reconstructed_input.shape[0]-10:
            print("count_dim_anomalies_1: ", count_dim_anomalies_1)
            print("count_dim_anomalies_2: ", count_dim_anomalies_2)
            print("idx_with_Anomaly_1: ", idx_with_Anomaly_1)
            print("idx_with_Anomaly_2: ", idx_with_Anomaly_2)
            print("feature_names 1: ", feature_names[idx_with_Anomaly_1])
            print("feature_names 2: ", feature_names[idx_with_Anomaly_2])
            print("eval_results_over_all_dimensions 1: ", eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][23])
            print("eval_results_over_all_dimensions 2: ", eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][23])

        #Get ordered and combined dictonary of anomalous data streams
        anomalies_combined_asc_ordered = order_anomalies(count_dim_anomalies_1, count_dim_anomalies_2,idx_with_Anomaly_1, idx_with_Anomaly_2, feature_names)
        anomaly_detected = False

        # store information
        #print("anomalies_combined_asc_ordered : ",anomalies_combined_asc_ordered)
        #print("np.where(np.in1d(anomalies_combined_asc_ordered, feature_names))[0] : ", np.where(np.in1d(anomalies_combined_asc_ordered, feature_names))[0] )
        store_relevant_attribut_idx[example_idx] = np.where(np.in1d(anomalies_combined_asc_ordered, feature_names))[0]  # np.argsort(-anomalous_attributes_dims_per_example)
        store_relevant_attribut_dis[example_idx] = anomalous_attributes_dims_per_example[example_idx]
        store_relevant_attribut_name[example_idx] = anomalies_combined_asc_ordered


        #Decide which condition / criterium is used for detecting anomalies
        if use_attribute_anomaly_as_condition:
            # A feature is seen as anomalous if row and column is higher than the threshold value
            if (feature_names[idx_with_Anomaly_1].shape[0] + feature_names[idx_with_Anomaly_2].shape[0]) > 1: # 0 if any occurend, 1 if double occurence (horizontal and vertical)
                anomaly_detected = True
        else:
            # A feature is seen as anomalous if the mse of its reconstruction is higher than the threshold
            if (mse_values[example_idx] > mse_threshold):
                anomaly_detected = True

        if anomaly_detected:
            y_pred_ano[example_idx] = 1
            if labels is not None:
                if labels[example_idx] == 'no_failure':
                    FN_NF += 1
                    FP_F += 1
                else:
                    TN_NF += 1
                    TP_F += 1
        else:
            # No Anomaly detected
            y_pred_ano[example_idx] = 0
            if labels is not None:
                if labels[example_idx] == 'no_failure':
                    TP_NF += 1
                    TN_F += 1
                else:
                    FP_NF += 1
                    FN_F += 1

        #Print output for each example
        if labels is not None:
            '''
            print(labels[example_idx],": ", mse_values[example_idx], " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1],
                  " with counts: ", count_dim_anomalies_1,"idx_with_Anomaly_2: ", feature_names[idx_with_Anomaly_2],
                  " with counts: ", count_dim_anomalies_2)
             '''
            if print_all_examples:
                print("Label: ", labels[example_idx], " -  Detected Anomaly: ",anomaly_detected," based on: MSE:  %.8f" % mse_values[example_idx], "Anomalies combined: ",
                      anomalies_combined_asc_ordered)
        else:
            '''
            print("NoFailure: ", " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1], " with counts: ",
                  count_dim_anomalies_1, feature_names[idx_with_Anomaly_2], " with counts: ",
                  count_dim_anomalies_2)
            '''
            if print_all_examples:
                print("Label: no_failure (validation) -  Detected Anomaly: ",anomaly_detected," based on: MSE:  %.8f" % mse_values[example_idx], "Anomalies combined: ",
                      anomalies_combined_asc_ordered)

    if labels is not None:
        '''
        print("------------------------------------------------------------------------")
        print("\nResults: ")
        print("No_Failure TP: \t\t", TP_NF, "\t\tFailure TP: \t\t", TP_F)
        print("No_Failure TN: \t\t", TN_NF, "\t\tFailure TN: \t\t", TN_F)
        print("No_Failure FP: \t\t", FP_NF, "\t\tFailure FP: \t\t", FP_F)
        print("No_Failure FN: \t\t", FN_NF, "\t\tFailure FN: \t\t", FN_F)
        prec_NF = TP_NF/(TP_NF+FP_NF+ tf.keras.backend.epsilon())
        rec_NF = TP_NF/(TP_NF+FN_NF+ tf.keras.backend.epsilon())
        acc_NF = (TP_NF+TN_NF)/(TP_NF+TN_NF+FP_NF+FN_NF+ tf.keras.backend.epsilon())
        f1_NF = 2*((prec_NF*rec_NF)/(prec_NF+rec_NF+ tf.keras.backend.epsilon()))
        prec_F = TP_F / (TP_F + FP_F+ tf.keras.backend.epsilon())
        rec_F = TP_F / (TP_F + FN_F+ tf.keras.backend.epsilon())
        acc_F = (TP_F + TN_F) / (TP_F + TN_F + FP_F + FN_F+ tf.keras.backend.epsilon())
        f1_F = 2 * ((prec_F * rec_F) / (prec_F + rec_F+ tf.keras.backend.epsilon()))
        print("------------------------------------------------------------------------")
        print("No_Failure Acc: \t %.3f" % acc_NF, "\t\tFailure Acc: \t\t %.3f" % acc_F)
        print("No_Failure Precision: \t %.3f" % prec_NF, "\t\tFailure Precision: \t %.3f" % prec_F)
        print("No_Failure Recall: \t %.3f" %rec_NF, "\t\tFailure Recall: \t %.3f" % rec_F)
        print("No_Failure F1: \t \t %.3f" % f1_NF, "\t\tFailure F1:  \t\t %.3f" % f1_F)
        print("------------------------------------------------------------------------")
        TP_A = TP_NF+TP_F
        TN_A = TN_NF+TN_F
        FP_A = FP_NF+FP_F
        FN_A = FN_NF+FN_F
        prec_A = TP_A / (TP_A + FP_A+ tf.keras.backend.epsilon())
        rec_A = TP_A / (TP_A + FN_A+ tf.keras.backend.epsilon())
        acc_A = (TP_A + TN_A) / (TP_A + TN_A + FP_A + FN_A+ tf.keras.backend.epsilon())
        f1_A = 2 * ((prec_A * rec_A) / (prec_A + rec_A+ tf.keras.backend.epsilon()))
        '''
        roc_auc_mse_w,roc_auc_mse_m = calculate_RocAuc(labels, mse_values)
        roc_auc_attri_count_w,roc_auc_attri_count_m = calculate_RocAuc(labels, anomalous_attributes_per_example)
        roc_auc_attri_dims_count_w, roc_auc_attri_dims_count_m = calculate_RocAuc(labels, anomalous_attributes_dims_per_example)
        avgpr_w, avgpr_m, pr_auc_valid_knn = calculate_PRCurve(labels,mse_values)

        #mse_values_normalized = (mse_values - np.min(mse_values)) / np.ptp(mse_values)
        #anomalous_attributes_dims_per_example_normalized = ((anomalous_attributes_dims_per_example + 1) - np.min((anomalous_attributes_dims_per_example))) / np.ptp((anomalous_attributes_dims_per_example + 1))
        #print("New ROC-AUC: ",calculate_RocAuc(labels, (mse_values_normalized/(anomalous_attributes_dims_per_example_normalized+1))))
        print("------------------------------------------------------------------------")
        #print("OverAll Acc: \t\t %.3f" % acc_A)
        #print("OverAll Precision: \t %.3f" % prec_A)
        #print("OverAll Recall: \t %.3f" %rec_A)
        #print("OverAll F1: \t \t %.3f" % f1_A)
        print("OverAll RocAuc MSE: \t %.3f" % roc_auc_mse_w,"\t %.3f" % roc_auc_mse_m)
        print("OverAll RocAuc Att: \t %.3f" % roc_auc_attri_count_w,"\t %.3f" % roc_auc_attri_count_m)
        print("OverAll RocAuc AttDim: \t %.3f" % roc_auc_attri_dims_count_w,"\t %.3f" % roc_auc_attri_dims_count_m)
        print("------------------------------------------------------------------------")
        y_pred = np.where(anomalous_attributes_dims_per_example >= 2, 1, 0)
        y_true = np.where(labels == 'no_failure', 0, 1)
        roc_auc_wie_eval_w, roc_auc_wie_eval_m = calculate_RocAuc(y_true, y_pred_ano)
        roc_auc_wie_eval_w_2, roc_auc_wie_eval_m_2 = calculate_RocAuc(y_true, y_pred)
        print(" +++ +++ +++ +++ +++ FINAL EVAL TEST +++ +++ +++ +++ +++ +++ +++")
        print("")
        print(classification_report(y_true, y_pred_ano, target_names=['normal', 'anomaly'], digits=4))
        print("ROCAUC wie verwendet: ", roc_auc_wie_eval_w)
        print("ROCAUC wie verwendet: ", roc_auc_wie_eval_w_2)

        #return f1_A, roc_auc_mse_w, roc_auc_attri_count_w, roc_auc_attri_dims_count_w
        return [store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name, y_pred]

    def printEvaluation3(reconstructed_input, eval_results_over_all_dimensions, feature_names, labels=None,
                         num_of_dim_under_threshold=0, num_of_dim_over_threshold=1000, mse_threshold=None,
                         mse_values=None,
                         use_attribute_anomaly_as_condition=True, print_all_examples=True, y_pred=None):
        # Counter variables for no_failure and failure class
        TP_NF = TN_NF = FP_NF = FN_NF = 0
        TP_F = TN_F = FP_F = FN_F = 0

        # Required to store the information for further processing and evaluation
        store_relevant_attribut_idx = {}
        store_relevant_attribut_dis = {}
        store_relevant_attribut_name = {}
        y_pred_ano = np.zeros((reconstructed_input.shape[0]))

        # Save number of attribute anomalies to calculate roc-auc values on this
        anomalous_attributes_per_example = np.zeros((reconstructed_input.shape[0]))
        anomalous_attributes_dims_per_example = np.zeros((reconstructed_input.shape[0]))

        for example_idx in range(reconstructed_input.shape[0]):
            # Adding 1 if the number of reconstruction errors over different dimension is reached, otherwise 0
            idx_with_Anomaly_1 = np.where(np.logical_and(
                num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx,
                                             :reconstructed_input.shape[1]],
                eval_results_over_all_dimensions[example_idx,
                :reconstructed_input.shape[1]] > num_of_dim_over_threshold))
            count_dim_anomalies_1 = eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][
                idx_with_Anomaly_1]
            # print("eval_results_over_all_dimensions_f: ", eval_results_over_all_dimensions_f[example_idx,:pred.shape[1]])
            # idx_with_Anomaly_2 = np.where(eval_results_over_all_dimensions_f[example_idx,pred.shape[1]:] > num_of_dim_over_threshold)
            idx_with_Anomaly_2 = np.where(np.logical_and(
                num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx,
                                             reconstructed_input.shape[1]:],
                eval_results_over_all_dimensions[example_idx,
                reconstructed_input.shape[1]:] > num_of_dim_over_threshold))
            count_dim_anomalies_2 = eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][
                idx_with_Anomaly_2]

            anomalous_attributes_per_example[example_idx] = np.sum(idx_with_Anomaly_1) + np.sum(idx_with_Anomaly_2)
            anomalous_attributes_dims_per_example[example_idx] = np.sum(count_dim_anomalies_1) + np.sum(
                count_dim_anomalies_2)

            if example_idx >= reconstructed_input.shape[0] - 10:
                print("count_dim_anomalies_1: ", count_dim_anomalies_1)
                print("count_dim_anomalies_2: ", count_dim_anomalies_2)
                print("idx_with_Anomaly_1: ", idx_with_Anomaly_1)
                print("idx_with_Anomaly_2: ", idx_with_Anomaly_2)
                print("feature_names 1: ", feature_names[idx_with_Anomaly_1])
                print("feature_names 2: ", feature_names[idx_with_Anomaly_2])
                print("eval_results_over_all_dimensions 1: ",
                      eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][23])
                print("eval_results_over_all_dimensions 2: ",
                      eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][23])

            # Get ordered and combined dictonary of anomalous data streams
            anomalies_combined_asc_ordered = order_anomalies(count_dim_anomalies_1, count_dim_anomalies_2,
                                                             idx_with_Anomaly_1, idx_with_Anomaly_2, feature_names)
            anomaly_detected = False

            # store information
            # print("anomalies_combined_asc_ordered : ",anomalies_combined_asc_ordered)
            # print("np.where(np.in1d(anomalies_combined_asc_ordered, feature_names))[0] : ", np.where(np.in1d(anomalies_combined_asc_ordered, feature_names))[0] )
            store_relevant_attribut_idx[example_idx] = np.where(np.in1d(anomalies_combined_asc_ordered, feature_names))[
                0]  # np.argsort(-anomalous_attributes_dims_per_example)
            store_relevant_attribut_dis[example_idx] = anomalous_attributes_dims_per_example[example_idx]
            store_relevant_attribut_name[example_idx] = anomalies_combined_asc_ordered

            # Decide which condition / criterium is used for detecting anomalies
            if use_attribute_anomaly_as_condition:
                # A feature is seen as anomalous if row and column is higher than the threshold value
                if (feature_names[idx_with_Anomaly_1].shape[0] + feature_names[idx_with_Anomaly_2].shape[
                    0]) > 1:  # 0 if any occurend, 1 if double occurence (horizontal and vertical)
                    anomaly_detected = True
            else:
                # A feature is seen as anomalous if the mse of its reconstruction is higher than the threshold
                if (mse_values[example_idx] > mse_threshold):
                    anomaly_detected = True

            if anomaly_detected:
                y_pred_ano[example_idx] = 1
                if labels is not None:
                    if labels[example_idx] == 'no_failure':
                        FN_NF += 1
                        FP_F += 1
                    else:
                        TN_NF += 1
                        TP_F += 1
            else:
                # No Anomaly detected
                y_pred_ano[example_idx] = 0
                if labels is not None:
                    if labels[example_idx] == 'no_failure':
                        TP_NF += 1
                        TN_F += 1
                    else:
                        FP_NF += 1
                        FN_F += 1

            # Print output for each example
            if labels is not None:
                '''
                print(labels[example_idx],": ", mse_values[example_idx], " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1],
                      " with counts: ", count_dim_anomalies_1,"idx_with_Anomaly_2: ", feature_names[idx_with_Anomaly_2],
                      " with counts: ", count_dim_anomalies_2)
                 '''
                if print_all_examples:
                    print("Label: ", labels[example_idx], " -  Detected Anomaly: ", anomaly_detected,
                          " based on: MSE:  %.8f" % mse_values[example_idx], "Anomalies combined: ",
                          anomalies_combined_asc_ordered)
            else:
                '''
                print("NoFailure: ", " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1], " with counts: ",
                      count_dim_anomalies_1, feature_names[idx_with_Anomaly_2], " with counts: ",
                      count_dim_anomalies_2)
                '''
                if print_all_examples:
                    print("Label: no_failure (validation) -  Detected Anomaly: ", anomaly_detected,
                          " based on: MSE:  %.8f" % mse_values[example_idx], "Anomalies combined: ",
                          anomalies_combined_asc_ordered)

        avgpr_w, avgpr_m, pr_auc_valid_knn = calculate_PRCurve(labels, mse_values)

        y_pred = np.where(anomalous_attributes_dims_per_example >= 2, 1, 0)
        y_true = np.where(labels == 'no_failure', 0, 1)
        roc_auc_wie_eval_w, roc_auc_wie_eval_m = calculate_RocAuc(y_true, y_pred_ano)
        roc_auc_wie_eval_w_2, roc_auc_wie_eval_m_2 = calculate_RocAuc(y_true, y_pred)
        print(" +++ +++ +++ +++ +++ FINAL EVAL TEST +++ +++ +++ +++ +++ +++ +++")
        print("")
        print(classification_report(y_true, y_pred_ano, target_names=['normal', 'anomaly'], digits=4))
        print("ROCAUC wie verwendet: ", roc_auc_wie_eval_w)
        print("ROCAUC wie verwendet: ", roc_auc_wie_eval_w_2)

        # return f1_A, roc_auc_mse_w, roc_auc_attri_count_w, roc_auc_attri_dims_count_w
        return [store_relevant_attribut_idx, store_relevant_attribut_dis, store_relevant_attribut_name, y_pred]

# Not used anymore
def plot_heatmap_of_reconstruction_error(input, output, rec_error_matrix, id, dim):
    print("example id: ", id)
    fig, axs = plt.subplots(3)
    fig.suptitle('Reconstruction Error of '+str(id)+ "in Dimension: "+str(dim))
    axs[0].imshow(input, cmap='hot', interpolation='nearest')
    axs[1].imshow(output, cmap='hot', interpolation='nearest')
    axs[2].imshow(rec_error_matrix, cmap='hot', interpolation='nearest')
    #plt.imshow(rec_error_matrix, cmap='hot', interpolation='nearest')
    filename="heatmaps/" +str(id)+"-"+str(dim)+"rec_error_matrix.png"
    print("filename: ", filename)
    plt.savefig(filename)
    print("plot_heatmap_of_reconstruction_error")

# Generates a heat map of the reconstruction error and shows the real input and its reconstruction
def plot_heatmap_of_reconstruction_error2(input, output, rec_error_matrix, id, y_label=""):
    #print("example id: ", id)
    fig, axs = plt.subplots(3,4, gridspec_kw = {'wspace':0.1, 'hspace':-0.1}) # rows/dim
    fig.suptitle('Reconstruction Error of '+str(id))
    fig.subplots_adjust(hspace=0, wspace=0)
    #print("input shape: ", input.shape)
    #print("output shape: ", output.shape)
    #print("rec_error_matrix shape: ", rec_error_matrix.shape)
    #X_valid_y[i_example, :, :, :],
    #output = pred[i_example, :, :, :], rec_error_matrix = reconstruction_error_matrixes[i_example, :, :, :]

    # the left end of the figure differs
    bottom = 0.05
    height = 0.9
    width = 0.15  # * 4 = 0.6 - minus the 0.1 padding 0.3 left for space
    left1, left2, left3, left4 = 0.05, 0.25, 1 - 0.25 - width, 1 - 0.05 - width

    for dim in range(config.dim_of_dataset):
        #print("dim: ", dim)
        axs[0,dim].imshow(input[:,:,dim], cmap='hot', interpolation='nearest')
        #cax = plt.axes([0.95, 0.05, 0.05, 0.9])
        axs[1,dim].imshow(output[:,:,dim], cmap='hot', interpolation='nearest')
        pos0 = axs[2,dim].imshow(rec_error_matrix[:,:,dim], cmap='hot', interpolation='nearest')
        cbar = fig.colorbar(pos0, ax=axs[2, dim],orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=3, labelrotation=45)
        #plt.colorbar(axs[0,dim])
        #axs[1, dim].colorbar()
        #axs[2, dim].colorbar()
        axs[0,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[0,dim].set_yticks([], [])  # same for y ticks
        axs[1,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[1,dim].set_yticks([], [])  # same for y ticks
        axs[2,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[2,dim].set_yticks([], [])  # same for y ticks
        #axs[0,dim].set_aspect('equal')
        #axs[1, dim].set_aspect('equal')
        #axs[2, dim].set_aspect('equal')
        plt.subplots_adjust(hspace=.001)
    #plt.imshow(rec_error_matrix, cmap='hot', interpolation='nearest')
    filename="heatmaps/" +str(id)+"-"+"rec_error_matrix_"+y_label+".png"
    #print("filename: ", filename)
    fig.tight_layout(pad=0.1, h_pad=0.01)
    #fig.subplots_adjust(wspace=1.1,hspace=-0.1)
    #plt.subplots_adjust(bottom=0.1,top=0.2, hspace=0.1)
    plt.savefig(filename, dpi=500)
    plt.clf()
    #print("plot_heatmap_of_reconstruction_error")

def plotHistogram(anomaly_scores, labels, filename="plotHistogramWithMissingFilename.png", min=-1, max=1, num_of_bins=100):
    # divide examples in normal and anomalous
    font = {'family': 'serif','size': 14}
    plt.rc('font', **font)

    # Get idx of examples with this label
    example_idx_of_no_failure_label = np.where(labels == 'no_failure')
    example_idx_of_opposite_labels = np.squeeze(np.array(np.where(labels != 'no_failure')))
    #feature_data = np.expand_dims(feature_data, -1)
    anomaly_scores_normal = anomaly_scores[example_idx_of_no_failure_label[0]]
    anomaly_scores_unnormal = anomaly_scores[example_idx_of_opposite_labels[0]]

    bins = np.linspace(min, max, num_of_bins)
    bins2 = np.linspace(min, max, 10)
    plt.clf()
    plt.hist(anomaly_scores_normal, bins, alpha=0.5, label='normal')
    plt.hist(anomaly_scores_unnormal, bins2, alpha=0.5, label='unnormal')
    plt.legend(loc='upper right')
    plt.savefig(filename) #pyplot.show()

def remove_failure_examples(lables, feature_data, label_to_retain="no_failure"):
    # Input labels as strings with size (e,1) and feature data with size (e, d)
    # where e is the number of examples and d the number of feature dimensions
    # Return both inputs without any labels beside the label given via parameter label_to_retain

    # Get idx of examples with this label
    example_idx_of_curr_label = np.where(lables == label_to_retain)
    #feature_data = np.expand_dims(feature_data, -1)
    feature_data = feature_data[example_idx_of_curr_label[0],:]
    lables = lables[example_idx_of_curr_label]
    return lables, feature_data

# Loss Function that receives an external matrix where relevant correlations are defined manually (based on domain
# knowledge)
# Input: corr_rel_mat (Attributes,Attributes) with {0,1}
def corr_rel_matrix_weighted_loss(corr_rel_mat):
    def loss(y_true, y_pred):
        #loss = tf.square(y_true - y_pred)
        #print("corr_rel_matrix_weighted_loss loss dim: ", loss.shape)
        #corr_rel_mat_reshaped = np.reshape(corr_rel_mat, (1, 1, 61, 61, 1)).astype(np.float32)
        #loss = loss * corr_rel_mat_reshaped
        #loss = tf.reduce_sum(loss, axis=-1)
        #loss = loss / np.sum(corr_rel_mat) # normalize by number of considered correlations
        #loss = tf.reduce_mean(loss, axis=-1)
        #'''

        # Reshape with batch dim None: https://stackoverflow.com/questions/36668542/flatten-batch-in-tensorflow
        shape = y_true.get_shape().as_list()
        dim = np.prod(shape[1:])
        y_true = tf.reshape(y_true, [-1, dim])
        y_pred = tf.reshape(y_pred, [-1, dim])

        # MSCRED loss according to paper p. 1413, Eq. 6
        d = tf.reduce_sum(tf.square(tf.abs(y_true - y_pred)), axis=1)
        # To avoid NAN loss
        dx = tf.sqrt(tf.maximum(d, 1e-9))
        squarred_frobenius_norm_per_example = tf.square(dx)
        loss = squarred_frobenius_norm_per_example

        return loss
    return loss  # Note the `axis=-1`

# Loss function according to the MSCRED paper, but in the official impl. MSE is used
def mscred_loss(y_true, y_pred):
    shape = y_true.get_shape().as_list()
    dim = np.prod(shape[1:])
    y_true = tf.reshape(y_true, [-1, dim])
    y_pred = tf.reshape(y_pred, [-1, dim])

    # MSCRED loss according to paper p. 1413, Eq. 6
    d = tf.reduce_sum(tf.square(tf.abs(y_true - y_pred)), axis=1)

    # To avoid NAN loss
    dx = tf.sqrt(tf.maximum(d, 1e-9))
    squarred_frobenius_norm_per_example = tf.square(dx)
    loss = squarred_frobenius_norm_per_example

    # verfiy forbenius norm implementation
    #a = np.array([[1, 3, -1], [0, 1, 5], [1, -4, 1]])
    #a_ = np.square(np.linalg.norm((a), ord='fro'))
    #a__ = np.square(np.sqrt(np.sum(np.square(np.abs(a)))))
    #print("a_:", a_,"a__:",a__)

    # MSCRED loss according to paper p. 1413, Eq. 6 (Numpy version)
    '''
    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[1] * y_true.shape[2] * y_true.shape[3]))
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3]))
    squarred_frobenius_norm = np.square(np.linalg.norm((y_true - y_pred), ord='fro'),axis=1)
    # mean for all signature matrices in batch
    loss = np.mean(squarred_frobenius_norm)
    '''
    return loss

def MseLoss(y_true, y_pred):
    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[1]*y_true.shape[2]*y_true.shape[3]))
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3]))
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true, y_pred)
    # Paper acc. implementation: https://github.com/7fantasysz/MSCRED/blob/4bdfcacf3d92b7f1b83a10708251453b7cf68075/code/MSCRED_TF.py#L355
    #loss = tf.square(y_true - y_pred)
    #loss = tf.reduce_mean(loss, axis=-1)
    return loss
def L2Loss(y_true, y_pred):
    loss =  tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss  # Note the `axis=-1`

def SimLoss(y_true, y_pred):
    ### For doing something like this: https://www.sciencedirect.com/science/article/abs/pii/S1361841519301562
    y_pred = tf.squeeze(y_pred)
    #cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

    #Cosine Sim:#
    x = tf.nn.l2_normalize(y_pred, axis=0)
    y = tf.nn.l2_normalize(y_pred, axis=0)
    loss = tf.matmul(x, y, transpose_b=True)
    #print("x loss dim: ", loss.shape)
    loss = tf.math.abs(loss)

    # Eucl. Sim:#
    '''
    s = 2 * tf.matmul(y_pred, y_pred, transpose_b=True)
    diag_x = tf.reduce_sum(y_pred * y_pred, axis=-1, keepdims=True)
    diag_y = tf.reshape(tf.reduce_sum(y_pred * y_pred, axis=-1), (1, -1))
    loss = s - diag_x - diag_y
    #maximize:
    #loss = -loss
    '''
    loss = tf.reduce_mean(loss, axis=-1)
    loss = loss / config.batch_size
    #loss = loss/128/(128*128)
    print("SimLoss loss dim: ", loss.shape)

    return loss

    #return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`

def MemEntropyLoss(y_true, y_pred):
    # Loss fosters memory access to be 1 or 0
    loss = tf.reduce_mean((-y_pred) * tf.math.log(y_pred + 1e-12), axis=-1)
    return loss

def plot_training_process_history(history, curr_run_identifier):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    # summarize history for loss
    plt.clf()
    font = {'family': 'serif', 'size': 14}
    plt.rc('font', **font)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # dict_keys(['loss', 'conv2d_transpose_2_loss', 'concatenate_loss', 'val_loss', 'val_conv2d_transpose_2_loss', 'val_concatenate_loss'])
    plt.title('History of Reconstruction Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    plt.savefig('loss_history_' + curr_run_identifier + '.png')

def apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input, use_corr_rel_matrix_for_input_replace_by_epsilon,input_data,np_corr_rel_matrix):
    if use_corr_rel_matrix_for_input:
        if use_corr_rel_matrix_for_input_replace_by_epsilon:
            print("Start replacing 0 values by epsilon  from the input")
            input_data[input_data == 0] = tf.keras.backend.epsilon()
            print("Finished replacing 0 values by epsilon  from the input")
        print("Start removing irrelevant correlations from the input")
        np_corr_rel_matrix_reshaped = np.reshape(np_corr_rel_matrix, (1, 1, 61, 61, 1))
        input_data = input_data * np_corr_rel_matrix_reshaped
        print("Finished removing irrelevant correlations from the input")
    return input_data

def apply_corr_rel_matrix_on_residual_matrix(use_corr_rel_matrix_for_input_replace_by_epsilon,input_data,np_corr_rel_matrix):
    print("input_data shape: ", input_data.shape)
    if use_corr_rel_matrix_for_input_replace_by_epsilon:
        input_data[input_data == 0] = tf.keras.backend.epsilon()
    np_corr_rel_matrix_reshaped = np.reshape(np_corr_rel_matrix, (1, 61, 61, 1))
    input_data = input_data * np_corr_rel_matrix_reshaped
    print("input_data shape after: ", input_data.shape)
    np.any(np.isnan(input_data))
    np.all(np.isfinite(input_data))
    return input_data

def order_anomalies(count_dim_anomalies_1, count_dim_anomalies_2,idx_with_Anomaly_1, idx_with_Anomaly_2, feature_names):
    # Combing anomalous attributes with its anomaly count in a dictonary
    d1 = dict(zip(feature_names[idx_with_Anomaly_1], count_dim_anomalies_1))
    d2 = dict(zip(feature_names[idx_with_Anomaly_2], count_dim_anomalies_2))
    #print("d1: ", d1)
    #print("d2: ", d2)
    merged_od1_od2 = dict(collections.Counter(d1) + collections.Counter(d2))
    #print("merged_od1_od2: ", merged_od1_od2)
    anomalies_combined_asc_ordered_flipped_dict = dict(sorted(merged_od1_od2.items(), key=lambda item: item[1], reverse=True))
    #print("anomalies_combined_asc_ordered_flipped_dict: ", anomalies_combined_asc_ordered_flipped_dict)
    return anomalies_combined_asc_ordered_flipped_dict #dict(anomalies_combined_asc_ordered)

# This function calculates the RocAuc Score when given a label list and a anomaly score per example
def calculate_RocAuc(test_failure_labels_y, score_per_example):
    if "no_failure" in test_failure_labels_y:
        y_true = np.where(test_failure_labels_y == 'no_failure', 0, 1)
    else:
        y_true = test_failure_labels_y
    #print("mse_per_example_test:", mse_per_example_test.shape)
    score_per_example_test_normalized = (score_per_example - np.min(score_per_example)) / np.ptp(score_per_example)
    np.nan_to_num(score_per_example_test_normalized)
    roc_auc_score_value = roc_auc_score(y_true, score_per_example_test_normalized, average='weighted')
    roc_auc_score_value_m = roc_auc_score(y_true, score_per_example_test_normalized, average='macro')
    return roc_auc_score_value, roc_auc_score_value_m

def calculate_PRCurve(test_failure_labels_y, score_per_example):
    # Replace 'no_failure' string with 0 (for negative class) and failures (anomalies) with 1 (for positive class)
    if "no_failure" in test_failure_labels_y:
        y_true = np.where(test_failure_labels_y == 'no_failure', 0, 1)
    else:
        y_true = test_failure_labels_y
    #print("y_true: ", y_true)
    #print("y_true: ", y_true.shape)
    #print("mse_per_example_test:", mse_per_example_test.shape)
    score_per_example_test_normalized = (score_per_example - np.min(score_per_example)) / np.ptp(score_per_example)
    avgP = average_precision_score(y_true, score_per_example_test_normalized, average='weighted')
    avgP_m = average_precision_score(y_true, score_per_example_test_normalized, average='macro')
    precision, recall, _ = precision_recall_curve(y_true, score_per_example_test_normalized)
    auc_score = auc(recall, precision)
    return avgP, avgP_m, auc_score

class Batch():
    def __init__(self, total, batch_size):
        self.total = total
        self.batch_size = batch_size
        self.current = 0

    def next(self):
        max_index = self.current + self.batch_size
        indices = [i if i < self.total else i - self.total
                   for i in range(self.current, max_index)]
        self.current = max_index % self.total
        return indices
# Returns a matrix of size (#test_examples,valid_examples) where each entry is the pairwise cosine similarity
def get_Similarity_Matrix(valid_vector, test_vector):
    from sklearn.metrics.pairwise import cosine_similarity
    examples_matrix = np.concatenate((valid_vector, test_vector), axis=0)
    pairwise_cosine_sim_matrix = cosine_similarity(examples_matrix)
    print("pairwise_cosine_sim_matrix: ", pairwise_cosine_sim_matrix)
    pairwise_cosine_sim_matrix = pairwise_cosine_sim_matrix[valid_vector.shape[0]:,:valid_vector.shape[0]]

    return pairwise_cosine_sim_matrix

def find_anomaly_threshold(nn_distance_valid, labels_valid):
    # the threshold is optimized based on the given data set, typically the validation set
    '''
    print("nn_distance_valid shape: ", nn_distance_valid.shape)
    print("nn_distance_valid: ", nn_distance_valid)
    print("nn_distance_valid: ", nn_distance_valid)
    print("labels_valid shape: ", labels_valid.shape)
    threshold_min=np.amin(nn_distance_valid)
    threshold_max= np.amax(nn_distance_valid)
    curr_threshold = threshold_min
    '''
    # labels
    y_true = np.where(labels_valid == 'no_failure', 0, 1)
    y_true = np.reshape(y_true, y_true.shape[0])
    #print("y_true shape: ", y_true.shape)

    #sort all anomaly scores and iterate over them for finding the highest score
    nn_distance_valid_sorted = np.sort(nn_distance_valid)
    f1_weighted_max_threshold   = 0
    f1_weighted_max_value       = 0
    f1_macro_max_threshold      = 0
    f1_macro_max_value          = 0
    for curr_threshold in nn_distance_valid_sorted:
        y_pred = np.where(nn_distance_valid <= curr_threshold, 1, 0)
        #print(" ---- ")
        #print("Threshold: ", curr_threshold)
        #print(classification_report(y_true, y_pred, target_names=['normal', 'anomaly']))
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        p_r_f_s_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        p_r_f_s_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        #print("precision_recall_fscore_support: ", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
        #print(" ---- ")
        if f1_weighted_max_value < p_r_f_s_weighted[2]:
            f1_weighted_max_value = p_r_f_s_weighted[2]
            f1_weighted_max_threshold = curr_threshold
        if f1_macro_max_value < p_r_f_s_weighted[2]:
            f1_macro_max_value = p_r_f_s_macro[2]
            f1_macro_max_threshold = curr_threshold
    print(" ++++ ")
    print(" Best Threshold on Validation Split Found:")
    print(" F1 Score weighted: ", f1_weighted_max_value, "\t\t Threshold: ", f1_weighted_max_threshold)
    print(" F1 Score macro: ", f1_macro_max_value, "\t\t\t Threshold: ", f1_macro_max_threshold)
    print(" ++++ ")

    return f1_weighted_max_threshold, f1_macro_max_threshold


def main(run=""):
    # Configurations
    train_model = config.train_model
    test_model = config.test_model

    # Variants of the MSCRED
    guassian_noise_stddev = config.guassian_noise_stddev
    use_attention = config.use_attention
    use_convLSTM = config.use_convLSTM
    use_memory_restriction = config.use_memory_restriction
    use_graph_conv = config.use_graph_conv
    normalize_residual_matrices = config.normalize_residual_matrices

    use_loss_corr_rel_matrix = config.use_loss_corr_rel_matrix
    loss_use_batch_sim_siam = config.loss_use_batch_sim_siam
    use_corr_rel_matrix_for_input = config.use_corr_rel_matrix_for_input
    use_corr_rel_matrix_for_input_replace_by_epsilon = config.use_corr_rel_matrix_for_input_replace_by_epsilon
    plot_heatmap_of_rec_error = config.plot_heatmap_of_rec_error
    curr_run_identifier = config.curr_run_identifier + str(run)
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    step_size = config.step_max
    early_stopping_patience = config.early_stopping_patience
    split_train_test_ratio = config.split_train_test_ratio

    # Test Parameter:
    use_corr_rel_matrix_in_eval = config.use_corr_rel_matrix_in_eval
    threshold_selection_criterium = config.threshold_selection_criterium
    num_of_dim_over_threshold = config.num_of_dim_over_threshold
    num_of_dim_under_threshold = config.num_of_dim_under_threshold
    print_att_dim_statistics = config.print_att_dim_statistics
    generate_deep_encodings = config.generate_deep_encodings
    use_attribute_anomaly_as_condition = config.use_attribute_anomaly_as_condition
    print_all_examples = config.print_all_examples
    print_pandas_statistics_for_validation = config.print_pandas_statistics_for_validation
    use_corr_rel_matrix_on_masking_residual_matrices = config.use_corr_rel_matrix_on_masking_residual_matrices

    use_mass_evaulation = config.use_mass_evaulation
    threshold_selection_criterium_list = config.threshold_selection_criterium_list
    num_of_dim_over_threshold_list = config.num_of_dim_over_threshold_list
    num_of_dim_under_threshold_list = config.num_of_dim_under_threshold_list
    use_corr_rel_matrix_in_eval_list = config.use_corr_rel_matrix_in_eval_list
    use_attribute_anomaly_as_condition_list = config.use_attribute_anomaly_as_condition_list
    use_dim_for_anomaly_detection = config.use_dim_for_anomaly_detection

    use_MemEntropyLoss = config.use_MemEntropyLoss

    printInfo = True

    print("### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ")
    print("curr_run_identifier: ", curr_run_identifier)
    print("train_model: ", config.train_model, ", test_model: ", config.test_model, ", data set version: ", config.use_data_set_version)
    print("### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ")
    print("MSCRED Variant: Guassian Noise added: ", guassian_noise_stddev, "convLSTM: ",use_convLSTM, ", attention used: ", use_attention, ", memory restriction: ", use_memory_restriction)
    print("Corrleation Matrix used as Loss: ", use_loss_corr_rel_matrix, ", Input: ",use_corr_rel_matrix_for_input, ", Replace zero with Epsilon: ", use_corr_rel_matrix_for_input_replace_by_epsilon)
    print("Training related parameters: batchsize: ", batch_size, ", epochs: ", epochs,", learning_rate: ",learning_rate," early_stopping_patience: ", early_stopping_patience, " split_train_test_ratio: ", split_train_test_ratio)
    print("### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ")
    if use_mass_evaulation == False:
        print("Testing related parameters: threshold_selection_criterium: ", threshold_selection_criterium, ", use_corr_rel_matrix_in_eval: ", use_corr_rel_matrix_in_eval, ", num_of_dim_over_threshold: ",
              num_of_dim_over_threshold, " num_of_dim_under_threshold: ", num_of_dim_under_threshold, " split_train_test_ratio: ",
              split_train_test_ratio)
    else:
        print("use_mass_evaulation: ", use_mass_evaulation)
    print("### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ")

    training_data_set_path = config.training_data_set_path
    valid_split_save_path = config.valid_split_save_path
    test_matrix_path = config.test_matrix_path
    test_labels_y_path = config.test_labels_y_path
    test_matrix_path = config.test_matrix_path
    test_labels_y_path = config.test_labels_y_path
    test_matrix_path = test_matrix_path
    test_labels_y_path = test_labels_y_path
    feature_names_path = config.feature_names_path
    valid_matrix_path_wF = config.valid_matrix_path_wF
    valid_labels_y_path_wF = config.valid_labels_y_path_wF

    ### Load Adj Matrix
    adj_matrix_attr_df = pd.read_csv(config.graph_adjacency_matrix_attributes_file, sep=';', index_col=0)
    adj_mat = adj_matrix_attr_df.values.astype(dtype=np.float)

    ### Create MSCRED model as TF graph
    print("config.relevant_features: ", config.relevant_features)
    test_labels_y = np.load(test_labels_y_path)
    print("test_labels_y: ", test_labels_y)
    ### Load Correlation Relevance Matrix ###
    df_corr_rel_matrix = pd.read_csv('../data/Attribute_Correlation_Relevance_Matrix_v0.csv', sep=';',index_col=0)
    np_corr_rel_matrix = df_corr_rel_matrix.values
    print("np_corr_rel_matrix shape: ", np_corr_rel_matrix.shape)
    print("Only ", np.sum(np_corr_rel_matrix)," of ",np_corr_rel_matrix.shape[0] * np_corr_rel_matrix.shape[1]," correlations")

    print('-------------------------------')
    print('Creation of the model')

    # create graph structure of the NN
    print("Loaded AdjMat: ", adj_mat )
    print("AdjMat shape: ", adj_mat.shape)
    if use_graph_conv:
        model_MSCRED = MSGCRED().create_model(guassian_noise_stddev=guassian_noise_stddev,
                                             use_attention=use_attention, use_ConvLSTM=use_convLSTM,
                                             use_memory_restriction=use_memory_restriction,
                                             use_encoded_output=loss_use_batch_sim_siam, adj_mat = adj_mat)
    else:
        model_MSCRED = MSCRED().create_model(guassian_noise_stddev=guassian_noise_stddev,
                                         use_attention=use_attention, use_ConvLSTM=use_convLSTM,
                                         use_memory_restriction=use_memory_restriction, use_encoded_output=loss_use_batch_sim_siam)
    print(model_MSCRED.summary())
    tf.keras.utils.plot_model(model_MSCRED, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    ### Train the model
    if train_model:
        print('-------------------------------')
        print('Start NN for training dataset ')
        #path = os.path.abspath(".") + config.datasets[config.no_failure][0][2::] + config.directoryname_training_data
        print("Epochs: ",epochs," |  Batch size: ",batch_size," | Step Size: ", step_size," | Learning rate: ",learning_rate)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)
        mc = tf.keras.callbacks.ModelCheckpoint('best_model_'+curr_run_identifier+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        #  Loading the training data
        curr_xTrainData = np.load(training_data_set_path)
        if config.use_data_set_version == 2022:
            # change format from: (example_dim, 4, len(win_size), x_features.shape[2], x_features.shape[2]))
            curr_xTrainData = np.transpose(curr_xTrainData, [0,1,3,4,2])
            print("Training failure-free data swapped acc. old dimensional order (examples,lenght,sensor,sensor,dim")

        print("Loaded (failure-free training) data: ", curr_xTrainData.shape)

        #Remove irrelevant correlations
        curr_xTrainData = apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input, use_corr_rel_matrix_for_input_replace_by_epsilon,curr_xTrainData,np_corr_rel_matrix)

        X_train, X_valid = model_selection.train_test_split(curr_xTrainData, test_size=split_train_test_ratio, random_state=42)
        #X_train = curr_xTrainData[0:90,:,:,:,:]
        #X_valid = curr_xTrainData[90:99, :, :, :, :]
        print("Splitted with ratio: ",split_train_test_ratio," into X_train: ", X_train.shape, " and X_valid: ", X_valid.shape)
        np.save(valid_split_save_path,X_valid)
        print("Test split saved as: ", valid_split_save_path )

        # Generating labels / desired output: Reducing input to the last time step
        X_train_y = X_train[:,step_size-1,:,:,:]
        X_valid_y = X_valid[:,step_size-1,:,:,:]
        print("Size of desired output: X_train_y: ", X_train_y.shape, " and X_valid_y: ", X_valid_y.shape)

        # Training of the model
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        csv_logger = tf.keras.callbacks.CSVLogger('trainingLog_'+curr_run_identifier+'.log')

        if use_memory_restriction:
            if use_loss_corr_rel_matrix:
                if use_MemEntropyLoss:
                    model_MSCRED.compile(optimizer=opt,
                                         loss=[corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix),
                                               MemEntropyLoss],
                                         loss_weights=[0.9998, 0.0002]) # [0.9998, 0.0002] # [0.99999999, 0.00000001]
                else:
                    model_MSCRED.compile(optimizer=opt,
                                         loss=corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix))  # [0.9998, 0.0002]
            else:
                if use_MemEntropyLoss:
                    model_MSCRED.compile(optimizer=opt, loss=[mscred_loss, MemEntropyLoss], loss_weights=[0.9998, 0.0002]) # [0.9998, 0.0002]
                else:
                    model_MSCRED.compile(optimizer=opt, loss=mscred_loss)  # [0.9998, 0.0002]
        elif use_loss_corr_rel_matrix:
            model_MSCRED.compile(optimizer=opt, loss=corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix))
        elif loss_use_batch_sim_siam:
            model_MSCRED.compile(optimizer=opt, loss=[mscred_loss, SimLoss],loss_weights=[0.9, 0.1])
        else:
            model_MSCRED.compile(optimizer=opt, loss=mscred_loss)

        # Adj Mat Test
        if config.use_graph_conv == True:
            adj_mat = np.ones((1000,3721,3721))
            print("adj_mat shape: ", adj_mat)
            history = model_MSCRED.fit([X_train[:1000,:,:,:],adj_mat],X_train_y[:1000,:,:,:],epochs=epochs,batch_size=batch_size, shuffle=True, validation_split=0.1, callbacks=[es, mc, csv_logger])
        else:
            history = model_MSCRED.fit(X_train, X_train_y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1, callbacks=[es, mc, csv_logger])
        plot_training_process_history(history=history, curr_run_identifier=curr_run_identifier)

    ### Test ###
    if test_model:
        f1_all_list = []
        rocauc_mse_all_list = []
        rocauc_attr_all_list = []
        rocauc_attrDim_all_list = []
        #Load previous trained model
        if use_memory_restriction or loss_use_batch_sim_siam:
            model_MSCRED = tf.keras.models.load_model('best_model_' + curr_run_identifier + '.h5', custom_objects={
                'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix), 'Memory': Memory,'Memory2': Memory2}, compile=False)
        else:
            model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', custom_objects={'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix), 'Memory': Memory})

        print("Pretrained Model loaded ...")

        # Load validation split of the training data (used for defining the anomaly thresholds)
        X_valid = np.load(valid_split_save_path)

        # Load data with failures
        X_valid_wF = np.load(valid_matrix_path_wF)
        if config.use_data_set_version == 2022:
            X_valid_wF = np.transpose(X_valid_wF, [0, 1, 3, 4, 2])
        valid_labels_y = np.load(valid_labels_y_path_wF)

        # Remove irrelevant correlations
        X_valid = apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input,
                                                         use_corr_rel_matrix_for_input_replace_by_epsilon,
                                                         X_valid, np_corr_rel_matrix)
        X_valid_wF = apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input,
                                                 use_corr_rel_matrix_for_input_replace_by_epsilon,
                                                 X_valid_wF, np_corr_rel_matrix)

        X_valid_y = X_valid[:, step_size - 1, :, :, :]
        X_valid_wF_y = X_valid_wF[:, step_size - 1, :, :, :]

        # Load test data (as used in PredM Siamese NN)
        X_test = np.load(test_matrix_path)
        if config.use_data_set_version == 2022:
            X_test = np.transpose(X_test, [0, 1, 3, 4, 2])
        X_test = apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input,
                                                 use_corr_rel_matrix_for_input_replace_by_epsilon,
                                                 X_test, np_corr_rel_matrix)
        X_test_y = X_test[:, step_size - 1, :, :, :]

        # Load failure examples of the training data (failure examples excluded from the training) for further evaluation?

        print("Validation split: ", X_valid.shape,"Validation split with Failure: ", X_valid_wF.shape, "Test data:", X_test.shape)

        # Reconstruct loaded input data
        print("Data for evaluation loaded  ... start with reconstruction ...")
        if loss_use_batch_sim_siam or use_memory_restriction:
            #b = Batch(51, 97)
            b = Batch(23, 99)
            for i in range(23):
                if i == 0:
                    if use_MemEntropyLoss:
                        output = model_MSCRED.predict(X_valid[b.next(), :, :, :, :])
                        X_valid_recon = output[0]
                        X_valid_memAccess = output[1]
                    else:
                        X_valid_recon = model_MSCRED.predict(X_valid[b.next(), :, :, :, :])
                else:
                    if use_MemEntropyLoss:
                        #X_valid_recon = np.concatenate((X_valid_recon, model_MSCRED.predict(X_valid[b.next(),:,:,:,:])[0]), axis=0)
                        output = model_MSCRED.predict(X_valid[b.next(),:,:,:,:])
                        X_valid_recon = np.concatenate((X_valid_recon, output[0]), axis=0)
                        X_valid_memAccess = np.concatenate((X_valid_memAccess, output[1]), axis=0)
                    else:
                        X_valid_recon = np.concatenate((X_valid_recon, model_MSCRED.predict(X_valid[b.next(), :, :, :, :])), axis=0)
                #X_valid_recon = model_MSCRED.predict(X_valid[b.next(),:,:,:,:])[0]
            #X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)[0]
        else:
            X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)
        print("Reconstruction of validation data set done with shape :", X_valid_recon.shape) #((9, 61, 61, 3))
        if loss_use_batch_sim_siam or use_memory_restriction:
            #b = Batch(51, 97)
            b = Batch(3, 127)
            for i in range(3):
                if i == 0:
                    if use_MemEntropyLoss:
                        output = model_MSCRED.predict(X_valid_wF[b.next(), :, :, :, :])
                        X_valid_recon_wf = output[0]
                        X_valid_memAccess = output[1]
                    else:
                        X_valid_recon_wf = model_MSCRED.predict(X_valid_wF[b.next(), :, :, :, :])
                else:
                    if use_MemEntropyLoss:
                        #X_valid_recon = np.concatenate((X_valid_recon, model_MSCRED.predict(X_valid[b.next(),:,:,:,:])[0]), axis=0)
                        output = model_MSCRED.predict(X_valid_wF[b.next(),:,:,:,:])
                        X_valid_recon_wf = np.concatenate((X_valid_recon_wf, output[0]), axis=0)
                        X_valid_memAccess = np.concatenate((X_valid_memAccess, output[1]), axis=0)
                    else:
                        X_valid_recon_wf = np.concatenate((X_valid_recon_wf, model_MSCRED.predict(X_valid[b.next(), :, :, :, :])), axis=0)
                #X_valid_recon = model_MSCRED.predict(X_valid[b.next(),:,:,:,:])[0]
            #X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)[0]
        else:
            X_valid_recon_wf = model_MSCRED.predict(X_valid_wF, batch_size=128)
        print("Reconstruction of validation data with Failures set done with shape :", X_valid_recon_wf.shape) #((9, 61, 61, 3))
        if loss_use_batch_sim_siam or use_memory_restriction:
            #b = Batch(44, 149) # https://www.matheretter.de/rechner/primzahltest
            b = Batch(28, 121) # https://www.matheretter.de/rechner/primzahltest
            for i in range(28):
                if i == 0:
                    if use_MemEntropyLoss:
                        output = model_MSCRED.predict(X_test[b.next(), :, :, :, :])
                        X_test_recon = output[0]
                        X_test_memAccess = output[1]

                    else:
                        output = model_MSCRED.predict(X_test[b.next(), :, :, :, :])
                        X_test_recon = output
                else:
                    if use_MemEntropyLoss:
                        output = model_MSCRED.predict(X_test[b.next(), :, :, :, :])
                        X_test_recon = np.concatenate((X_test_recon, output[0]), axis=0)
                        X_test_memAccess = np.concatenate((X_test_memAccess, output[1]), axis=0)
                    else:
                        output = model_MSCRED.predict(X_test[b.next(), :, :, :, :])
                        X_test_recon = np.concatenate((X_test_recon, output), axis=0)
            #output = model_MSCRED.predict(X_test[:128,:,:,:,:], batch_size=128)
            #X_test_recon = output[0]
            #X_test_memAccess = output[1]
            print("X_test_recon shape:",X_test_recon.shape)
            if use_MemEntropyLoss: print("X_test_memAccess shape: ", X_test_memAccess.shape)
        else:
            X_test_recon = model_MSCRED.predict(X_test, batch_size=128)
        print("Reconstruction of test data set done with shape :", X_test_recon.shape)  # ((9, 61, 61, 3))

        # Generate deep encodings
        # Dummy model for obtaining access to latent encodings / space
        if generate_deep_encodings:
            layer_name = 'Reshape_ToOrignal_ConvLSTM_4'
            intermediate_layer_model = tf.keras.Model(inputs=model_MSCRED.input,
                                                   outputs=model_MSCRED.get_layer(layer_name).output)
            encoded_output_test = intermediate_layer_model.predict(X_test, batch_size=128)
            encoded_output_valid_wf = intermediate_layer_model.predict(X_valid_wF, batch_size=128)
            print("Encoded_output_valid_wf shape:", encoded_output_valid_wf.shape,"Encoded_output_test shape:", encoded_output_test.shape)
            np.save('encoded_test.npy', encoded_output_test)
            np.save('encoded_output_valid_wf.npy', encoded_output_valid_wf)

        feature_names = np.load(feature_names_path)

        # Remove any dimension with size of 1
        X_valid_y = np.squeeze(X_valid_y)
        X_valid_wF_y = np.squeeze(X_valid_wF_y)
        X_test_y = np.squeeze(X_test_y)

        '''
        x = get_Similarity_Matrix(X_valid_memAccess, X_test_memAccess)
        print(x)
        print(x.shape)
        x_ = np.mean(x, axis=1)
        print(x_)
        print(x_.shape)
        cosine_sim_mean_norm = (x_ - np.min(x_)) / np.ptp(x_)
        y_true = pd.factorize(test_failure_labels_y)[0].tolist()
        y_true = np.where(np.asarray(y_true) > 1, 1,0)
        score = roc_auc_score(y_true, cosine_sim_mean_norm)
        print("Roc-Auc_score based on cosine sim betw. valid of memory access: ", score)
        '''

        test_configurations = list(zip(threshold_selection_criterium_list, num_of_dim_over_threshold_list,
                                       num_of_dim_under_threshold_list, use_corr_rel_matrix_in_eval_list, use_attribute_anomaly_as_condition_list))
        #for i in range(len(test_configurations)):

        threshold_selection_criterium, num_of_dim_over_threshold, num_of_dim_under_threshold, use_corr_rel_matrix_in_eval,\
        use_attribute_anomaly_as_condition = test_configurations[0][0], test_configurations[0][1], test_configurations[0][2], test_configurations[0][3], test_configurations[0][4]
        print("threshold_selection_criterium: ", threshold_selection_criterium, " with: ",num_of_dim_over_threshold,"/",num_of_dim_under_threshold, ". CorrMatrix: ", use_corr_rel_matrix_in_eval,"Anomaly based on Attributes: ", use_attribute_anomaly_as_condition)

        ### Calcuation of reconstruction error on the validation data set ###
        recon_err_matrixes_valid, recon_err_perAttrib_valid, mse_per_example_valid, mse_per_example_per_dims = calculateReconstructionError(
            real_input=X_valid_y, reconstructed_input=X_valid_recon, plot_heatmaps=plot_heatmap_of_rec_error,
            use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix)


        ### Calcuation of reconstruction error on the validation data set with Failures###
        recon_err_matrixes_valid_wf, recon_err_perAttrib_valid_wf, mse_per_example_valid_wf, mse_per_example_per_dims_wf = calculateReconstructionError(
            real_input=X_valid_wF_y, reconstructed_input=X_valid_recon_wf, plot_heatmaps=plot_heatmap_of_rec_error,
            use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix)
        print("recon_err_matrixes_valid_wf:", recon_err_matrixes_valid_wf.shape,"recon_err_perAttrib_valid_wf:", recon_err_perAttrib_valid_wf.shape,"mse_per_example_valid_wf:",mse_per_example_valid_wf.shape ,"mse_per_example_per_dims_wf:", mse_per_example_per_dims_wf.shape)
        # recon_err_matrixes_valid_wf: (381, 61, 61, 4) recon_err_perAttrib_valid_wf: (381, 122, 4) mse_per_example_valid_wf: (381,) mse_per_example_per_dims_wf: (381, 4)

        print("X_test_y shape:",X_test_y.shape,"X_test_recon shape:",X_test_recon.shape)
        ### Calcuation of reconstruction error on the test data set ###
        recon_err_matrixes_test, recon_err_perAttrib_test, mse_per_example_test, mse_per_example_per_dims = calculateReconstructionError(
            real_input=X_test_y, reconstructed_input=X_test_recon, plot_heatmaps=plot_heatmap_of_rec_error,
            use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix, y_labels=test_labels_y)

        '''
        This is already done in the methode calculateReconstructionError
        if use_corr_rel_matrix_on_masking_residual_matrices:
            #use_corr_rel_matrix_for_input_replace_by_epsilon,input_data,np_corr_rel_matrix
            recon_err_matrixes_valid = apply_corr_rel_matrix_on_residual_matrix(use_corr_rel_matrix_for_input_replace_by_epsilon, recon_err_matrixes_valid, np_corr_rel_matrix)
            recon_err_matrixes_valid_wf = apply_corr_rel_matrix_on_residual_matrix(use_corr_rel_matrix_for_input_replace_by_epsilon, recon_err_matrixes_valid_wf, np_corr_rel_matrix)
            recon_err_matrixes_test = apply_corr_rel_matrix_on_residual_matrix(use_corr_rel_matrix_for_input_replace_by_epsilon, recon_err_matrixes_test, np_corr_rel_matrix)
        '''

        # Define Thresholds for each dimension and attribute
        ''' # Only faultfree data and a fixed threshold selection criterium is used
        thresholds, mse_threshold = calculateThreshold(reconstructed_input=X_valid_recon,recon_err_perAttrib_valid=recon_err_perAttrib_valid,
                                                               threshold_selection_criterium=threshold_selection_criterium, mse_per_example=mse_per_example_valid,
                                                       print_pandas_statistics_for_validation=print_pandas_statistics_for_validation, feature_names=feature_names)
        print("thresholds",thresholds.shape, "mse_threshold:",mse_threshold.shape)
        print("calculateThreshold with X_valid_recon: mse_threshold:", mse_threshold, "thresholds:", thresholds,)
        # thresholds (4, 122) mse_threshold: (1,)
        '''
        scaler_dict = {}
        recon_err_matrixes_valid_axis1 = recon_err_matrixes_valid.copy()
        recon_err_matrixes_valid_axis2 = recon_err_matrixes_valid.copy()
        recon_err_matrixes_valid_wf_axis1 = recon_err_matrixes_valid_wf.copy()
        recon_err_matrixes_valid_wf_axis2 = recon_err_matrixes_valid_wf.copy()
        recon_err_matrixes_test_axis1 = recon_err_matrixes_test.copy()
        recon_err_matrixes_test_axis2 = recon_err_matrixes_test.copy()
        if normalize_residual_matrices:
            for dim in range(recon_err_matrixes_valid.shape[3]):
                # makes mean to zero and std to 1 for each attribute.
                recon_err_matrixes_valid_ = np.reshape(recon_err_matrixes_valid,(recon_err_matrixes_valid.shape[0], recon_err_matrixes_valid.shape[1] * recon_err_matrixes_valid.shape[2], recon_err_matrixes_valid.shape[3]))
                #scaler = preprocessing.StandardScaler().fit(recon_err_matrixes_valid_[:, :, dim])
                scaler = preprocessing.Normalizer().fit(recon_err_matrixes_valid_[:, :, dim])
                #scaler = preprocessing.MinMaxScaler().fit(recon_err_matrixes_valid_[:, :, dim])

                # Process Validation data ...
                recon_err_matrixes_valid_wf_ = np.reshape(recon_err_matrixes_valid_wf,(recon_err_matrixes_valid_wf.shape[0], recon_err_matrixes_valid_wf.shape[1] * recon_err_matrixes_valid_wf.shape[2],recon_err_matrixes_valid_wf.shape[3]))
                recon_err_matrixes_valid_wf_scaled = scaler.transform(recon_err_matrixes_valid_wf_[:, :, dim])
                recon_err_matrixes_valid_wf_ = np.reshape(recon_err_matrixes_valid_wf_scaled, (recon_err_matrixes_valid_wf.shape[0], recon_err_matrixes_valid_wf.shape[1], recon_err_matrixes_valid_wf.shape[2]))
                recon_err_matrixes_valid_wf[:, :, :, dim] = recon_err_matrixes_valid_wf_

                # Process test data ...
                recon_err_matrixes_test_ = np.reshape(recon_err_matrixes_test, (
                recon_err_matrixes_test.shape[0],
                recon_err_matrixes_test.shape[1] * recon_err_matrixes_test.shape[2],
                recon_err_matrixes_test.shape[3]))
                recon_err_matrixes_test_scaled = scaler.transform(recon_err_matrixes_test_[:, :, dim])
                recon_err_matrixes_test_ = np.reshape(recon_err_matrixes_test_scaled, (
                recon_err_matrixes_test.shape[0], recon_err_matrixes_test.shape[1],
                recon_err_matrixes_test.shape[2]))
                recon_err_matrixes_test[:, :, :, dim] = recon_err_matrixes_test_
                #recon_err_perAttrib_valid_scaled = np.squeeze(recon_err_perAttrib_valid_scaled)
                # print("data_curr_dim shape: ", data_curr_dim.shape)
                #recon_err_perAttrib_valid_scaled = pd.DataFrame(recon_err_perAttrib_valid_scaled)
                #print(recon_err_perAttrib_valid_scaled.describe().loc['mean', :].values)
                #print(recon_err_perAttrib_valid_scaled.describe().loc['std', :].values)
                scaler_dict[dim] = scaler
        #'''
        dict_results = {}
        #scale attributewise
        '''
        for dim in range(recon_err_matrixes_valid.shape[3]):
            for att in range(recon_err_matrixes_valid.shape[2]):
                recon_err_matrixes_valid_dim_att = recon_err_matrixes_valid_axis1[:, att,:, dim]
                #print("recon_err_matrixes_valid_dim_att shape:",recon_err_matrixes_valid_dim_att.shape)
                recon_err_matrixes_valid_dim_att = np.squeeze(recon_err_matrixes_valid_dim_att) # to get (examples,features)
                #print("recon_err_matrixes_valid_dim_att shape:", recon_err_matrixes_valid_dim_att.shape)
                scaler = preprocessing.Normalizer().fit(recon_err_matrixes_valid_dim_att)
                recon_err_matrixes_valid_dim_att_scaled = scaler.transform(recon_err_matrixes_valid_dim_att)
                recon_err_matrixes_valid_axis1[:, att,:, dim] = recon_err_matrixes_valid_dim_att_scaled

                # Process Validation data ...
                recon_err_matrixes_valid_wf_dim_att = recon_err_matrixes_valid_wf_axis1[:, att, :, dim]
                recon_err_matrixes_valid_wf_dim_att_scaled = scaler.transform(recon_err_matrixes_valid_wf_dim_att)
                recon_err_matrixes_valid_wf_axis1[:, att, :, dim] = recon_err_matrixes_valid_wf_dim_att_scaled

                # Process test data ...
                recon_err_matrixes_test_dim_att = recon_err_matrixes_test_axis1[:, att, :, dim]
                recon_err_matrixes_test_dim_att_scaled = scaler.transform(recon_err_matrixes_test_dim_att)
                recon_err_matrixes_test_axis1[:, att, :, dim] = recon_err_matrixes_test_dim_att_scaled

            for att in range(recon_err_matrixes_valid.shape[2]):
                recon_err_matrixes_valid_dim_att = recon_err_matrixes_valid_axis2[:, :,att, dim]
                #print("recon_err_matrixes_valid_dim_att shape:",recon_err_matrixes_valid_dim_att.shape)
                recon_err_matrixes_valid_dim_att = np.squeeze(recon_err_matrixes_valid_dim_att) # to get (examples,features)
                #print("recon_err_matrixes_valid_dim_att shape:", recon_err_matrixes_valid_dim_att.shape)
                scaler = preprocessing.Normalizer().fit(recon_err_matrixes_valid_dim_att)
                recon_err_matrixes_valid_dim_att_scaled = scaler.transform(recon_err_matrixes_valid_dim_att)
                recon_err_matrixes_valid_axis2[:, :,att, dim] = recon_err_matrixes_valid_dim_att_scaled

                # Process Validation data ...
                recon_err_matrixes_valid_wf_dim_att = recon_err_matrixes_valid_wf_axis2[:, :, att, dim]
                recon_err_matrixes_valid_wf_dim_att_scaled = scaler.transform(recon_err_matrixes_valid_wf_dim_att)
                recon_err_matrixes_valid_wf_axis2[:, :, att, dim] = recon_err_matrixes_valid_wf_dim_att_scaled

                # Process test data ...
                recon_err_matrixes_test_dim_att = recon_err_matrixes_test_axis2[:, :, att, dim]
                recon_err_matrixes_test_dim_att_scaled = scaler.transform(recon_err_matrixes_test_dim_att)
                recon_err_matrixes_test_axis2[:, :, att, dim] = recon_err_matrixes_test_dim_att_scaled
        '''
        # f1_weighted_max_threshold, [f1_weighted_max_threshold_per_dim, best_dim], [f1_weighted_max_threshold_per_dim_attr,best_dim_attr], thresholds
        '''
        f1_weighted_max_threshold, f1_weighted_max_threshold_per_dim, f1_weighted_max_threshold_per_dim_attr, thresholds_percentil  = calculateThresholdWithLabels(reconstructed_input=X_valid_recon_wf,recon_err_perAttrib_valid=recon_err_perAttrib_valid_wf, labels_valid=valid_labels_y,
                                                               mse_per_example=mse_per_example_valid_wf, print_pandas_statistics_for_validation=print_pandas_statistics_for_validation, feature_names=feature_names,
                                                               mse_per_example_per_dims_wf=mse_per_example_per_dims_wf,scaler_dict=scaler_dict)
        print("thresholds_percentil", thresholds_percentil.shape, "f1_weighted_max_threshold:", f1_weighted_max_threshold.shape)
        print("calculateThresholdWithLabels with X_valid_recon_wf: f1_weighted_max_threshold:", f1_weighted_max_threshold, "thresholds_percentil:", thresholds_percentil)
        '''

        anomaly_score_per_example_test, best_threshold_tau,dict_results = calculateThresholdWithLabelsAsMSCRED(residual_matrix=recon_err_matrixes_valid_wf, labels_valid=valid_labels_y,residual_matrix_test=recon_err_matrixes_test, residual_matrix_wo_FaF=recon_err_matrixes_valid, attr_names=feature_names,curr_run_identifier=curr_run_identifier, dict_results=dict_results)
        '''
        anomaly_score_per_example_test, best_threshold_tau = calculateThresholdWithLabelsAsMSCRED_Scaled(residual_matrix_scaled=recon_err_matrixes_valid_wf, residual_matrix_scaled_axis1=recon_err_matrixes_valid_wf_axis1, residual_matrix_scaled_axis2=recon_err_matrixes_valid_wf_axis2,
                                                                                                         labels_valid=valid_labels_y,residual_matrix_test=recon_err_matrixes_test, residual_matrix_wo_FaF=recon_err_matrixes_valid, attr_names=feature_names,
                                                                                                         residual_matrix_test_axis1=recon_err_matrixes_test_axis1,residual_matrix_test_axis2=recon_err_matrixes_test_axis2s,curr_run_identifier=curr_run_identifier)
        '''
        print("anomaly_score_per_example_test shape: ", anomaly_score_per_example_test.shape,"test_labels_y shape:", test_labels_y.shape)

        dict_results = evaluate(anomaly_score_per_example_test, test_labels_y, best_threshold_tau, average='weighted', curr_run_identifier=curr_run_identifier,dict_results=dict_results)

        return dict_results


if __name__ == '__main__':
    dict_measures_collection = {}
    num_of_runs = 5

    for run in range(num_of_runs):
        print(" ############## START OF RUN "+str(run)+" ##############")
        print()
        dict_measures = main(run)
        dict_measures_collection[run] = dict_measures
        print()
        print(" ############## END OF RUN " + str(run) + " ##############")

    print()
    print("Saved data:")
    print("dict_measures_collection: ", dict_measures_collection)
    print()
    mean_dict_0 = {}
    for key in dict_measures_collection[0]:
        mean_dict_0[key] = []

    #print("mean_dict_0: ", mean_dict_0)

    for i in range(num_of_runs):
        for key in dict_measures_collection[i]:
            #print("key: ", key)
            mean_dict_0[key].append(dict_measures_collection[i][key])

    # print("mean_dict_0: ", mean_dict_0)
    print("### FINAL RESULTS OF " +str(num_of_runs) + " RUNS ###")
    print("Key;Mean;Std")
    # compute mean
    dict_mean = {}
    for key in mean_dict_0:
        mean = np.mean(mean_dict_0[key])
        std = np.std(mean_dict_0[key], axis=0)
        print(key, ";", mean, ";", std)
