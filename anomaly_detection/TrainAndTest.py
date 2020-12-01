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
#from mscred import MemoryInstanceBased
from configuration.Configuration import Configuration


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
        d1 = dict(zip(df_curr_dim_described.loc[threshold_selection_criterium, :].values, feature_names))
        #print(i_dim, ": ", df_curr_dim_described.loc[threshold_selection_criterium, :].values)
        #print("d1: ", d1)
        #arr_index = np.where(feature_names == 'txt16_i4')
        #print("INDEX txt16_i4: ", arr_index)
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

# This method computes reconstruction errors
def calculateReconstructionError(real_input, reconstructed_input, plot_heatmaps, use_corr_rel_matrix=False, corr_rel_matrix=None):
    reconstruction_error_matrixes = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1], reconstructed_input.shape[2], reconstructed_input.shape[3]))
    reconstruction_errors_perAttribute = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1] + reconstructed_input.shape[2], reconstructed_input.shape[3]))
    mse_per_example_over_all_dims = np.zeros((reconstructed_input.shape[0]))
    mse_per_example_per_dims = np.zeros((reconstructed_input.shape[0],reconstructed_input.shape[3]))

    for i_example in range(reconstructed_input.shape[0]):  # Iterate over all examples
        for i_dim in range(reconstructed_input.shape[3]):  # Iterate over all "time-dimensions"
            # Get reconstructed and real input data
            curr_matrix_input_recon = reconstructed_input[i_example, :, :, i_dim]
            curr_matrix_input_real = real_input[i_example, :, :, i_dim]
            # Calculate the reconstruction error
            diff = curr_matrix_input_recon - curr_matrix_input_real
            if use_corr_rel_matrix:
                diff = diff * corr_rel_matrix
            # print("curr_matrix_input_recon shape: ", curr_matrix_input_recon.shape)
            reconstruction_error_matrixes[i_example, :, :, i_dim] = np.square(diff)
            '''
            plot_heatmap_of_reconstruction_error(id=i_example, dim=i_dim, input=curr_matrix_input_real,
                                                 output=curr_matrix_input_recon,
                                                 rec_error_matrix=reconstruction_error_matrixes[i_example,:,:,i_dim])
            '''
            # Alternative variants for calculation of reconstruction error
            mse = np.mean(np.square(diff))
            diff_paper_formula = np.square(np.linalg.norm(diff, ord='fro'))
            # diff_paper_formula_axis0 = np.square(np.linalg.norm(diff, ord='fro', axis=0))
            # diff_paper_formula_axis1 = np.square(np.linalg.norm(diff, ord='fro', axis=1))
            mse_axis0 = np.mean(np.square(diff), axis=0)
            mse_axis1 = np.mean(np.square(diff), axis=1)
            reconstruction_errors_perAttribute[i_example, :reconstructed_input.shape[1], i_dim] = mse_axis0
            reconstruction_errors_perAttribute[i_example, reconstructed_input.shape[1]:, i_dim] = mse_axis1
            mse_per_example_over_all_dims[i_example] = mse_per_example_over_all_dims[i_example] + mse
            mse_per_example_per_dims[i_example, i_dim] = mse
            #print("example: ", i_example, "dim: ", i_dim, "Rec.Err.: ", diff_paper_formula, "MSE: ", mse)
            # print("axis0: ", mse_axis0)
            # print("axis0: ", mse_axis0.shape)
            # print("reconstruction_errors_perAttribute:", reconstruction_errors_perAttribute[i_example, :, i_dim])
        if plot_heatmaps:
            plot_heatmap_of_reconstruction_error2(id=i_example, input=real_input[i_example, :, :, :],
                                                  output=reconstructed_input[i_example, :, :, :],
                                                  rec_error_matrix=reconstruction_error_matrixes[i_example, :, :, :])
    # Durch Anzahl an Dimensionen teilen
    mse_per_example_over_all_dims = mse_per_example_over_all_dims / reconstructed_input.shape[3]

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

        if example_idx == reconstructed_input.shape[0]-1:
            print("count_dim_anomalies_1: ", count_dim_anomalies_1)
            print("count_dim_anomalies_2: ", count_dim_anomalies_2)
            print("idx_with_Anomaly_1: ", idx_with_Anomaly_1)
            print("idx_with_Anomaly_2: ", idx_with_Anomaly_2)
            print("feature_names 1: ",feature_names[idx_with_Anomaly_1])
            print("feature_names 2: ", feature_names[idx_with_Anomaly_2])
            print("eval_results_over_all_dimensions 1: ", eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][23])
            print("eval_results_over_all_dimensions 2: ", eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][23])

        #Get ordered and combined dictonary of anomalous data streams
        anomalies_combined_asc_ordered = order_anomalies(count_dim_anomalies_1, count_dim_anomalies_2,idx_with_Anomaly_1, idx_with_Anomaly_2, feature_names)
        anomaly_detected = False


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
            if labels is not None:
                if labels[example_idx] == 'no_failure':
                    FN_NF += 1
                    FP_F += 1
                else:
                    TN_NF += 1
                    TP_F += 1
        else:
            # No Anomaly detected
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
        print("------------------------------------------------------------------------")
        print("OverAll Acc: \t\t %.3f" % acc_A)
        print("OverAll Precision: \t %.3f" % prec_A)
        print("OverAll Recall: \t %.3f" %rec_A)
        print("OverAll F1: \t \t %.3f" % f1_A)
        print("------------------------------------------------------------------------")

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
def plot_heatmap_of_reconstruction_error2(input, output, rec_error_matrix, id):
    #print("example id: ", id)
    fig, axs = plt.subplots(3,8, gridspec_kw = {'wspace':0.1, 'hspace':-0.1})
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

    for dim in range(8):
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
    filename="heatmaps3/" +str(id)+"-"+"rec_error_matrix.png"
    #print("filename: ", filename)
    fig.tight_layout(pad=0.1, h_pad=0.01)
    #fig.subplots_adjust(wspace=1.1,hspace=-0.1)
    #plt.subplots_adjust(bottom=0.1,top=0.2, hspace=0.1)
    plt.savefig(filename, dpi=500)
    plt.clf()
    #print("plot_heatmap_of_reconstruction_error")

# Loss Function that receives an external matrix where relevant correlations are defined manually (based on domain
# knowledge)
# Input: corr_rel_mat (Attributes,Attributes) with {0,1}
def corr_rel_matrix_weighted_loss(corr_rel_mat):
    def loss(y_true, y_pred):
        loss = tf.square(y_true - y_pred)
        print("corr_rel_matrix_weighted_loss loss dim: ", loss.shape)
        #if use_corr_rel_matrix:
        #loss = loss * corr_rel_mat
        loss = tf.reduce_sum(loss, axis=-1)
        loss = loss / np.sum(corr_rel_mat) # normalize by number of considered correlations
        #loss = tf.reduce_mean(loss, axis=-1)
        return loss
    return loss  # Note the `axis=-1`

# Loss function according to the MSCRED paper, but in the official impl. MSE is used
def mscred_loss_acc_paper(y_true, y_pred):
    # MSCRED loss according to paper
    # squared_difference = tf.square(y_true - y_pred)
    loss = np.square(np.linalg.norm((y_true - y_pred), ord='fro'))
    print("loss dim: ", loss.shape)
    return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`

def MseLoss(y_true, y_pred):
    loss = tf.square(y_true - y_pred)
    print("MseLoss loss dim: ", loss.shape)
    #loss = np.square(np.linalg.norm((y_true - y_pred[0]), ord='fro'))
    #K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)
    loss = tf.reduce_mean(loss, axis=-1)
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
    mem_etrp = tf.reduce_sum((-y_pred) * tf.math.log(y_pred + 1e-12))
    #print("mem_etrp shape: ", mem_etrp.shape)
    loss = tf.reduce_mean(mem_etrp)
    print("MemEntropyLoss loss dim: ", loss.shape)
    return loss

def plot_training_process_history(history, curr_run_identifier):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    # summarize history for loss
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

def main():
    # Configurations
    train_model = config.train_model
    test_model = config.test_model

    # Variants of the MSCRED
    guassian_noise_stddev = config.guassian_noise_stddev
    use_attention = config.use_attention
    use_convLSTM = config.use_convLSTM
    use_memory_restriction = config.use_memory_restriction

    use_loss_corr_rel_matrix = config.use_loss_corr_rel_matrix
    loss_use_batch_sim_siam = config.loss_use_batch_sim_siam
    use_corr_rel_matrix_for_input = config.use_corr_rel_matrix_for_input
    use_corr_rel_matrix_for_input_replace_by_epsilon = config.use_corr_rel_matrix_for_input_replace_by_epsilon
    plot_heatmap_of_rec_error = config.plot_heatmap_of_rec_error
    curr_run_identifier = config.curr_run_identifier
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

    use_mass_evaulation = config.use_mass_evaulation
    threshold_selection_criterium_list = config.threshold_selection_criterium_list
    num_of_dim_over_threshold_list = config.num_of_dim_over_threshold_list
    num_of_dim_under_threshold_list = config.num_of_dim_under_threshold_list
    use_corr_rel_matrix_in_eval_list = config.use_corr_rel_matrix_in_eval_list
    use_attribute_anomaly_as_condition_list = config.use_attribute_anomaly_as_condition_list
    use_dim_for_anomaly_detection = config.use_dim_for_anomaly_detection

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

    ### Create MSCRED model as TF graph
    print("config.relevant_features: ", config.relevant_features)
    test_failure_labels_y = np.load(test_labels_y_path)
    ### Load Correlation Relevance Matrix ###
    df_corr_rel_matrix = pd.read_csv('../data/Attribute_Correlation_Relevance_Matrix_v0.csv', sep=';',index_col=0)
    np_corr_rel_matrix = df_corr_rel_matrix.values
    print("np_corr_rel_matrix shape: ", np_corr_rel_matrix.shape)
    print("Only ", np.sum(np_corr_rel_matrix)," of ",np_corr_rel_matrix.shape[0] * np_corr_rel_matrix.shape[1]," correlations")

    print('-------------------------------')
    print('Creation of the model')

    # create graph structure of the NN
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
                model_MSCRED.compile(optimizer=opt,
                                     loss=[corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix),
                                           MemEntropyLoss],
                                     loss_weights=[0.9998, 0.0002])
            else:
                model_MSCRED.compile(optimizer=opt, loss=[MseLoss, MemEntropyLoss],
                                     loss_weights=[0.9998, 0.0002])
        elif use_loss_corr_rel_matrix:
            model_MSCRED.compile(optimizer=opt, loss=corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix))
        elif loss_use_batch_sim_siam:
            model_MSCRED.compile(optimizer=opt, loss=[MseLoss, SimLoss],
                                 loss_weights=[0.9, 0.1])
        else:
            model_MSCRED.compile(optimizer=opt, loss=tf.keras.losses.mse)
        history = model_MSCRED.fit(X_train,X_train_y,epochs=epochs,batch_size=batch_size, shuffle=True, validation_split=0.1, callbacks=[es, mc, csv_logger])
        plot_training_process_history(history=history, curr_run_identifier=curr_run_identifier)

    ### Test ###
    if test_model:
        #Load previous trained model
        if use_memory_restriction or loss_use_batch_sim_siam:
            model_MSCRED = tf.keras.models.load_model('best_model_' + curr_run_identifier + '.h5', custom_objects={
                'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix), 'Memory': Memory}, compile=False)
        else:
            model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', custom_objects={'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix), 'Memory': Memory})

        print("Pretrained Model loaded ...")

        # Load validation split of the training data (used for defining the anomaly thresholds)
        X_valid = np.load(valid_split_save_path)
        # Remove irrelevant correlations
        X_valid = apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input,
                                                         use_corr_rel_matrix_for_input_replace_by_epsilon,
                                                         X_valid, np_corr_rel_matrix)
        X_valid_y = X_valid[:, step_size - 1, :, :, :]

        # Load test data (as used in PredM Siamese NN)
        X_test = np.load(test_matrix_path)
        X_test = apply_corr_rel_matrix_on_input(use_corr_rel_matrix_for_input,
                                                 use_corr_rel_matrix_for_input_replace_by_epsilon,
                                                 X_test, np_corr_rel_matrix)
        X_test_y = X_test[:, step_size - 1, :, :, :]

        # Load failure examples of the training data (failure examples excluded from the training) for further evaluation?

        print("Validation split: ", X_valid.shape, "Test data:", X_test.shape)

        # Reconstruct loaded input data
        print("Data for evaluation loaded  ... start with reconstruction ...")
        if loss_use_batch_sim_siam or use_memory_restriction:
            X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)[0]
        else:
            X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)
        print("Reconstruction of validation data set done with shape :", X_valid_recon.shape) #((9, 61, 61, 3))
        if loss_use_batch_sim_siam or use_memory_restriction:
            X_test_recon = model_MSCRED.predict(X_test, batch_size=128)[0]
        else:
            X_test_recon = model_MSCRED.predict(X_test, batch_size=128)
        print("Reconstruction of test data set done with shape :", X_test_recon.shape)  # ((9, 61, 61, 3))

        # Generate deep encodings
        # Dummy model for obtaining access to latent encodings / space
        if generate_deep_encodings:
            layer_name = 'Reshape_ToOrignal_ConvLSTM_4'
            intermediate_layer_model = tf.keras.Model(inputs=model_MSCRED.input,
                                                   outputs=model_MSCRED.get_layer(layer_name).output)
            encoded_output = intermediate_layer_model.predict(X_test, batch_size=128)
            print("Encoded_output shape: ", encoded_output.shape)
            np.save('encoded_test.npy', encoded_output)

        feature_names = np.load(feature_names_path)

        # Remove any dimension with size of 1
        X_valid_y = np.squeeze(X_valid_y)
        X_test_y = np.squeeze(X_test_y)

        if use_mass_evaulation:

            test_configurations = list(zip(threshold_selection_criterium_list, num_of_dim_over_threshold_list,
                                           num_of_dim_under_threshold_list, use_corr_rel_matrix_in_eval_list, use_attribute_anomaly_as_condition_list))
            for i in range(len(test_configurations)):
                threshold_selection_criterium, num_of_dim_over_threshold, num_of_dim_under_threshold, use_corr_rel_matrix_in_eval,\
                use_attribute_anomaly_as_condition = test_configurations[i][0], test_configurations[i][1], test_configurations[i][2], test_configurations[i][3], test_configurations[i][4]
                print("threshold_selection_criterium: ", threshold_selection_criterium, " with: ",num_of_dim_over_threshold,"/",num_of_dim_under_threshold, ". CorrMatrix: ", use_corr_rel_matrix_in_eval,"Anomaly based on Attributes: ", use_attribute_anomaly_as_condition)

                ### Calcuation of reconstruction error on the validation data set ###
                recon_err_matrixes_valid, recon_err_perAttrib_valid, mse_per_example_valid, mse_per_example_per_dims = calculateReconstructionError(
                    real_input=X_valid_y, reconstructed_input=X_valid_recon, plot_heatmaps=plot_heatmap_of_rec_error,
                    use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix)

                ### Calcuation of reconstruction error on the test data set ###
                recon_err_matrixes_test, recon_err_perAttrib_test, mse_per_example_test, mse_per_example_per_dims = calculateReconstructionError(
                    real_input=X_test_y, reconstructed_input=X_test_recon, plot_heatmaps=plot_heatmap_of_rec_error,
                    use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix)

                # Define Thresholds for each dimension and attribute
                thresholds, mse_threshold = calculateThreshold(reconstructed_input=X_valid_recon,recon_err_perAttrib_valid=recon_err_perAttrib_valid,
                                                                       threshold_selection_criterium=threshold_selection_criterium, mse_per_example=mse_per_example_valid,
                                                               print_pandas_statistics_for_validation=print_pandas_statistics_for_validation, feature_names=feature_names)

                # Evaluate
                eval_results, eval_results_over_all_dimensions, eval_results_over_all_dimensions_for_each_example = calculateAnomalies(reconstructed_input=X_valid_recon, recon_err_perAttrib=recon_err_perAttrib_valid, thresholds=thresholds, print_att_dim_statistics = print_att_dim_statistics, use_dim_for_anomaly_detection=use_dim_for_anomaly_detection)
                eval_results_f, eval_results_over_all_dimensions_f, eval_results_over_all_dimensions_for_each_example_f = calculateAnomalies(reconstructed_input=X_test_recon, recon_err_perAttrib=recon_err_perAttrib_test, thresholds=thresholds, print_att_dim_statistics = print_att_dim_statistics, use_dim_for_anomaly_detection=use_dim_for_anomaly_detection)

                # Get Positions of anomalies


                print("#### Evaluate No-FAILURES / Validation data set #####")
                printEvaluation2(reconstructed_input=X_valid_recon, eval_results_over_all_dimensions=eval_results_over_all_dimensions, feature_names=feature_names,
                                num_of_dim_under_threshold=num_of_dim_under_threshold, num_of_dim_over_threshold=num_of_dim_over_threshold,
                                 mse_threshold=mse_threshold, mse_values=mse_per_example_valid, use_attribute_anomaly_as_condition=use_attribute_anomaly_as_condition,
                                 print_all_examples=print_all_examples)
                print("#### Evaluate No-FAILURES and FAILURES / Test data set #####")
                printEvaluation2(reconstructed_input=X_test_recon, eval_results_over_all_dimensions=eval_results_over_all_dimensions_f, feature_names=feature_names, labels = test_failure_labels_y,
                                num_of_dim_under_threshold=num_of_dim_under_threshold, num_of_dim_over_threshold=num_of_dim_over_threshold,
                                 mse_threshold=mse_threshold, mse_values=mse_per_example_test, use_attribute_anomaly_as_condition=use_attribute_anomaly_as_condition,
                                 print_all_examples=print_all_examples)
        else:
            ### Calcuation of reconstruction error on the validation data set ###
            recon_err_matrixes_valid, recon_err_perAttrib_valid, mse_per_example_valid, mse_per_example_per_dims = calculateReconstructionError(
                real_input=X_valid_y, reconstructed_input=X_valid_recon, plot_heatmaps=plot_heatmap_of_rec_error,
                use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix)

            ### Calcuation of reconstruction error on the test data set ###
            recon_err_matrixes_test, recon_err_perAttrib_test, mse_per_example_test, mse_per_example_per_dims = calculateReconstructionError(
                real_input=X_test_y, reconstructed_input=X_test_recon, plot_heatmaps=plot_heatmap_of_rec_error,
                use_corr_rel_matrix=use_corr_rel_matrix_in_eval, corr_rel_matrix=np_corr_rel_matrix)

            # Define Thresholds for each dimension and attribute
            thresholds, mse_threshold = calculateThreshold(reconstructed_input=X_valid_recon,
                                                           recon_err_perAttrib_valid=recon_err_perAttrib_valid,
                                                           threshold_selection_criterium=threshold_selection_criterium,
                                                           mse_per_example=mse_per_example_valid,
                                                           print_pandas_statistics_for_validation=print_pandas_statistics_for_validation,
                                                           feature_names=feature_names)

            # Evaluate
            eval_results, eval_results_over_all_dimensions, eval_results_over_all_dimensions_for_each_example = calculateAnomalies(
                reconstructed_input=X_valid_recon, recon_err_perAttrib=recon_err_perAttrib_valid, thresholds=thresholds,
                print_att_dim_statistics=print_att_dim_statistics, use_dim_for_anomaly_detection=use_dim_for_anomaly_detection)
            eval_results_f, eval_results_over_all_dimensions_f, eval_results_over_all_dimensions_for_each_example_f = calculateAnomalies(
                reconstructed_input=X_test_recon, recon_err_perAttrib=recon_err_perAttrib_test, thresholds=thresholds,
                print_att_dim_statistics=print_att_dim_statistics, use_dim_for_anomaly_detection=use_dim_for_anomaly_detection)

            # Get Positions of anomalies

            print("#### Evaluate No-FAILURES / Validation data set #####")
            printEvaluation2(reconstructed_input=X_valid_recon,
                             eval_results_over_all_dimensions=eval_results_over_all_dimensions,
                             feature_names=feature_names,
                             num_of_dim_under_threshold=num_of_dim_under_threshold,
                             num_of_dim_over_threshold=num_of_dim_over_threshold,
                             mse_threshold=mse_threshold, mse_values=mse_per_example_valid,
                             use_attribute_anomaly_as_condition=use_attribute_anomaly_as_condition,
                             print_all_examples=print_all_examples)
            print("#### Evaluate No-FAILURES and FAILURES / Test data set #####")
            printEvaluation2(reconstructed_input=X_test_recon,
                             eval_results_over_all_dimensions=eval_results_over_all_dimensions_f,
                             feature_names=feature_names, labels=test_failure_labels_y,
                             num_of_dim_under_threshold=num_of_dim_under_threshold,
                             num_of_dim_over_threshold=num_of_dim_over_threshold,
                             mse_threshold=mse_threshold, mse_values=mse_per_example_test,
                             use_attribute_anomaly_as_condition=use_attribute_anomaly_as_condition,
                             print_all_examples=print_all_examples)


if __name__ == '__main__':
    main()
