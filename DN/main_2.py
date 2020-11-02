# from _ctypes import sizeof
# from ctypes import memset

import numpy as np
from mscred import MSCRED
from mscred import MSCRED_woLSTM_woAttention
from mscred import MSCRED_woAttention
from mscred import MSCRED_with_LatentOutput
from mscred import MSCRED_w_Noise
from mscred import MSCRED_with_Memory
from mscred import MSCRED_with_Memory2
from mscred import MSCRED_with_Memory2_Auto
from mscred import MSCRED_with_Memory2_Auto_InstanceBased
from mscred import Memory
from mscred import MemoryInstanceBased
import os
import sys
import tensorflow.compat.v1 as tf
import pandas as pd
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt


tf.disable_v2_behavior()
# import logging

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

# run able server
sys.path.append(os.path.abspath("."))

from configuration.Configuration import Configuration

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = Configuration()
def calculateThreshold(reconstructed_input, recon_err_perAttrib_valid, threshold_selection_criterium='99%', mse_per_example= None):
    thresholds = np.zeros((reconstructed_input.shape[3], reconstructed_input.shape[1] + reconstructed_input.shape[2]))
    mse_threshold = None
    for i_dim in range(reconstructed_input.shape[3]):
        data_curr_dim = recon_err_perAttrib_valid[:, :, i_dim]
        # print("data_curr_dim shape: ", data_curr_dim.shape)
        data_curr_dim = np.squeeze(data_curr_dim)
        # print("data_curr_dim shape: ", data_curr_dim.shape)
        df_curr_dim = pd.DataFrame(data_curr_dim)
        # print(df_curr_dim.head())
        df_curr_dim_described = df_curr_dim.describe(percentiles=[.25, .5, .75, 0.9, 0.95, 0.97, 0.99])
        print("Dim: ", i_dim, "Healthy: ")
        print(df_curr_dim_described)
        # get max value for each sensor
        # max = df_curr_dim_described.loc['max', :]
        # print("max: ", max)
        thresholds[i_dim, :] = df_curr_dim_described.loc[threshold_selection_criterium, :].values  # 97%
    # MSE threshold
    if mse_per_example is not None:
        df_mse = pd.DataFrame(mse_per_example)
        df_mse_described = df_mse.describe(percentiles=[.25, .5, .75, 0.9, 0.95, 0.97, 0.99])
        print(df_mse_described)
        mse_threshold = df_mse_described.loc[threshold_selection_criterium].values
        print("mse_threshold: ", mse_threshold)

    return thresholds, mse_threshold

def calculateReconstructionError(real_input, reconstructed_input,plot_heatmaps,use_corr_rel_matrix_for_loss=False, corr_rel_matrix=None):
    reconstruction_error_matrixes = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1], reconstructed_input.shape[2], reconstructed_input.shape[3]))
    reconstruction_errors_perAttribute = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1] + reconstructed_input.shape[2], reconstructed_input.shape[3]))
    mse_per_example_over_all_dims = np.zeros((reconstructed_input.shape[0]))

    for i_example in range(reconstructed_input.shape[0]):  # Iterate over all examples
        for i_dim in range(reconstructed_input.shape[3]):  # Iterate over all "time-dimensions"
            # Get reconstructed and real input data
            curr_matrix_input_recon = reconstructed_input[i_example, :, :, i_dim]
            curr_matrix_input_real = real_input[i_example, :, :, i_dim]
            # Calculate the reconstruction error
            diff = curr_matrix_input_recon - curr_matrix_input_real
            if use_corr_rel_matrix_for_loss:
                diff = diff * corr_rel_matrix
            # print("curr_matrix_input_recon shape: ", curr_matrix_input_recon.shape)
            reconstruction_error_matrixes[i_example, :, :, i_dim] = np.square(diff)
            '''
            plot_heatmap_of_reconstruction_error(id=i_example, dim=i_dim, input=curr_matrix_input_real,
                                                 output=curr_matrix_input_recon,
                                                 rec_error_matrix=reconstruction_error_matrixes[i_example,:,:,i_dim])
            '''
            # Alternative variants for calculation of reconstruction error
            mse = np.mean(np.square(curr_matrix_input_real - curr_matrix_input_recon))
            diff_paper_formula = np.square(np.linalg.norm(diff, ord='fro'))
            # diff_paper_formula_axis0 = np.square(np.linalg.norm(diff, ord='fro', axis=0))
            # diff_paper_formula_axis1 = np.square(np.linalg.norm(diff, ord='fro', axis=1))
            mse_axis0 = np.mean(np.square(curr_matrix_input_real - curr_matrix_input_recon), axis=0)
            mse_axis1 = np.mean(np.square(curr_matrix_input_real - curr_matrix_input_recon), axis=1)
            reconstruction_errors_perAttribute[i_example, :reconstructed_input.shape[1], i_dim] = mse_axis0
            reconstruction_errors_perAttribute[i_example, reconstructed_input.shape[1]:, i_dim] = mse_axis1
            mse_per_example_over_all_dims[i_example] = mse_per_example_over_all_dims[i_example] + mse
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

    return reconstruction_error_matrixes, reconstruction_errors_perAttribute, mse_per_example_over_all_dims

def calculateAnomalies(reconstructed_input, recon_err_perAttrib_valid, thresholds):
    eval_results = np.zeros(
        (reconstructed_input.shape[0], reconstructed_input.shape[1] + reconstructed_input.shape[2], reconstructed_input.shape[3]))
    for i_example in range(recon_err_perAttrib_valid.shape[0]):
        rec_error_curr_example = recon_err_perAttrib_valid[i_example, :, :]
        for i_dim in range(reconstructed_input.shape[3]):
            # Compare reconstruction error if it exceeds threshold to detect an anomaly
            eval = rec_error_curr_example[:, i_dim] > thresholds[i_dim, :]
            eval_results[i_example, :, i_dim] = eval

    #print("eval_results shape: ", eval_results.shape)
    eval_results_over_all_dimensions = np.sum(eval_results, axis=2)
    eval_results_over_all_dimensions_for_each_example = np.sum(eval_results, axis=1)
    #print("eval_results_over_all_dimensions shape: ", eval_results_over_all_dimensions.shape)
    #print("eval_results_over_all_dimensions shape: ", eval_results_over_all_dimensions_for_each_example.shape)
    for i_dim in range(reconstructed_input.shape[3]):
        #print("Current Dimension: ", i_dim)
        for i in range(10):
            print("examples with ", i, " anomalies: ",(eval_results_over_all_dimensions_for_each_example[:, i_dim] == i).sum())
        # num_ones = (y == 1).sum()

    return eval_results, eval_results_over_all_dimensions, eval_results_over_all_dimensions_for_each_example

def printEvaluation(reconstructed_input, eval_results_over_all_dimensions,feature_names,num_of_dim_under_threshold=0, num_of_dim_over_threshold=1000):
    for example_idx in range(reconstructed_input.shape[0]):
        idx_with_Anomaly_1 = np.where(np.logical_and(
            num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]],
            eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]] > num_of_dim_over_threshold))
        count_dim_anomalies_1 = eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][
            idx_with_Anomaly_1]
        # print("eval_results_over_all_dimensions_f: ", eval_results_over_all_dimensions[example_idx,:pred.shape[1]])
        # idx_with_Anomaly_2 = np.where(eval_results_over_all_dimensions[example_idx,pred.shape[1]:] > num_of_dim_over_threshold)
        idx_with_Anomaly_2 = np.where(np.logical_and(
            num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:],
            eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:] > num_of_dim_over_threshold))
        count_dim_anomalies_2 = eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][
            idx_with_Anomaly_2]
        print("NoFailure: ", " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1], " with counts: ",
              count_dim_anomalies_1)
        print("NoFailure: ", "idx_with_Anomaly_2: ", feature_names[idx_with_Anomaly_2], " with counts: ",
              count_dim_anomalies_2)

def printEvaluation2(reconstructed_input, eval_results_over_all_dimensions, feature_names, labels=None,
                     num_of_dim_under_threshold=0, num_of_dim_over_threshold=1000, mse_threshold=None, mse_values=None):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for example_idx in range(reconstructed_input.shape[0]):
        # idx_with_Anomaly_1 = np.where(eval_results_over_all_dimensions_f[example_idx,:pred.shape[1]] > num_of_dim_over_threshold)
        idx_with_Anomaly_1 = np.where(np.logical_and(
            num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]],
            eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]] > num_of_dim_over_threshold))
        count_dim_anomalies_1 = eval_results_over_all_dimensions[example_idx, :reconstructed_input.shape[1]][
            idx_with_Anomaly_1]
        # print("eval_results_over_all_dimensions_f: ", eval_results_over_all_dimensions_f[example_idx,:pred.shape[1]])
        # idx_with_Anomaly_2 = np.where(eval_results_over_all_dimensions_f[example_idx,pred.shape[1]:] > num_of_dim_over_threshold)
        idx_with_Anomaly_2 = np.where(np.logical_and(
            num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:],
            eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:] > num_of_dim_over_threshold))
        count_dim_anomalies_2 = eval_results_over_all_dimensions[example_idx, reconstructed_input.shape[1]:][
            idx_with_Anomaly_2]
        # print(train_failure_labels_y[example_idx]," idx_with_Anomaly_1:  ", idx_with_Anomaly_1)

        if labels is not None:
            print(labels[example_idx],": ", mse_values[example_idx], " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1],
                  " with counts: ", count_dim_anomalies_1,"idx_with_Anomaly_2: ", feature_names[idx_with_Anomaly_2],
                  " with counts: ", count_dim_anomalies_2)
            #Criterium for defining an anomaly
            if ( mse_values[example_idx] > mse_threshold): # feature_names[idx_with_Anomaly_1].shape[0] + feature_names[idx_with_Anomaly_2].shape[0]) > 0:
                # Anomaly detected
                if labels[example_idx] == 'no_failure':
                    FN = FN + 1
                else:
                    TN = TN + 1
            else:
                # No Anomaly detected
                if labels[example_idx] == 'no_failure':
                    TP = TP + 1
                else:
                    FP = FP + 1
        else:
            print("NoFailure: ", " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1], " with counts: ",
                  count_dim_anomalies_1, feature_names[idx_with_Anomaly_2], " with counts: ",
                  count_dim_anomalies_2)

    if labels is not None:
        print("Results: ")
        print("No_Failure TP: ", TP)
        print("No_Failure TN: ", TN)
        print("No_Failure FP: ", FP)
        print("No_Failure FN: ", FN)
        prec = TP/(TP+FP)
        rec = TP/(TP+FN)
        acc = (TP+TN)/(TP+TN+FP+FN)
        f1 = 2*((prec*rec)/(prec+rec))
        print("No_Failure Acc: ", acc)
        print("No_Failure Precision: ", prec)
        print("No_Failure Recall: ", rec)
        print("No_Failure F1: ", f1)


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

def corr_rel_matrix_weighted_loss(corr_rel_mat):
    def loss(y_true, y_pred):
        # MSE
        loss = tf.square(y_true - y_pred)
        print("loss dim: ", loss.shape)
        #if use_corr_rel_matrix:
        #loss = loss * corr_rel_mat
        loss = tf.reduce_sum(loss, axis=-1)
        loss = loss / np.sum(corr_rel_mat) # normalize by number of considered correlations
        #loss = tf.reduce_mean(loss, axis=-1)
        return loss
    return loss  # Note the `axis=-1`

def my_loss_fn(y_true, y_pred):
    # MSCRED Implementation
    # squared_difference = tf.square(y_true - y_pred)
    loss = np.square(np.linalg.norm((y_true - y_pred), ord='fro'))
    print("loss dim: ", loss.shape)
    return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`

def my_loss_fn2_0(y_true, y_pred):
    # MSCRED Implementation
    print("y_pred dim_0: ", y_pred.shape)
    loss = tf.square(y_true - y_pred)
    #loss = np.square(np.linalg.norm((y_true - y_pred[0]), ord='fro'))
    print("loss dim: ", loss.shape)
    loss = tf.reduce_mean(loss, axis=-1)
    print("my_loss_fn2_0 loss dim: ", loss.shape)
    return loss  # Note the `axis=-1`
def my_loss_fn2_1(y_true, y_pred):
    # MSCRED Implementation
    #loss = tf.square(y_true - y_pred)
    #loss = np.square(np.linalg.norm((y_true - y_pred[0]), ord='fro'))
    print("y_pred dim_1: ", y_pred.shape)
    print("y_true dim_1: ", y_true.shape)
    y_pred = tf.squeeze(y_pred)
    print("y_pred dim_1: ", y_pred.shape)
    #cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

    #loss = cosine_loss(y_true, y_pred)
    '''
        indices_a = tf.range(config.batch_size)
        indices_a = tf.tile(indices_a, [config.batch_size])
        y_pred = tf.gather(y_pred, indices_a)
        # a shape: [T*T, C]
        print("y_pred dim_tiled: ", y_pred.shape)
        indices_b = tf.range(config.batch_size)
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, config.batch_size])
        indices_b = tf.reshape(indices_b, [-1])
        y_true = tf.gather(y_true, indices_b)
        loss = cosine_loss(y_true, y_pred)
     '''
    #Cosine Sim:#
    x = tf.nn.l2_normalize(y_pred, axis=0)
    y = tf.nn.l2_normalize(y_pred, axis=0)
    loss = tf.matmul(x, y, transpose_b=True)
    print("x loss dim: ", loss.shape)
    loss = tf.math.abs(loss)

    # Eucl. Sim:#

    s = 2 * tf.matmul(y_pred, y_pred, transpose_b=True)
    diag_x = tf.reduce_sum(y_pred * y_pred, axis=-1, keepdims=True)
    diag_y = tf.reshape(tf.reduce_sum(y_pred * y_pred, axis=-1), (1, -1))
    loss = s - diag_x - diag_y

    #loss = tf.reduce_mean(loss, axis=-1)
    #loss = tf.reduce_sum(loss, axis=-1)
    print("x loss dim: ", loss.shape)
    #loss = 128 - loss
    loss = loss/128/(128*128)
    #maximize:
    #loss = -loss
    '''
    print("y loss dim: ", loss.shape, "loss: ", loss)
    loss = tf.reshape(loss, [128, 1])
    loss = tf.tile(loss, [1,61*61])
    print("y loss dim: ", loss.shape, "loss: ", loss)
    loss = tf.reshape(loss, [128,61,61])
    print("y loss dim: ", loss.shape, "loss: ", loss)
    #loss = tf.reduce_mean(loss, axis=0)
    #print("y loss dim: ", loss.shape, "loss: ", loss)
    '''
    return loss

    #return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`
def my_loss_fn2_MemEntropy(y_true, y_pred):
    mem_etrp = tf.reduce_sum((-y_pred) * tf.math.log(y_pred + 1e-12))
    print("mem_etrp shape: ", mem_etrp.shape)
    loss = tf.reduce_mean(mem_etrp)
    return loss
def main():
    use_data_set_version = 3
    train_model = False
    use_mscred_wo_LSTM_wo_Attention = False     #Entweder das oder das nächste, sonst normales mscred
    use_mscred_wo_Attention = False
    use_mscred_w_Noise = False                   # denoising autoencoder
    use_mscred_memory = True
    use_mscred_memory2 = False
    test_model = True
    loss_use_corr_rel_matrix = False # Reconstruction error is only based on relevant correlations
    loss_use_batch_sim_siam = False
    use_corr_rel_matrix_for_input = False # input contains only relevant correlations, others set to zero
    use_corr_rel_matrix_for_input_replace_by_epsilon = False # meaningful correlation that would be zero, are now near zero
    plot_heatmap_of_rec_error = False
    curr_run_identifier = "data_set_3_use_mscred_memory2" #mat_data1_standardModell_eucl, mat_data1_standardModell_CorrMatLoss
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    step_size = config.step_max
    early_stopping_patience = 5
    split_train_test_ratio=0.1
    # Test Parameter:
    threshold_selection_criterium = '99%' # 'max', '99%'
    num_of_dim_over_threshold = 3 # normal: 0
    num_of_dim_under_threshold = 20 # normal: 10 (higher as max. dim value) # 3: 20
    #TODO: change data path self.training_data_folder = "../../../../data/pklein/PredMSiamNN/data/training_data/" #Im homeVerzeichnis: '../../PredMSiamNN2/data/training_data/'
    path= "../data/"
    if use_data_set_version == 1:
        training_data_set_path = "../data/training_data_set.npy"
        valid_split_save_path = "../data/training_data_set_test_split.npy"
        test_matrix_path = "../data/training_data_set_failure.npy"
        test_labels_y_path = "../data/training_data_set_failure_labels_test.npy"
        test_matrix_path = "../data/test_data_set.npy"
        test_labels_y_path = "../data/test_data_set_failure_labels.npy"
    elif use_data_set_version == 2:
        training_data_set_path = "../data/training_data_set_2_trainWoFailure.npy"
        valid_split_save_path = "../data/training_data_set_2_test_split.npy"
        test_matrix_path = "../data/training_data_set_2_trainWFailure.npy"
        test_labels_y_path = "../data/training_data_set_2_failure_labels.npy"
        test_matrix_path = "../data/test_data_set_2.npy"
        test_labels_y_path = "../data/test_data_set_2_failure_labels.npy"
    elif use_data_set_version == 3:
        training_data_set_path = "../data/training_data_set_3_trainWoFailure.npy"
        valid_split_save_path = "../data/training_data_set_3_test_split.npy"
        test_matrix_path = "../data/training_data_set_3_trainWFailure.npy"
        test_labels_y_path = "../data/training_data_set_3_failure_labels.npy"
        test_matrix_path = "../data/test_data_set_3.npy"
        test_labels_y_path = "../data/test_data_set_3_failure_labels.npy"


    test_matrix_path = test_matrix_path
    test_labels_y_path = test_labels_y_path
    feature_names_path = "../data/feature_names.npy"
    ### Create MSCRED model as TF graph
    print("config.relevant_features: ", config.relevant_features)
    train_failure_labels_y = np.load(test_labels_y_path)
    ### Load Correlation Relevance Matrix ###
    df_corr_rel_matrix = pd.read_csv('../data/Attribute_Correlation_Relevance_Matrix_v0.csv', sep=';',index_col=0)
    np_corr_rel_matrix = df_corr_rel_matrix.values
    print("np_corr_rel_matrix: ", np_corr_rel_matrix.shape)
    print("Only ", np.sum(np_corr_rel_matrix)," of ",np_corr_rel_matrix.shape[0] * np_corr_rel_matrix.shape[1]," correlations")

    print('-------------------------------')
    print('Creation of the model')
    # create graph structure of the NN
    if use_mscred_wo_LSTM_wo_Attention:
        model_MSCRED = MSCRED_woLSTM_woAttention().create_model()
    elif use_mscred_wo_Attention:
        model_MSCRED = MSCRED_woAttention().create_model()
    elif loss_use_batch_sim_siam:
        model_MSCRED = MSCRED_with_LatentOutput().create_model()
    elif use_mscred_w_Noise:
        model_MSCRED = MSCRED_w_Noise().create_model()
    elif use_mscred_memory:
        model_MSCRED = MSCRED_with_Memory2_Auto().create_model()
    elif use_mscred_memory2:
        model_MSCRED = MSCRED_with_Memory2_Auto_InstanceBased().create_model()
    else:
        model_MSCRED = MSCRED().create_model()
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
        if loss_use_batch_sim_siam:
            if use_data_set_version == 1:
                curr_xTrainData = curr_xTrainData[:8960,:,:,:,:] # (9918, 5, 61, 61, 17)
            elif use_data_set_version == 2:
                curr_xTrainData = curr_xTrainData[:48640,:,:,:,:] # (49469, 5, 61, 61, 8)
            print("Size adjusted (failure-free training) data: ", curr_xTrainData.shape)

        #Remove irrelevant correlations
        if use_corr_rel_matrix_for_input:
            if use_corr_rel_matrix_for_input_replace_by_epsilon:
                print("Start replacing 0 values by epsilon  from the input")
                curr_xTrainData[curr_xTrainData == 0] = tf.keras.backend.epsilon()
                print("Finished replacing 0 values by epsilon  from the input")
            print("Start removing irrelevant correlations from the input")
            np_corr_rel_matrix_reshaped = np.reshape(np_corr_rel_matrix, (1,1,61,61,1))
            curr_xTrainData = curr_xTrainData * np_corr_rel_matrix_reshaped
            print("Finished removing irrelevant correlations from the input")
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
        #history = model_MSCRED.compile(optimizer=opt, loss=[my_loss_fn2_0, my_loss_fn2_1], loss_weights=[1.0,0.1]) #tf.keras.losses.mse
        #history = model_MSCRED.compile(optimizer=opt, loss=tf.keras.losses.mse)  #
        if loss_use_corr_rel_matrix:
            history = model_MSCRED.compile(optimizer=opt, loss=corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix))
            model_MSCRED.fit(X_train, X_train_y, epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_split=0.2, callbacks=[es, mc])
        elif loss_use_batch_sim_siam:
            history = model_MSCRED.compile(optimizer=opt, loss=[my_loss_fn2_0, my_loss_fn2_1],
                                           loss_weights=[0.9, 0.1])  # tf.keras.losses.mse
            model_MSCRED.fit(X_train, [X_train_y, X_train_y], epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_split=(6/63), callbacks=[es, mc])  # validation_data=(X_valid, X_valid_y)
        elif use_mscred_memory:
            history = model_MSCRED.compile(optimizer=opt, loss=[my_loss_fn2_0, my_loss_fn2_MemEntropy],
                                           loss_weights=[0.9998, 0.0002])  # tf.keras.losses.mse
            model_MSCRED.fit(X_train, [X_train_y, X_train_y], epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_split=(6/63), callbacks=[es, mc])  # validation_data=(X_valid, X_valid_y)
        elif use_mscred_memory2:
            history = model_MSCRED.compile(optimizer=opt, loss=tf.keras.losses.mse, )  #
            model_MSCRED.fit(X_train, X_train_y, epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_split=(6/63), callbacks=[es, mc])  # validation_data=(X_valid, X_valid_y)
        else:
            history = model_MSCRED.compile(optimizer=opt, loss=tf.keras.losses.mse)
            model_MSCRED.fit(X_train,X_train_y,epochs=epochs,batch_size=batch_size, shuffle=True, validation_split=0.1, callbacks=[es, mc]) # validation_data=(X_valid, X_valid_y)

    ### Test ###
    if test_model:

        #Load previous trained model
        if loss_use_batch_sim_siam:
            model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', compile=False)
        elif use_mscred_memory:
            model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', custom_objects={'Memory': Memory}, compile=False)
            #model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', custom_objects={'Memory': Memory, 'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix)})
        elif use_mscred_memory2:
            model_MSCRED = tf.keras.models.load_model('best_model_' + curr_run_identifier + '.h5',
                                                      custom_objects={'MemoryInstanceBased': MemoryInstanceBased}, compile=False)
        else:
            model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', custom_objects={'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix)})

        print("Pretrained Model loaded ...")

        # Load validation split of the training data (used for defining the anomaly thresholds)
        X_valid = np.load(valid_split_save_path)
        if use_corr_rel_matrix_for_input:
            if use_corr_rel_matrix_for_input_replace_by_epsilon:
                print("Start replacing 0 values by epsilon  from the input")
                X_valid[X_valid == 0] = tf.keras.backend.epsilon()
                print("Finished replacing 0 values by epsilon  from the input")
            print("Start removing irrelevant correlations from the input")
            np_corr_rel_matrix_reshaped = np.reshape(np_corr_rel_matrix, (1,1,61,61,1))
            X_valid = X_valid * np_corr_rel_matrix_reshaped
            print("Finished removing irrelevant correlations from the input")
        X_valid_y = X_valid[:, step_size - 1, :, :, :]

        # Load test data (as used in PredM Siamese NN)
        X_test = np.load(test_matrix_path)
        if use_corr_rel_matrix_for_input:
            if use_corr_rel_matrix_for_input_replace_by_epsilon:
                print("Start replacing 0 values by epsilon from the input")
                X_test[X_test == 0] = tf.keras.backend.epsilon()
                print("Finished replacing 0 values by epsilon from the input")
            print("Start removing irrelevant correlations from the input")
            np_corr_rel_matrix_reshaped = np.reshape(np_corr_rel_matrix, (1,1,61,61,1))
            X_test = X_test * np_corr_rel_matrix_reshaped
            print("Finished removing irrelevant correlations from the input")
        X_test_y = X_test[:, step_size - 1, :, :, :]

        # Load failure examples of the training data (failure examples excluded from the training) for further evaluation?

        print("Validation split: ", X_valid.shape, "Test data:", X_test.shape)

        # Reconstruct loaded input data
        print("Data for evaluation loaded  ... start with reconstruction ...")
        if loss_use_batch_sim_siam or use_mscred_memory:
            X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)[0]
        else:
            X_valid_recon = model_MSCRED.predict(X_valid, batch_size=128)  # [0]
        print("Reconstruction of validation data set done with shape :", X_valid_recon.shape) #((9, 61, 61, 3))
        if loss_use_batch_sim_siam:
            if use_data_set_version == 1:
                X_test = X_test[:6528, :, :, :, :]#[:6528, :, :, :, :]
            elif use_data_set_version == 2:
                X_test = X_test[:1408, :, :, :, :]
            X_test_recon = model_MSCRED.predict(X_test, batch_size=128)[0]
        elif use_mscred_memory:
            X_test_recon = model_MSCRED.predict(X_test, batch_size=128)[0]
        else:
            X_test_recon = model_MSCRED.predict(X_test, batch_size=128)  # [0]
        print("Reconstruction of test data set done with shape :", X_test_recon.shape)  # ((9, 61, 61, 3))

        # Remove any dimension with size of 1
        X_valid_y = np.squeeze(X_valid_y)
        X_test_y = np.squeeze(X_test_y)

        ### Calcuation of reconstruction error on the validation data set ###
        recon_err_matrixes_valid, recon_err_perAttrib_valid, mse_per_example_valid = calculateReconstructionError(
            real_input=X_valid_y, reconstructed_input=X_valid_recon, plot_heatmaps=False,
            use_corr_rel_matrix_for_loss=False, corr_rel_matrix=None)

        ### Calcuation of reconstruction error on the test data set ###
        recon_err_matrixes_test, recon_err_perAttrib_test, mse_per_example_test = calculateReconstructionError(
            real_input=X_test_y, reconstructed_input=X_test_recon, plot_heatmaps=False,
            use_corr_rel_matrix_for_loss=False, corr_rel_matrix=None)

        # Define Thresholds for each dimension and attribute
        thresholds, mse_threshold = calculateThreshold(reconstructed_input=X_valid_recon,recon_err_perAttrib_valid=recon_err_perAttrib_valid,
                                                               threshold_selection_criterium=threshold_selection_criterium, mse_per_example=mse_per_example_valid)

        # Evaluate
        eval_results, eval_results_over_all_dimensions, eval_results_over_all_dimensions_for_each_example = calculateAnomalies(reconstructed_input=X_valid_recon, recon_err_perAttrib_valid=recon_err_perAttrib_valid, thresholds=thresholds)
        eval_results_f, eval_results_over_all_dimensions_f, eval_results_over_all_dimensions_for_each_example_f = calculateAnomalies(reconstructed_input=X_test_recon, recon_err_perAttrib_valid=recon_err_perAttrib_test, thresholds=thresholds)

        # Get Positions of anomalies
        feature_names = np.load(feature_names_path)
        print("feature_names: ", feature_names)
        print("#### No-FAILURES #####")
        printEvaluation2(reconstructed_input=X_valid_recon, eval_results_over_all_dimensions=eval_results_over_all_dimensions, feature_names=feature_names,
                        num_of_dim_under_threshold=num_of_dim_under_threshold, num_of_dim_over_threshold=num_of_dim_over_threshold,
                         mse_threshold=mse_threshold, mse_values=mse_per_example_valid)
        print("#### FAILURES #####")
        printEvaluation2(reconstructed_input=X_test_recon, eval_results_over_all_dimensions=eval_results_over_all_dimensions_f, feature_names=feature_names, labels = train_failure_labels_y,
                        num_of_dim_under_threshold=num_of_dim_under_threshold, num_of_dim_over_threshold=num_of_dim_over_threshold,
                         mse_threshold=mse_threshold, mse_values=mse_per_example_test)


if __name__ == '__main__':
    main()
