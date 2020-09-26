# from _ctypes import sizeof
# from ctypes import memset

import numpy as np
from mscred import MSCRED
from mscred import MSCRED_woLSTM_woAttention
from mscred import MSCRED_woAttention
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
    return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`
def my_loss_fn2_1(y_true, y_pred):
    # MSCRED Implementation
    #loss = tf.square(y_true - y_pred)
    #loss = np.square(np.linalg.norm((y_true - y_pred[0]), ord='fro'))
    print("y_pred dim_1: ", y_pred.shape)
    y_pred = tf.squeeze(y_pred)
    print("y_pred dim_1: ", y_pred.shape)
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

    loss = cosine_loss(y_true, y_pred)
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
    '''
    x = tf.nn.l2_normalize(y_pred, axis=0)
    y = tf.nn.l2_normalize(y_pred, axis=0)
    loss = tf.matmul(x, y, transpose_b=True)
    print("loss dim: ", loss.shape)
    loss = tf.reduce_mean(loss, axis=-1)
    '''
    print("loss dim: ", loss.shape, "loss: ", loss)
    return loss

    #return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`

def main():
    train_model = False
    use_mscred_wo_LSTM_wo_Attention = False #Entweder das oder das nächste, sonst normales mscred
    use_mscred_wo_Attention = False
    test_model = True
    use_corr_rel_matrix = True # Reconstruction error is only based on relevant correlations
    use_corr_rel_matrix_for_input = True # input contains only relevant correlations, others set to zero
    use_corr_rel_matrix_for_input_replace_by_epsilon = False # meaningful correlation that would be zero, are now near zero
    plot_heatmap_of_rec_error = False
    curr_run_identifier= "mat_woEpsilon" #_sep_0-0001 3:"mat_woEpsilon-" 4:"mat_woEpsilon-woAtt"
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    step_size = config.step_max
    early_stopping_patience = 10
    split_train_test_ratio=0.1
    num_of_dim_over_threshold = 0 # normal: 0
    num_of_dim_under_threshold = 10 # normal: 10 (higher as max. dim value)
    valid_split_save_path = "../data/training_data_set_test_split.npy"
    train_failure_matrix_path = "../data/training_data_set_failure.npy"
    train_failure_labels_y_path = "../data/training_data_set_failure_labels_test.npy"
    test_matrix_path = "../data/test_data_set.npy"
    test_labels_y_path = "../data/test_data_set_failure_labels.npy"
    train_failure_matrix_path = test_matrix_path
    train_failure_labels_y_path = test_labels_y_path
    feature_names_path = "../data/feature_names.npy"
    ### Create MSCRED model as TF graph
    print("config.relevant_features: ", config.relevant_features)
    train_failure_labels_y = np.load(train_failure_labels_y_path)
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
        curr_xTrainData = np.load("../data/training_data_set.npy")
        print("Loaded (failure-free training) data: ", curr_xTrainData.shape)
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
        if use_corr_rel_matrix:
            history = model_MSCRED.compile(optimizer=opt, loss=corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix))
        else:
            history = model_MSCRED.compile(optimizer=opt, loss=tf.keras.losses.mse)
        model_MSCRED.fit(X_train,X_train_y,epochs=epochs,batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[es, mc]) # validation_data=(X_valid, X_valid_y)

    ### Test ###
    if test_model:
        #Load previous trained model
        model_MSCRED = tf.keras.models.load_model('best_model_'+curr_run_identifier+'.h5', custom_objects={'loss': corr_rel_matrix_weighted_loss(corr_rel_mat=np_corr_rel_matrix)})

        #input_pred = np.expand_dims(X_train[0, :, :, :, :], axis=0)
        #input_pred.astype(np.float32)
        #input_pred = tf.cast(input_pred, 'float32')

        # Load validation split of the training data
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
        # Load failure examples of the training data
        X_train_failure = np.load(train_failure_matrix_path)
        if use_corr_rel_matrix_for_input:
            if use_corr_rel_matrix_for_input_replace_by_epsilon:
                print("Start replacing 0 values by epsilon from the input")
                X_train_failure[X_train_failure == 0] = tf.keras.backend.epsilon()
                print("Finished replacing 0 values by epsilon from the input")
            print("Start removing irrelevant correlations from the input")
            np_corr_rel_matrix_reshaped = np.reshape(np_corr_rel_matrix, (1,1,61,61,1))
            X_train_failure = X_train_failure * np_corr_rel_matrix_reshaped
            print("Finished removing irrelevant correlations from the input")
        X_train_failure_y = X_train_failure[:, step_size - 1, :, :, :]
        # Load test data
        X_test = np.load(valid_split_save_path)
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
        print("Test data loaded ... predicting ...")
        print("Test_data no failure: ", X_valid.shape, "test_data failure:", X_train_failure.shape)
        pred = model_MSCRED.predict(X_valid, batch_size=128)#[0]
        print("Predictions (no failure):", pred.shape) #((9, 61, 61, 3))
        pred_failure = model_MSCRED.predict(X_train_failure, batch_size=128)#[0]
        print("Predictions (with failure):", pred_failure.shape)  # ((9, 61, 61, 3))

        X_valid_y = np.squeeze(X_valid_y)
        X_train_failure_y = np.squeeze(X_train_failure_y)

        ### Calcuation of reconstruction error no-failure / healthy for test data###
        reconstruction_error_matrixes = np.zeros((pred.shape[0],pred.shape[1],pred.shape[2],pred.shape[3]))
        reconstruction_errors_perAttribute = np.zeros((pred.shape[0], pred.shape[1]+pred.shape[2], pred.shape[3]))

        for i_example in range(pred.shape[0]): #pred.shape[0] #Iterate over all predictions
            for i_dim in range(pred.shape[3]): #Iterate over all "time-dimisons"
                curr_matrix_pred = pred[i_example,:,:,i_dim]
                curr_matrix_input = X_valid_y[i_example,:,:,i_dim]
                diff = curr_matrix_pred - curr_matrix_input
                if use_corr_rel_matrix:
                    diff = diff * np_corr_rel_matrix
                #print("curr_matrix_pred shape: ", curr_matrix_pred.shape)
                reconstruction_error_matrixes[i_example,:,:,i_dim] = np.square(diff)
                '''
                plot_heatmap_of_reconstruction_error(id=i_example, dim=i_dim, input=curr_matrix_input,
                                                     output=curr_matrix_pred,
                                                     rec_error_matrix=reconstruction_error_matrixes[i_example,:,:,i_dim])
                '''
                mse = np.mean(np.square(curr_matrix_input - curr_matrix_pred))
                diff_paper_formula = np.square(np.linalg.norm(diff, ord='fro'))
                #diff_paper_formula_axis0 = np.square(np.linalg.norm(diff, ord='fro', axis=0))
                #diff_paper_formula_axis1 = np.square(np.linalg.norm(diff, ord='fro', axis=1))
                mse_axis0 = np.mean(np.square(curr_matrix_input - curr_matrix_pred), axis=0)
                mse_axis1 = np.mean(np.square(curr_matrix_input - curr_matrix_pred), axis=1)
                reconstruction_errors_perAttribute[i_example, :pred.shape[1], i_dim] = mse_axis0
                reconstruction_errors_perAttribute[i_example, pred.shape[1]:, i_dim] = mse_axis1
                #print("example: ", i_example, "dim: ", i_dim, "Rec.Err.: ", diff_paper_formula, "MSE: ", mse)
                #print("axis0: ", mse_axis0)
                #print("axis0: ", mse_axis0.shape)
                #print("reconstruction_errors_perAttribute:", reconstruction_errors_perAttribute[i_example, :, i_dim])
            if plot_heatmap_of_rec_error:
                plot_heatmap_of_reconstruction_error2(id=i_example, input=X_valid_y[i_example,:,:,:],
                    output = pred[i_example,:,:,:], rec_error_matrix = reconstruction_error_matrixes[i_example, :, :, :])


        ### Calcuation of reconstruction error failure test data###
        reconstruction_error_matrixes_f = np.zeros((pred_failure.shape[0],pred_failure.shape[1],pred_failure.shape[2],pred_failure.shape[3]))
        reconstruction_errors_perAttribute_f = np.zeros((pred_failure.shape[0], pred_failure.shape[1]+pred_failure.shape[2], pred_failure.shape[3]))

        for i_example in range(pred_failure.shape[0]): #pred.shape[0] #Iterate over all predictions
            for i_dim in range(pred_failure.shape[3]): #Iterate over all "time-dimisons"
                curr_matrix_pred = pred_failure[i_example,:,:,i_dim]
                curr_matrix_input = X_train_failure_y[i_example,:,:,i_dim]
                diff = curr_matrix_pred - curr_matrix_input
                if use_corr_rel_matrix:
                    diff = diff * np_corr_rel_matrix
                #print("curr_matrix_pred shape: ", curr_matrix_pred.shape)
                reconstruction_error_matrixes_f[i_example,:,:,i_dim] = np.square(diff)
                mse = np.mean(np.square(curr_matrix_input - curr_matrix_pred))
                diff_paper_formula = np.square(np.linalg.norm(diff, ord='fro'))
                #diff_paper_formula_axis0 = np.square(np.linalg.norm(diff, ord='fro', axis=0))
                #diff_paper_formula_axis1 = np.square(np.linalg.norm(diff, ord='fro', axis=1))
                mse_axis0 = np.mean(np.square(curr_matrix_input - curr_matrix_pred), axis=0)
                mse_axis1 = np.mean(np.square(curr_matrix_input - curr_matrix_pred), axis=1)
                reconstruction_errors_perAttribute_f[i_example, :pred_failure.shape[1], i_dim] = mse_axis0
                reconstruction_errors_perAttribute_f[i_example, pred_failure.shape[1]:, i_dim] = mse_axis1
                #print("example: ", i_example, "dim: ", i_dim, "Rec.Err.: ", diff_paper_formula, "MSE: ", mse)
                #print("axis0: ", mse_axis0)
                #print("axis0: ", mse_axis0.shape)
                #print("reconstruction_errors_perAttribute:", reconstruction_errors_perAttribute[i_example, :, i_dim])

        # Define Thresholds
        max_values_for_threshold = np.zeros((pred.shape[3], pred.shape[1] + pred.shape[2]))
        max_values_for_threshold_f = np.zeros((pred_failure.shape[3], pred_failure.shape[1]+pred_failure.shape[2]))
        percentile90_values_for_threshold = np.zeros((pred.shape[3], pred.shape[1] + pred.shape[2]))
        percentile90_values_for_threshold_f = np.zeros((pred_failure.shape[3], pred_failure.shape[1] + pred_failure.shape[2]))
        for i_dim in range(pred.shape[3]):
            data_curr_dim = reconstruction_errors_perAttribute[:,:,i_dim]
            data_curr_dim_f = reconstruction_errors_perAttribute_f[:, :, i_dim]
            #print("data_curr_dim shape: ", data_curr_dim.shape)
            data_curr_dim = np.squeeze(data_curr_dim)
            data_curr_dim_f = np.squeeze(data_curr_dim_f)
            #print("data_curr_dim shape: ", data_curr_dim.shape)
            df_curr_dim = pd.DataFrame(data_curr_dim)
            df_curr_dim_f = pd.DataFrame(data_curr_dim_f)
            #print(df_curr_dim.head())
            df_curr_dim_described = df_curr_dim.describe(percentiles=[.25, .5, .75, 0.9, 0.95, 0.97])
            df_curr_dim_f_described = df_curr_dim_f.describe(percentiles=[.25, .5, .75, 0.9, 0.95, 0.97])
            print("Dim: ", i_dim, "Healthy: ")
            print(df_curr_dim_described)
            print("Dim: ", i_dim, "Faulty: ")
            print(df_curr_dim_f_described)
            # get max value for each sensor
            max = df_curr_dim_described.loc['max',:]
            max_f = df_curr_dim_f_described.loc['max', :]
            #print("max: ", max)
            max_values_for_threshold[i_dim,:] = max.values
            max_values_for_threshold_f[i_dim, :] = max_f.values
            percentile90_values_for_threshold[i_dim,:] =  df_curr_dim_described.loc['97%',:].values
            percentile90_values_for_threshold_f[i_dim, :] =  df_curr_dim_f_described.loc['97%',:].values

        # Evaluate
        eval_results = np.zeros((pred.shape[0], pred.shape[1] + pred.shape[2], pred.shape[3]))
        for i_example in range(reconstruction_errors_perAttribute.shape[0]):
            rec_error_curr_example = reconstruction_errors_perAttribute[i_example,:,:]
            for i_dim in range(pred.shape[3]):
                # Compare reconstruction error if threshold to detect an anomaly
                #print("rec_error_curr_example[:, i_dim] shape: ", rec_error_curr_example[:, i_dim].shape)
                #print("percentile90_values_for_threshold[i_dim, :] shape: ", percentile90_values_for_threshold[i_dim, :].shape)
                #print(rec_error_curr_example[:, i_dim] > percentile90_values_for_threshold[i_dim, :])
                #eval = rec_error_curr_example[:,i_dim]> percentile90_values_for_threshold[i_dim,:]
                eval = rec_error_curr_example[:, i_dim] > max_values_for_threshold[i_dim, :]
                eval_results[i_example,:,i_dim] = eval

        print("eval_results shape: ", eval_results.shape)
        eval_results_over_all_dimensions = np.sum(eval_results, axis=2)
        eval_results_over_all_dimensions_for_each_example = np.sum(eval_results, axis=1)
        print("eval_results_over_all_dimensions shape: ", eval_results_over_all_dimensions.shape)
        print("eval_results_over_all_dimensions shape: ", eval_results_over_all_dimensions_for_each_example.shape)
        for i_dim in range(pred.shape[3]):
            print("Current Dimension: ", i_dim)
            for i in range(10):
                    print("examples with ",i," anomalies: ", (eval_results_over_all_dimensions_for_each_example[:,i_dim] == i).sum())
            #num_ones = (y == 1).sum()

        # Evaluate Failures
        eval_results_f = np.zeros((pred_failure.shape[0], pred_failure.shape[1] + pred_failure.shape[2], pred_failure.shape[3]))
        for i_example in range(reconstruction_errors_perAttribute_f.shape[0]):
            rec_error_curr_example_f = reconstruction_errors_perAttribute_f[i_example,:,:]
            for i_dim in range(pred.shape[3]):
                # Compare reconstruction error if threshold to detect an anomaly
                #print("rec_error_curr_example[:, i_dim] shape: ", rec_error_curr_example[:, i_dim].shape)
                #print("percentile90_values_for_threshold[i_dim, :] shape: ", percentile90_values_for_threshold[i_dim, :].shape)
                #print(rec_error_curr_example[:, i_dim] > percentile90_values_for_threshold[i_dim, :])
                eval_f = rec_error_curr_example_f[:,i_dim]> percentile90_values_for_threshold[i_dim,:]
                eval_results_f[i_example,:,i_dim] = eval_f

        print("eval_results_f shape: ", eval_results_f.shape)
        eval_results_over_all_dimensions_f = np.sum(eval_results_f, axis=2)
        eval_results_over_all_dimensions_for_each_example_f = np.sum(eval_results_f, axis=1)
        print("eval_results_over_all_dimensions_f shape: ", eval_results_over_all_dimensions_f.shape)
        print("eval_results_over_all_dimensions_f shape: ", eval_results_over_all_dimensions_for_each_example_f.shape)
        for i_dim in range(pred.shape[3]):
            print("Current Dimension: ", i_dim)
            for i in range(10):
                print("examples with ",i," anomalies: ", (eval_results_over_all_dimensions_for_each_example_f[:,i_dim] == i).sum())
            #num_ones = (y == 1).sum()

        # Get Positions of anomalies
        feature_names = np.load(feature_names_path)
        print("feature_names: ", feature_names)
        print("#### No-FAILURES #####")

        for example_idx in range(pred.shape[0]):
                idx_with_Anomaly_1 = np.where(np.logical_and(num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx,:pred.shape[1]], eval_results_over_all_dimensions[example_idx,:pred.shape[1]] > num_of_dim_over_threshold))
                count_dim_anomalies_1 = eval_results_over_all_dimensions[example_idx, :pred.shape[1]][
                    idx_with_Anomaly_1]
                #print("eval_results_over_all_dimensions_f: ", eval_results_over_all_dimensions[example_idx,:pred.shape[1]])
                #idx_with_Anomaly_2 = np.where(eval_results_over_all_dimensions[example_idx,pred.shape[1]:] > num_of_dim_over_threshold)
                idx_with_Anomaly_2 = np.where(np.logical_and(
                    num_of_dim_under_threshold > eval_results_over_all_dimensions[example_idx, pred.shape[1]:],
                    eval_results_over_all_dimensions[example_idx, pred.shape[1]:] > num_of_dim_over_threshold))
                count_dim_anomalies_2 = eval_results_over_all_dimensions[example_idx, pred.shape[1]:][idx_with_Anomaly_2]
                print("NoFailure: "," idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1], " with counts: ", count_dim_anomalies_1)
                print("NoFailure: ","idx_with_Anomaly_2: ", feature_names[idx_with_Anomaly_2], " with counts: ", count_dim_anomalies_2)

        print("#### FAILURES #####")

        for example_idx in range(pred_failure.shape[0]):
                #idx_with_Anomaly_1 = np.where(eval_results_over_all_dimensions_f[example_idx,:pred.shape[1]] > num_of_dim_over_threshold)
                idx_with_Anomaly_1 = np.where(np.logical_and(
                    num_of_dim_under_threshold > eval_results_over_all_dimensions_f[example_idx, :pred.shape[1]],
                    eval_results_over_all_dimensions_f[example_idx, :pred.shape[1]] > num_of_dim_over_threshold))
                count_dim_anomalies_1 = eval_results_over_all_dimensions_f[example_idx, :pred.shape[1]][
                    idx_with_Anomaly_1]
                #print("eval_results_over_all_dimensions_f: ", eval_results_over_all_dimensions_f[example_idx,:pred.shape[1]])
                #idx_with_Anomaly_2 = np.where(eval_results_over_all_dimensions_f[example_idx,pred.shape[1]:] > num_of_dim_over_threshold)
                idx_with_Anomaly_2 = np.where(np.logical_and(
                    num_of_dim_under_threshold > eval_results_over_all_dimensions_f[example_idx, pred.shape[1]:],
                    eval_results_over_all_dimensions_f[example_idx, pred.shape[1]:] > num_of_dim_over_threshold))
                count_dim_anomalies_2 = eval_results_over_all_dimensions_f[example_idx, pred.shape[1]:][idx_with_Anomaly_2]
                #print(train_failure_labels_y[example_idx]," idx_with_Anomaly_1:  ", idx_with_Anomaly_1)
                print(train_failure_labels_y[example_idx], " idx_with_Anomaly_1:  ", feature_names[idx_with_Anomaly_1], " with counts: ", count_dim_anomalies_1)
                #print(train_failure_labels_y[example_idx],"idx_with_Anomaly_2: ", idx_with_Anomaly_2)
                print(train_failure_labels_y[example_idx], "idx_with_Anomaly_2: ", feature_names[idx_with_Anomaly_2], " with counts: ", count_dim_anomalies_2)



if __name__ == '__main__':
    main()
