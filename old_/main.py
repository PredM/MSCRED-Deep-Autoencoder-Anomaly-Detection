# from _ctypes import sizeof
# from ctypes import memset

import numpy as np
from DN.MscredModel import MSCRED
import os
import sys
import tensorflow.compat.v1 as tf
import pandas as pd
#import sklearn.model_selection as model_selection

tf.disable_v2_behavior()
# import logging

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

# run able server
sys.path.append(os.path.abspath("."))

from configuration.Configuration import Configuration

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = Configuration()


# method to import and concatenate signature matrix of different length in the right order starting with the shortest
# time sequence
def importSignatureMatrix(path):
    print('Import Signature Matrix')
    curr_data_set = []
    for w in range(0, len(config.win_size)):
        for f in os.listdir(path):
            if f.find(config.filename_matrix_data) != (-1) and f[len(f) - 4: len(f)] == '.npy' and f.find(
                    str(config.win_size[w])) != (-1):
                curr_data_set_w = np.expand_dims(np.load(path + f), axis=3)
                if w == 0:
                    curr_data_set = curr_data_set_w
                else:
                    curr_data_set = np.concatenate((curr_data_set, curr_data_set_w), axis=3)
                break
    return curr_data_set

def importSignatureMatrix2(path):
    print('Import Signature Matrix 2')
    curr_data_set = []
    for run in range(4): #1
        print("imported:",w," of 71")
        for w in range(0, len(config.win_size)):
            #curr_data_set_w = np.expand_dims(np.load(path + "matrix_win_"+str(config.win_size[w])+"_51.npy"), axis=3)
            curr_data_set_w = np.expand_dims(np.load(path + "matrix_win_" + str(config.win_size[w]) + "_"+ str(run) + ".npy"),
                                             axis=3)

            if w == 0:
                curr_data_set = curr_data_set_w
            else:
                curr_data_set = np.concatenate((curr_data_set, curr_data_set_w), axis=3)
    return curr_data_set


# method to import signature matrix of different length in the right order starting with the shortest time sequence
# multiply imported matrix with filter matrix and concatenate filtered matrices
def importfilteredSignatureMatrix(path, filter):
    print('Import Signature Matrix and create filtered Signaturematrix')
    curr_data_set = []
    for w in range(0, len(config.win_size)):
        for f in os.listdir(path):
            if f.find(config.filename_matrix_data) != (-1) and f[len(f) - 4: len(f)] == '.npy' and f.find(
                    str(config.win_size[w])) != (-1):
                curr_data_set_w = np.load(path + f)
                curr_dataset_filtered = np.zeros(curr_data_set_w.shape)
                for i in range(0, curr_data_set_w.shape[0] - 1):
                    for rows in range(0, curr_data_set_w.shape[1] - 1):
                        for columns in range(0, curr_data_set_w.shape[2] - 1):
                            curr_dataset_filtered[i][rows][columns] = curr_data_set_w[i][rows][columns] * filter[rows][
                                columns]
                curr_data_set_w = np.expand_dims(curr_data_set_w, axis=3)
                if w == 0:
                    curr_data_set = curr_data_set_w
                else:
                    curr_data_set = np.concatenate((curr_data_set, curr_data_set_w), axis=3)
                break
    return curr_data_set

def main():
    # start trained NN with evaluation or test data
    def runNetwork(path, pos, curr_data_set):
        interation_counter = 0
        print("Aktuelle Trainingdaten: (Bsp, AnzAttribute, AnzAttribute, ZeitDimensionen): ", curr_data_set.shape)

        num_of_examples = curr_data_set.shape[0]
        start_pos = 0
        curr_pos = start_pos + batch_size

        # iterate over all data to reconstruct inputs
        while curr_pos < num_of_examples:
            # define data for the current run
            curr_batch_input_of_sig_matrixes = curr_data_set[start_pos:curr_pos, :, :, :]

            # start NN with data
            res_no_failure, error_val, loss_val = sess.run([reconstructed_input, reconstruction_error, loss],
                                                           feed_dict={input_ph: curr_batch_input_of_sig_matrixes})

            print("Iteration: ", interation_counter, " | current position: ", curr_pos, " | loss: ", loss_val)

            # concatenante the reconstruction error for every reconstructed signature matrix
            if (start_pos == 0):
                res = error_val
            else:
                res = np.concatenate((res, error_val), axis=0)

            curr_pos = curr_pos + config.step_size_reconstruction
            start_pos = start_pos + config.step_size_reconstruction
            interation_counter = interation_counter + 1
        if not os.path.exists(path + config.directoryname_NNresults):
            os.makedirs(path + config.directoryname_NNresults)

        # save reconstruction error
        path_save_file = path + config.directoryname_NNresults + pos + config.filename_reconstruction_error
        np.save(path_save_file, res)
        print('-------------------------------')
        print()

    ### creation of a matrix which contains 0 for sensors without a relationship and 1 for sensors with a relationship
    if config.filter == True:
        print('Create Filter')
        sensor_n = len(config.featuresAll)
        data = pd.read_pickle(os.path.abspath(".") + config.datasets[config.no_failure][0][2::] +
                              config.directoryname_training_data + config.filename_pkl)
        col = data.columns
        filter = np.zeros((sensor_n, sensor_n))
        for i in range(0, sensor_n):
            for j in range(0, sensor_n):
                for controler in config.controler_names:
                    bool_i = False
                    bool_j = False
                    for n in range(0, len(controler)):
                        if col[i].find(controler[n]) != (-1):
                            bool_i = True
                        if col[j].find(controler[n]) != (-1):
                            bool_j = True
                        if bool_i == True and bool_j == True:
                            filter[i][j] = 1
                            break

    ### create model
    print('-------------------------------')
    print('Creation of the model')

    # number of examples during one iteration
    batch_size = config.batch_size
    learning_rate = config.learning_rate

    # create graph structure of the NN
    model2 = MSCRED.create_model()
    print(model2.summary())

    ### training
    trainings_interation_counter = 0
    print('-------------------------------')
    print('Start NN for training dataset ')
    path = os.path.abspath(".") + config.datasets[config.no_failure][0][2::] + config.directoryname_training_data

    if config.filter == False:
        #curr_data_set = importSignatureMatrix(path + config.directoryname_matrix_data + d + '/')
        curr_data_set = importSignatureMatrix2( '../data/matrix_data/')
        print("curr_data_set shape: ", curr_data_set.shape)
    else:
        curr_data_set = importfilteredSignatureMatrix(path +config.directoryname_matrix_data + d + '/', filter)
    print("Aktuelle Trainingdaten: (Bsp, AnzAttribute, AnzAttribute, ZeitDimensionen): ", curr_data_set.shape)

    num_of_examples = curr_data_set.shape[0]
    start_pos = 0
    curr_pos = start_pos + batch_size

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    #Generate training sequences of size s
    s = 5
    curr_example = np.zeros((s,curr_data_set.shape[1],curr_data_set.shape[2],curr_data_set.shape[3]))
    curr_xTrainData = np.zeros((1,s, curr_data_set.shape[1], curr_data_set.shape[2], curr_data_set.shape[3]))
    for i in range(100): # num_of_examples - s
        if i == 0:
            curr_example = curr_data_set[i:i+s,:,:,:]
            curr_xTrainData[0,:,:,:,:] = curr_example
        else:
            #curr_example = np.concatenate((curr_example, curr_data_set[i:i+s,:,:,:]), axis=0)
            curr_example = curr_data_set[i:i + s, :, :, :]
            curr_example = np.expand_dims(curr_example, axis=0)
            #print("curr_example: ", curr_example.shape)
            #print("curr_xTrainData: ", curr_xTrainData.shape)
            curr_xTrainData = np.concatenate((curr_xTrainData, curr_example), axis=0)

    print("curr_example: ", curr_example.shape)
    #X_train, X_test = model_selection.train_test_split(curr_example, test_size=0.1, random_state=42)
    X_train = curr_xTrainData[0:90,:,:,:,:]
    X_test = curr_xTrainData[90:99, :, :, :, :]
    print("X_train.shape:", X_train.shape)
    history = model2.compile(optimizer='adam', loss=tf.keras.losses.mse) #
    # Reducing input to the last time step
    X_train_y = X_train[:,4,:,:,:]
    X_test_y = X_test[:,4,:,:,:]
    print("X_train_y shape: ", X_train_y.shape)
    #model2.fit(X_train,X_train_y,epochs=3,batch_size=16, shuffle=True, validation_data=(X_test, X_test_y), callbacks=[es, mc]) # validation_split=0.2

    model2 = tf.keras.models.load_model('best_model.h5')

    input_pred =  np.expand_dims(X_train[0, :, :, :, :], axis=0)
    input_pred.astype(np.float32)
    input_pred = tf.cast(input_pred, 'float32')
    pred = model2.predict(X_test, steps=1)
    print("Predictions:", pred.shape) #((9, 61, 61, 3))

    X_test_y = np.squeeze(X_test_y)

    reconstruction_error = np.zeros((pred.shape[0],pred.shape[3]))

    for i_example in range(pred.shape[0]): #Iterate over all predictions
        for i_dim in range(pred.shape[3]): #Iterate over all dimensions
            curr_matrix_pred = pred[i_example,:,:,i_dim]
            curr_matrix_input = X_test_y[i_example,:,:,i_dim]
            diff = curr_matrix_pred - curr_matrix_input
            print("curr_matrix_pred shape: ", curr_matrix_pred.shape)
            mse = np.mean(np.square(curr_matrix_input - curr_matrix_pred))
            diff_paper_formula = np.square(np.linalg.norm(diff, ord='fro'))
            #diff_paper_formula_axis0 = np.square(np.linalg.norm(diff, ord='fro', axis=0))
            #diff_paper_formula_axis1 = np.square(np.linalg.norm(diff, ord='fro', axis=1))
            mse_axis0 = np.mean(np.square(curr_matrix_input - curr_matrix_pred), axis=0)
            mse_axis1 = np.mean(np.square(curr_matrix_input - curr_matrix_pred), axis=1)
            print("example: ", i_example, "dim: ", i_dim, "Rec.Err.: ", diff_paper_formula, "MSE: ", mse)
            #print("axis0: ", mse_axis0)






    # iterate over all training examples to train the NN

    print('finished training')
    print('-------------------------------')
    print()

    # start NN with data
    nbr_datasets = len(config.datasets)

    # Check Resconstruction Error over the whole training data set



    for i in range(0, nbr_datasets):
        print('-------------------------------')
        print('Start trained NN with dataset', i)
        # if dataset is not training data set
        if i != config.no_failure:
            path = os.path.abspath(".") + config.datasets[i][0][2::]
            # concatenate signature matrixes of different length
            if config.filter == False:
                curr_data_set = importSignatureMatrix(path + config.directoryname_matrix_data)
            else:
                curr_data_set = importfilteredSignatureMatrix(path + config.directoryname_matrix_data, filter)
            runNetwork(path, 'start0', curr_data_set)
        else:
            path = os.path.abspath(".") + config.datasets[i][0][2::] + config.directoryname_eval_data
            for root, dirs, files in os.walk(path + config.directoryname_matrix_data, topdown=False):
                for d in dirs:
                    # concatenate signature matrixes of different length
                    if config.filter == False:
                        curr_data_set = importSignatureMatrix(path + config.directoryname_matrix_data + d + '/')
                    else:
                        curr_data_set = importfilteredSignatureMatrix(path + config.directoryname_matrix_data + d + '/',
                                                                      filter)
                    runNetwork(path, d, curr_data_set)

    def my_loss_fn(y_true, y_pred):
        # MSCRED Implementation
        #squared_difference = tf.square(y_true - y_pred)
        loss = np.square(np.linalg.norm((y_true - y_pred), ord='fro'))
        print("loss dim: ", loss.shape)
        return tf.reduce_mean(loss, axis=-1)  # Note the `axis=-1`
if __name__ == '__main__':
    main()
