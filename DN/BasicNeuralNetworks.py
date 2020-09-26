import sys
from os import listdir, path

import tensorflow as tf

#from configuration.Hyperparameter import Hyperparameters


class NN:

    def __init__(self, hyperparameters, input_shape):
        #self.hyper: Hyperparameters = hyperparameters
        self.input_shape = input_shape
        self.model: tf.keras.Sequential = tf.keras.Sequential()

    def create_model(self):
        raise AssertionError('No model creation for abstract NN class possible')

    def print_model_info(self):
        self.model.summary()

    def get_parameter_count(self):
        total_parameters = 0

        for variable in self.model.trainable_variables:
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim

            total_parameters += variable_parameters

        return total_parameters

    def load_model_weights(self, model_folder):
        if self.model is None:
            raise AttributeError('Model not initialised. Can not load weights.')

        if type(self) == CNN or type(self) == RNN or type(self) == TCN or type(self) == CNNWithClassAttention or type(
                self) == CNN1DWithClassAttention or type(self) == CNN2D:
            prefix = 'encoder'
        elif type(self) == FFNN:
            prefix = 'ffnn'
        elif type(self) == FFNN2:
            prefix = 'ffnn'
        else:
            raise AttributeError('Can not import models of type', type(self))

        found = False
        for file_name in listdir(model_folder):

            if file_name.startswith(prefix):
                self.model.load_weights(path.join(model_folder, file_name))
                found = True

        if not found:
            raise FileNotFoundError('Model file for this type could not be found in ' + str(model_folder))
        else:
            print('Model has been loaded successfully')

    def get_output_shape(self):
        return self.model.output_shape


# Used for Neural Warp
class FFNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN for input shape: ', self.input_shape)

        layers = self.hyper.ffnn_layers.copy()

        if len(layers) < 1:
            print('FFNN with less than one layer is not possible')
            sys.exit(1)

        # first layer must be handled separately because the input shape parameter must be set
        num_units_first = layers.pop(0)
        input = tf.keras.Input(shape=self.input_shape, name="Input")

        x = tf.keras.layers.Dense(units=num_units_first, activation=tf.keras.activations.relu,
                                  input_shape=self.input_shape)(input)

        for num_units in layers:
            x = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(x)

        # regardless of the configured number of layers, add a layer with
        # a single neuron that provides the indicator function output.
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        self.model = tf.keras.Model(inputs=input, outputs=output)


# Used for Taigman 2014 approach
class FFNN2(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating FFNN2 with a single sigmoid neurorn for input shape: ', self.input_shape)

        input = tf.keras.Input(shape=(1952 + 64 + 61,), name="Input")
        '''
        x = tf.keras.layers.Dense(units=num_units_first, activation=tf.keras.activations.relu,
                                  input_shape=self.input_shape)(input)

        for num_units in layers:
            x = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(x)

        # regardless of the configured number of layers, add a layer withd
        # a single neuron that provides the indicator function output.
        output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)
        '''
        '''
        i1 = tf.keras.layers.Lambda(lambda x: x[:,0:1951])(input)
        i2 = tf.keras.layers.Lambda(lambda x: x[:, 1952:1952+64])(input)
        o1 = tf.keras.layers.Dense(1, activation='sigmoid', )(i1)
        o2 = tf.keras.layers.Dense(1, activation='sigmoid', )(i2)
        output = (o1+o2)/3
        '''
        output = tf.keras.layers.Dense(1952 + 64, activation='sigmoid', )(input)
        output = tf.keras.layers.Dense(1, activation='sigmoid', )(output)
        '''
        x = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid', )(x)
        '''
        '''
        x = tf.reshape(input,(32,61))
        x = tf.transpose(x)
        x = tf.reshape(x, (1, 61, 32))
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu), )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu), )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=1, activation=tf.keras.activations.tanh), )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.squeeze(x)
        x = tf.reshape(x,(1,61))
        output = tf.keras.layers.Dense(1, activation='sigmoid',)(x)
        '''
        # use_bias=False, kernel_constraint= tf.keras.constraints.NonNeg()
        self.model = tf.keras.Model(inputs=input, outputs=output)


class RNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    # RNN structure matching the description in the neural warp paper
    # currently not used
    def create_model_nw(self):
        print('Creating LSTM encoder')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM encoder with less than one layer is not possible')
            sys.exit(1)

        # bidirectional LSTM network, type where timelines are only combined ones
        # create one timeline and stack into StackedRNNCell
        cells = []
        for num_units in layers:
            cells.append(tf.keras.layers.LSTMCell(units=num_units, activation=tf.keras.activations.tanh))

        stacked_cells = tf.keras.layers.StackedRNNCells(cells)
        rnn = tf.keras.layers.RNN(stacked_cells, return_sequences=True)

        # create a bidirectional network using the created timeline, backward timeline will be generated automatically
        model.add(tf.keras.layers.Bidirectional(rnn, input_shape=self.input_shape))

        # add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model

    def create_model(self):
        print('Creating LSTM encoder')

        model = tf.keras.Sequential(name='RNN')

        layers = self.hyper.lstm_layers

        if len(layers) < 1:
            print('LSTM encoder with less than one layer is not possible')
            sys.exit(1)

        for i in range(len(layers)):
            num_units = layers[i]

            # first layer must be handled separately because the input shape parameter must be set Usage of default
            # parameters should ensure cuDNN usage (
            # https://www.tensorflow.org/beta/guide/keras/rnn#using_cudnn_kernels_when_available)
            # Even though .LSTM should use cuDnn Kernel the .RNN is faster
            # Also a not yet fixable error occurs, which is why this could be the case
            if i == 0:
                layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units), return_sequences=True,
                                            input_shape=self.input_shape)
                # layer = tf.keras.layers.LSTM(units=num_units, return_sequences=True, input_shape=self.input_shape)
            else:
                layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(num_units), return_sequences=True)
                # layer = tf.keras.layers.LSTM(units=num_units, return_sequences=True)
            model.add(layer)

        # add Batch Norm and Dropout Layers
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        self.model = model


class CNNWithClassAttention(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)
        self.output_shape = None

    def create_model(self):

        print('Creating CNN with 2d kernel encoder with a sensor data input shape: ', self.input_shape[0],
              " and additional input shape: ", self.input_shape[1])

        # Input definition of sensor data and masking
        sensor_data_input = tf.keras.Input(shape=(self.input_shape[0][0], self.input_shape[0][1], 1),
                                           name="SensorDataInput")
        case_dependent_vector_input_i = tf.keras.Input(self.input_shape[1], name="MaskingVectorInput")
        masking_vec_len = self.input_shape[1]

        # Splitting masking vectors in normal and strict
        if self.hyper.use_additional_strict_masking == 'True':
            print("Masking: normal + strict")
            half = int(masking_vec_len / 2)
            case_dependent_vector_input = tf.keras.layers.Lambda(lambda x: x[:, :half], name="SplitMaskVec_Context")(
                case_dependent_vector_input_i)
            case_dependent_vector_input_strict = tf.keras.layers.Lambda(lambda x: x[:, half:masking_vec_len],
                                                                        name="SplitMaskVec_Strict")(
                case_dependent_vector_input_i)
        else:
            print("Masking: normal + strict")
            case_dependent_vector_input = case_dependent_vector_input_i
            case_dependent_vector_input_strict = case_dependent_vector_input_i

        layers = self.hyper.cnn2d_layers

        # Different options to learn the weights that are used for the distance measure
        if self.hyper.learnFeatureWeights == 'True':
            # Takes 0-1 input mask of relevant features and learns weights for relevant features
            # (that are not masked as zero)
            print("learnFeatureWeights:True - Feature weights are learned based on masking vector (original/best)")

            case_dependent_vector_input_ = tf.expand_dims(case_dependent_vector_input_strict, -1,
                                                          name="PrepareInputDim")
            case_dependent_vector_input_o = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu, ),
                name="distance_weight_adjustment")(case_dependent_vector_input_)
            case_dependent_vector_input_o = tf.squeeze(case_dependent_vector_input_o, name="PrepareOutputDim")

        elif self.hyper.learnFeatureWeights == 'OneWeight':
            print("learnFeatureWeights:OneWeight - One weight is learned for all features")
            # case_dependent_vector_input_o = WeightScalingLayer()(case_dependent_vector_input)

            kvar = tf.keras.backend.ones((1, 1))
            one = tf.keras.backend.eval(kvar)
            case_dependent_vector_input_o = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu,
                                                                  name="distance_weight_adjustment")(one)
            case_dependent_vector_input_o = tf.keras.backend.repeat_elements(case_dependent_vector_input_o, rep=61,
                                                                             axis=1)
            case_dependent_vector_input_o = tf.multiply(case_dependent_vector_input_o,
                                                        case_dependent_vector_input_strict)

        elif self.hyper.learnFeatureWeights == 'OneWeightPredicted':
            print("learnFeatureWeights:OneWeightPredicted - Feature weights are learned based on masking vector "
                  "(scaling factor predicted based on masking vector)")
            # scaler = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(case_dependent_vector_input)
            # scaler = tf.keras.layers.BatchNormalization()(scaler)
            scaler = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu,
                                           name="distance_weight_adjustment")(case_dependent_vector_input)
            # case_dependent_vector_input_o = tf.add(case_dependent_vector_input, scaler)
            case_dependent_vector_input_o = tf.multiply(case_dependent_vector_input_strict, scaler)

        elif self.hyper.learnFeatureWeights == 'OneIndividualWeight':
            print("learnFeatureWeights:OneIndividualWeight - Feature weights are learned based on masking vector and "
                  "one hot sensor encoding (scaling factor predicted based on masking vector and one-hot-encoding)")
            case_dependent_matrix_input_ = tf.tile(case_dependent_vector_input, [1, 61])
            # case_dependent_matrix_input = case_dependent_matrix_input / tf.reduce_sum(case_dependent_matrix_input)
            reshape = tf.keras.layers.Reshape((61, 61))
            case_dependent_matrix_input_ = reshape(case_dependent_matrix_input_)
            kvar = tf.keras.backend.eye(61)
            one_hot_sensor = tf.keras.backend.eval(kvar)
            one_hot_sensor = tf.expand_dims(one_hot_sensor, 0)

            # TRAINING: bei batchsize 64 zu 128, bei training batchsize auf 4 ändern und dann zu 8
            # TODO 132 durch doppelten Batchsizewert (aktuell 66) ersetzen, Problem das bei Initalisierung nicht bekannt
            one_hot_sensor = tf.keras.backend.tile(one_hot_sensor, [132, 1, 1])

            # one_hot_sensor = tf.tile(one_hot_sensor, [1, 8])
            # one_hot_sensor = tf.keras.backend.reshape(one_hot_sensor, [8, 61,61])
            case_with_sensor_concatenated = tf.keras.backend.concatenate((case_dependent_matrix_input_, one_hot_sensor))
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu), )(case_with_sensor_concatenated)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=16, activation=tf.keras.activations.relu), )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid), )(x)
            case_dependent_vector_input_o = tf.multiply(tf.squeeze(x), case_dependent_vector_input_strict)
            case_dependent_vector_input_o = tf.add(case_dependent_vector_input_o, case_dependent_vector_input_strict)
            # debug = case_dependent_vector_input_o

        else:
            print("learnFeatureWeights:False Feature weights are similar to masking vector")
            # case_dependent_vector_input_o = tf.keras.layers.GaussianNoise(0.3)(case_dependent_vector_input_strict)
            # case_dependent_vector_input_o = tf.multiply(case_dependent_vector_input_o, case_dependent_vector_input_strict)
            case_dependent_vector_input_o = case_dependent_vector_input_strict

        # Create a matrix based on masking vectors for using it as "attention"-like in ABCNN-1
        # (https://arxiv.org/abs/1512.05193)
        if self.hyper.abcnn1 == 'softmax' or self.hyper.abcnn1 == 'weighted':
            if self.hyper.abcnn1 == 'softmax':
                print("ABCNN1 softmax variant used")
                case_dependent_vector_input_processed = tf.keras.layers.Softmax()(case_dependent_vector_input)

            elif self.hyper.abcnn1 == 'weighted':
                print("ABCNN1 masking-vector weighted variant used")
                case_dependent_vector_input_processed = case_dependent_vector_input / tf.reduce_sum(
                    case_dependent_vector_input)
            case_dependent_matrix_input = tf.tile(case_dependent_vector_input_processed, [1, self.input_shape[0][0]])
            # case_dependent_matrix_input = case_dependent_matrix_input / tf.reduce_sum(case_dependent_matrix_input)
            reshape = tf.keras.layers.Reshape((self.input_shape[0][0], self.input_shape[0][1]))
            case_dependent_matrix_input = reshape(case_dependent_matrix_input)
        else:
            print("ABCNN1 not used")
            self.hyper.abcnn1 = None

        if len(layers) < 1:
            print('CNN encoder with less than one layer for 2d kernels is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn2d_layers, self.hyper.cnn2d_kernel_length, self.hyper.cnn2d_strides))

        # Creating 2d-CNN encoder for sensor data
        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv2d_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                       kernel_size=(filter_size),
                                                       strides=stride, input_shape=sensor_data_input.shape)

                # Add 1D-Conv Layer to provide information across time steps in the first layer
                if self.hyper.use1dContext == 'True':
                    print('1d Conv as add. Input used to get context across time series')
                    conv1d_layer = tf.keras.layers.Conv1D(filters=self.input_shape[0][1], padding='VALID',
                                                          kernel_size=1,
                                                          strides=1)
                    # inp = tf.squeeze(sensor_data_input)
                    reshape = tf.keras.layers.Reshape((self.input_shape[0][0], self.input_shape[0][1]))
                    inp = reshape(sensor_data_input)
                    temp = conv1d_layer(inp)
                    # temp = tf.keras.layers.BatchNormalization()(temp)
                    # temp = tf.keras.activations.sigmoid(temp)
                    temp = tf.expand_dims(temp, -1)
                    # Add 1d contextual information as input
                    sensor_data_input2 = tf.concat([sensor_data_input, temp], axis=3)

                else:
                    print('1d Conv as add. Input NOT used to get context across time series')
                    sensor_data_input2 = sensor_data_input

                # Add ABCNN matrix from beginning
                if self.hyper.abcnn1 is not None:
                    case_dependent_matrix_input = tf.expand_dims(case_dependent_matrix_input, -1)
                    sensor_data_input2 = tf.concat([sensor_data_input2, case_dependent_matrix_input], axis=3,
                                                   name="Final_Sensor_Data_Input")

                x = conv2d_layer1(sensor_data_input2)
                # x = tf.keras.layers.SpatialDropout2D(rate=self.hyper.dropout_rate)(x)
            else:
                conv2d_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                      kernel_size=(filter_size),
                                                      strides=stride)
                x = conv2d_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

        # use of 1d CNN after 2d CNN
        if len(self.hyper.cnn_layers) < 1:
            print('No 1d conv on top of 2d conv is used.')
        else:
            print('1d conv on top of 2d conv is used.')

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        # creating CNN encoder for sensor data
        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                strides=stride)
            x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # Skip Connection / Shortcut
        '''
        shortcut = tf.keras.layers.Conv2D(1, (8, 1), strides=(8,1),
                                 kernel_initializer='he_normal',
                                 name="Shortcut" + '1')(sensor_data_input2)
        shortcut = tf.keras.layers.BatchNormalization( name="BN" + '1')(shortcut)
        shortcut = tf.keras.layers.Lambda(lambda x: x[:, :-2], name='slice')(shortcut)
        reshape = tf.keras.layers.Reshape((123, 61))
        shortcut = reshape(shortcut)
        #shortcut = tf.squeeze(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        #x = tf.keras.layers.add()([x, shortcut])
        '''
        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # Attribute-wise feature aggregation via (time-distributed) fully-connected layers
        # TODO PK change naming of channels to features
        if self.hyper.useChannelWiseAggregation:
            print('Adding FC layers for attribute wise feature merging/aggregation')
            layers_fc = self.hyper.cnn2d_channelWiseAggregation.copy()
            # x = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
            for num_units in layers_fc:
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu),
                    name="FC_FeatureWise_Aggreg_Layer_" + str(num_units) + "U")(x)
            x = tf.keras.layers.Permute((2, 1))(x)  # transpose
        # Output 1, used for weighted distance measure
        o1 = tf.keras.layers.Multiply()([x, case_dependent_vector_input_strict])

        # Using an additional context vector that is calculated on the previously defined output
        if self.hyper.useAddContextForSim == "True":
            print('Additional feature restricted content vector is used')

            # Learn a weight value how much the context should be considered in sim against single feature weighted
            if self.hyper.useAddContextForSim_LearnOrFixWeightVale == "True":
                print('Learn weight value how much context is considered for each failure mode')
                layers_fc = self.hyper.cnn2d_learnWeightForContextUsedInSim.copy()

                for num_units in layers_fc:
                    case_dependent_vector_input_2 = tf.keras.layers.Dense(units=num_units,
                                                                          activation=tf.keras.activations.relu,
                                                                          name="Weight_Betw_Distances_" + str(
                                                                              num_units) + "U")(
                        case_dependent_vector_input)
                    case_dependent_vector_input_2 = tf.keras.layers.BatchNormalization()(case_dependent_vector_input_2)

                w = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                          name="Weight_Betw_Distances")(case_dependent_vector_input_2)

            else:
                # using a fixed value as output does not work. fix defined in simple similarity measure
                # w = 0.5 #tf.Variable(0.5)#tf.convert_to_tensor(0.5)#float(self.hyper.useAddContextForSim_LearnOrFixWeightVale)
                # w = tf.convert_to_tensor(tf.keras.backend.var(0.5))
                # w = float(self.hyper.useAddContextForSim_LearnOrFixWeightVale)
                print('Fixed weight value how much context is considered for each failure mode: ',
                      self.hyper.useAddContextForSim_LearnOrFixWeightVale)

            print('Adding FC layers for context merging/aggregation')
            layers_fc = self.hyper.cnn2d_contextModule.copy()

            # Context Module: connect only features from relevant attributes

            # gate: only values from relevant sensors:
            # gates = tf.nn.sigmoid(case_dependent_vector_input)
            c = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            # build context module:
            c = tf.keras.layers.Flatten()(c)

            for num_units in layers_fc:
                c = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu,
                                          name="FC_Layer_Context_" + str(num_units) + "U")(c)
                c = tf.keras.layers.BatchNormalization()(c)
            o2 = tf.keras.layers.Reshape([layers_fc[len(layers_fc) - 1], 1])(c)

        else:
            print("No additional context pair for similarity calculation used.")

        if self.hyper.use_weighted_distance_as_standard_ffnn_hyper == "True":
            print("Taigman Approach for learning weighted distance")
            x = tf.keras.layers.Multiply()([x, case_dependent_vector_input])
            x = tf.keras.layers.Flatten()(x)
            c = tf.squeeze(o2)
            o1 = tf.concat([x, c], 1)

        # Create Model:
        if self.hyper.useAddContextForSim == "True":
            # Output:
            # o1: encoded time series as timeSteps x attributes Matrix (if useChannelWiseAggregation==False, else features x attributes Matrix
            # case_dependent_vector_input_o: same as masking vector if learnFeatureWeights==False, else values weights learned (but not for 0s)
            # o2: context vector, FC Layer on masked output (only relevant attributes considered)
            # w: weight value (scalar) how much the similiarity for each failuremode should be based on invidivual features (x) or context (c)
            # debug: used for debugging
            if self.hyper.useAddContextForSim_LearnOrFixWeightVale == "True":
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i],
                                            outputs=[o1, case_dependent_vector_input_o, o2, w])
            else:
                self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i],
                                            outputs=[o1, case_dependent_vector_input_o, o2])
        else:
            self.model = tf.keras.Model(inputs=[sensor_data_input, case_dependent_vector_input_i],
                                        outputs=[o1, case_dependent_vector_input_o])
        '''
        self.intermediate_layer_model = tf.keras.Model(inputs=case_dependent_vector_input,
                                                      outputs=self.model.get_layer("reshape").output)
        '''

    def get_output_shape(self):
        # output shape onyl from first output x
        return self.model.output_shape[0]
        # raise NotImplementedError('Must be added in order for ffnn version to work with this encoder')


# TODO @klein Remove old code that is / will not be used
class CNN1DWithClassAttention(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating CNN encoder with 1DConv and Modulation/Conditioning  with an input shape: ', self.input_shape)
        sensorDataInput = tf.keras.Input(self.input_shape[0], name="Input0")
        # this is the number of classes in y_train
        caseDependentVectorInput = tf.keras.Input(self.input_shape[1], name="Input1")

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        x = sensorDataInput
        # creating CNN encoder for sensor data
        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride, input_shape=self.input_shape)
            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)

            x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # creating a case vector encoder
        num_filterLastLayer = layer_properties[len(layer_properties) - 1][0]
        # caseDepVectEmbedding = tf.keras.layers.Dense((int(num_filterLastLayer/2)), activation='relu')(caseDependentVectorInput)
        # caseDepVectEmbedding = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(caseDepVectEmbedding)
        gates = tf.keras.layers.Dense(num_filterLastLayer, activation='sigmoid')(caseDependentVectorInput)

        # merging case vector and sensor encoding
        # a) ADD
        # caseDepVectEmbedding = tf.keras.layers.Softmax()(caseDepVectEmbedding)
        # embedding = tf.keras.layers.Add()([x, caseDepVectEmbedding])
        # b) MULTIPLY
        # gates = tf.nn.sigmoid(caseDepVectEmbedding)
        embedding = tf.keras.layers.Multiply()([x, gates])
        # c) CONCATENATE
        '''
        flat = tf.keras.layers.Flatten()(x)
        concat = tf.concat([flat, caseDepVectEmbedding], 1)
        # concat = tf.keras.layers.Concatenate()([flat, caseDepVectEmbedding])
        embedding = tf.keras.layers.Dense(1024, activation='relu')(concat)
        embedding = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(embedding)
        embedding = tf.keras.layers.Dense(256, activation='relu')(embedding)
        embedding = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(embedding)
        embedding = tf.keras.layers.Dense(64, activation='sigmoid')(embedding)
        print("sensorDataInput.shape[0]: ", sensorDataInput.shape[0])
        dim0 = self.hyper.batch_size * 2  # for Inference: 16450  #in Training: self.hyper.batch_size*2
        embedding = tf.reshape(embedding, [dim0, embedding.shape[1], 1])
        '''
        self.model = tf.keras.Model(inputs=[sensorDataInput, caseDependentVectorInput], outputs=embedding)
        # Add: softmax
        # Multiply: dense_1
        '''
        self.intermediate_layer_model = tf.keras.Model(inputs=caseDependentVectorInput,
                                                       outputs=self.model.get_layer("dense").output)
        self.intermediate_layer_model1 = tf.keras.Model(inputs=[sensorDataInput, caseDependentVectorInput],
                                                       outputs=self.model.get_layer("re_lu_1").output)
        self.intermediate_layer_model2 = tf.keras.Model(inputs=[sensorDataInput, caseDependentVectorInput],
                                                       outputs=self.model.get_layer("multiply").output)
        '''
        # Query-value attention of shape [batch_size, Tq, filters].
        # print("inputs", inputs)
        # print("x: ", x)
        # input_lastConvLayer_attention_seq = tf.keras.layers.Attention()([x, caseDepVectEmbedding])


class CNN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        print('Creating CNN encoder')
        model = tf.keras.Sequential(name='CNN')

        layers = self.hyper.cnn_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride, input_shape=self.input_shape)
            else:
                conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                    strides=stride)

            model.add(conv_layer)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Dropout(rate=self.hyper.dropout_rate))

        if self.hyper.fc_after_cnn1d_layers is not None:
            print('Adding FC layers')
            layers_fc = self.hyper.fc_after_cnn1d_layers.copy()
            if len(layers) < 1:
                print('Adding FC with less than one layer is not possible')
                sys.exit(1)
            model.add(tf.keras.layers.Flatten())
            for num_units in layers_fc:
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu))

            # Normalize final output as recommended in Roy et al (2019) Siamese Networks: The Tale of Two Manifolds
            # model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.Softmax()) # Martin et al. (2017) ICCBR
            model.add(tf.keras.layers.Reshape((model.layers[len(model.layers) - 1].output.shape[1], 1)))

        self.model = model


# TODO @klein Remove old code that is / will not be used
class CNN2D(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)

    def create_model(self):
        '''
        Based on https://www.ijcai.org/proceedings/2019/0932.pdf
        '''
        print('Creating CNN with 2d kernel encoder with an input shape: ', self.input_shape)
        sensorDataInput = tf.keras.Input(shape=self.input_shape, name="Input0")

        layers = self.hyper.cnn2d_layers

        if len(layers) < 1:
            print('CNN encoder with less than one layer for 2d kernels is not possible')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn2d_layers, self.hyper.cnn2d_kernel_length, self.hyper.cnn2d_strides))

        # creating CNN encoder for sensor data
        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            # first layer must be handled separately because the input shape parameter must be set
            if i == 0:
                conv_layer1 = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                     kernel_size=(filter_size),
                                                     strides=stride, input_shape=sensorDataInput.shape)
                # Added 1D-Conv Layer to provide information across time steps in the first layer

                # TODO @Klein Fix create_model for cbs
                # Possible fix: self.input_shape[1], but then new error occurs:
                #  ValueError: Shape must be rank 3 but is rank 4 for 'model/tf_op_layer_concat/concat'
                #  (op: 'ConcatV2') with input shapes: [128,1000,6], [128,1000,6,1], [].
                # print(self.input_shape)

                conv_layer1d = tf.keras.layers.Conv1D(filters=61, padding='VALID', kernel_size=1,
                                                      strides=1)
                # inp = tf.squeeze(sensorDataInput)
                reshape = tf.keras.layers.Reshape((self.input_shape[0], self.input_shape[1]))
                inp = reshape(sensorDataInput)
                temp = conv_layer1d(inp)
                temp = tf.expand_dims(temp, -1)
                sensor_data_input2 = tf.concat([sensorDataInput, temp], axis=3)

                x = conv_layer1(sensor_data_input2)
            else:
                conv_layer = tf.keras.layers.Conv2D(filters=num_filter, padding='VALID',
                                                    kernel_size=(filter_size),
                                                    strides=stride)
                x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        # conv1x1_layer = tf.keras.layers.Conv2D(filters=1, padding='VALID',
        #                                     kernel_size=(1, 1),
        #                                     strides=stride)
        # x = conv1x1_layer(x)
        # x = tf.keras.layers.ReLU()(x)
        # reshape necessary to provide a 3d instead of 4 dim for the FFNN or 1D Conv operations on top
        reshape = tf.keras.layers.Reshape((x.shape[1], x.shape[2]))
        x = reshape(x)

        if len(layers) < 1:
            print('Attention: no one 1d conv on top of 2d conv is used!')
            sys.exit(1)

        layer_properties = list(zip(self.hyper.cnn_layers, self.hyper.cnn_kernel_length, self.hyper.cnn_strides))

        # creating CNN encoder for sensor data
        for i in range(len(layer_properties)):
            num_filter, filter_size, stride = layer_properties[i][0], layer_properties[i][1], layer_properties[i][2]

            conv_layer = tf.keras.layers.Conv1D(filters=num_filter, padding='VALID', kernel_size=filter_size,
                                                strides=stride)
            x = conv_layer(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dropout(rate=self.hyper.dropout_rate)(x)

        if self.hyper.fc_after_cnn1d_layers is not None:
            print('Adding FC layers')
            layers_fc = self.hyper.fc_after_cnn1d_layers.copy()
            if len(layers) < 1:
                print('Adding FC with less than one layer is not possible')
                sys.exit(1)
            x = tf.keras.layers.Flatten()(x)
            last_layer_size = 0
            for num_units in layers_fc:
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(x)
                last_layer_size = num_units

            # Normalize final output as recommended in Roy et al (2019) Siamese Networks: The Tale of Two Manifolds
            # model.add(tf.keras.layers.BatchNormalization())
            # model.add(tf.keras.layers.Softmax()) # Martin et al. (2017) ICCBR
            x = tf.keras.layers.Reshape((last_layer_size, 1))(x)

        self.model = tf.keras.Model(inputs=sensorDataInput, outputs=x)


class TemporalBlock(tf.keras.Model):

    def compute_output_signature(self, input_signature):
        pass

    def print_layer_info(self):
        print("dilation_rate: ", self.dilation_rate, "| nb_filters: ", self.nb_filters, "| kernel_size: ",
              self.kernel_size, "| padding: ", self.padding, "| dropout_rate: ", self.dropout_rate)

    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0, input_shape=None):
        super(TemporalBlock, self).__init__()
        assert padding in ['causal', 'same']

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate

        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # block1
        if input_shape is not None:
            self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                                dilation_rate=dilation_rate, padding=padding, kernel_initializer=init,
                                                input_shape=input_shape)
        else:
            self.conv1 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                                                dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)

        self.batch1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac1 = tf.keras.layers.Activation('relu')
        self.drop1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # block2
        self.conv2 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, kernel_initializer=init)
        self.batch2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.ac2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

        #
        self.downsample = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1, padding='same',
                                                 kernel_initializer=init)
        self.ac3 = tf.keras.layers.Activation('relu')

    # noinspection PyMethodOverriding
    def call(self, x, training=False):

        prev_x = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        # print("prev_x.shape[-1]: ", prev_x.shape[-1], "x.shape[-1]: ", x.shape[-1])
        if prev_x.shape[-1] != x.shape[-1]:  # match the dimension
            prev_x = self.downsample(prev_x)
        # print("prev_x.shape: ", prev_x.shape, "x.shape: ", x.shape)

        assert prev_x.shape == x.shape

        return self.ac3(prev_x + x)  # skip connection


class TCN(NN):

    def __init__(self, hyperparameters, input_shape):
        super().__init__(hyperparameters, input_shape)
        self.output_shape = None
        self.layers = []

    def create_model(self):
        print('Creating TCN encoder')
        num_channels = self.hyper.tcn_layers
        num_levels = len(num_channels)
        kernel_size = self.hyper.tcn_kernel_length
        dropout = self.hyper.dropout_rate

        model = tf.keras.Sequential(name='TCN')

        for i in range(num_levels):
            dilation_rate = 2 ** i  # exponential growth
            if i == 0:
                tb = TemporalBlock(dilation_rate, num_channels[i], kernel_size[i], padding='causal',
                                   dropout_rate=dropout, input_shape=self.input_shape)
            else:
                tb = TemporalBlock(dilation_rate, num_channels[i], kernel_size[i], padding='causal',
                                   dropout_rate=dropout)
            self.layers.append(tb)
            model.add(tb)

        self.model = model
        # Required to load previous model, None causes AssertionError
        self.model.build(input_shape=(10, self.input_shape[0], self.input_shape[1]))
        self.output_shape = (None, self.input_shape[0], num_channels[num_levels - 1])

    def get_output_shape(self):
        return self.output_shape

    def print_model_info(self):
        print('Model: "TCN"')
        print('_________________________________________________________________')
        for i in range(len(self.layers)):
            print('Layer', i, self.layers[i].name)
            self.layers[i].print_layer_info()
        print('_________________________________________________________________')


# This layer learns only one parameter to scale predefined values
class WeightScalingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightScalingLayer, self).__init__()
        self.scaling_weight = None
        # self.num_outputs = num_outputs

    def build(self, input_shape):
        def initializer(*args, **kwargs):
            # not used, not working
            w = tf.keras.backend.variable(1.0)
            return w

        self.scaling_weight = self.add_weight(name='scalingWeight',
                                              shape=(1,),
                                              initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.05,
                                                                                             seed=42),
                                              trainable=True)
        super(WeightScalingLayer, self).build(input_shape)

    # noinspection PyMethodOverriding
    def call(self, input_):
        # Add weight
        weight_vector = tf.multiply(input_, self.scaling_weight)
        return weight_vector

    def compute_output_shape(self, input_shape):
        return input_shape
