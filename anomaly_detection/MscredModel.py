import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))
from configuration.Configuration import Configuration
config = Configuration()
import tensorflow.keras.backend as K
import spektral


# Paper "A Deep Neural Network for Unsupervised AnomalyDetection and Diagnosis in Multivariate Time Series Data" by Chuxu Zhang,§∗Dongjin Song,†∗Yuncong Chen,†Xinyang Feng,‡∗Cristian Lumezanu,†Wei Cheng,†Jingchao Ni,†Bo Zong,†Haifeng Chen,†Nitesh V. Chawla
# Implementation of Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED) for Anomaly Detection and Diagnosis in Multivariate Time Series
# Original Impl.: https://github.com/7fantasysz/MSCRED/blob/master/code/MSCRED_TF.py

class MSCRED(tf.keras.Model):

    def __init__(self):

        super(MSCRED, self).__init__()


    def create_model(self, guassian_noise_stddev = None, use_attention = True, use_ConvLSTM = True, use_memory_restriction=False, use_encoded_output=False):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(config.step_max, config.num_datastreams, config.num_datastreams, config.dim_of_dataset), batch_size=None, name="Input0")

        if guassian_noise_stddev is not None:
            adding_noise = tf.keras.layers.GaussianNoise(stddev=guassian_noise_stddev)
            adding_noise_dropout = tf.keras.layers.SpatialDropout3D(guassian_noise_stddev)
            a = adding_noise(signatureMatrixInput)
            signatureMatrixInput_ = adding_noise_dropout(a)
        else:
            signatureMatrixInput_ = signatureMatrixInput


        conv2d_layer1 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[0], strides=config.strides_encoder[0], kernel_size=config.kernel_size_encoder[0],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[0],
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[1], strides=config.strides_encoder[1], kernel_size=config.kernel_size_encoder[1],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[1],
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[2], strides=config.strides_encoder[2], kernel_size=config.kernel_size_encoder[2],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[2],
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[3], strides=config.strides_encoder[3], kernel_size=config.kernel_size_encoder[3],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[3],
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[0], strides=1, kernel_size=config.kernel_size_encoder[0], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[1], strides=1, kernel_size=config.kernel_size_encoder[1], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[2], strides=1, kernel_size=config.kernel_size_encoder[2], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[3], strides=1, kernel_size=config.kernel_size_encoder[3], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[2], strides=config.strides_encoder[3],
                                                              kernel_size=config.kernel_size_encoder[3], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[1], strides=config.strides_encoder[2],
                                                              kernel_size=config.kernel_size_encoder[2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[0], strides=config.strides_encoder[1],
                                                              kernel_size=config.kernel_size_encoder[1], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=config.strides_encoder[0],
                                                              kernel_size=config.kernel_size_encoder[0], padding='same',
                                                              activation='selu')


        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput_)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        if use_ConvLSTM:
            x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
            x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
            x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
            x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

            if use_attention:
                if config.keras_attention_layer_instead_of_own_impl:

                    # Flatten to get (batchsize,Tq,dim)
                    print("config.output_dim[0]: ", config.output_dim[0], config.output_dim[0],config.filter_dimension_encoder[3])
                    x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[0]  * config.output_dim[0]  * config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4_Tq")(x_endcoded_4_CLSTM)
                    x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3_Tq")(x_endcoded_3_CLSTM)
                    x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2_Tq")(x_endcoded_2_CLSTM)
                    x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1_Tq")(x_endcoded_1_CLSTM)
                    # Flatten to get (batchsize,Tv,dim)
                    x_endcoded_4_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[0]  * config.output_dim[0]  * config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4_Tv")(x_endcoded_4_CLSTM_flatten[:, config.step_max - 1, :])
                    x_endcoded_3_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3_Tv")(x_endcoded_3_CLSTM_flatten[:, config.step_max - 1, :])
                    x_endcoded_2_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2_Tv")(x_endcoded_2_CLSTM_flatten[:, config.step_max - 1, :])
                    x_endcoded_1_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1_Tv")(x_endcoded_1_CLSTM_flatten[:, config.step_max - 1, :])

                    # Applying attention after reshaping the input into required format
                    x_endcoded_4_CLSTM = tf.keras.layers.Attention()([x_endcoded_4_CLSTM_flatten_Tv, x_endcoded_4_CLSTM_flatten])
                    x_endcoded_3_CLSTM = tf.keras.layers.Attention()([x_endcoded_3_CLSTM_flatten_Tv, x_endcoded_3_CLSTM_flatten])
                    x_endcoded_2_CLSTM = tf.keras.layers.Attention()([x_endcoded_2_CLSTM_flatten_Tv, x_endcoded_2_CLSTM_flatten])
                    x_endcoded_1_CLSTM = tf.keras.layers.Attention()([x_endcoded_1_CLSTM_flatten_Tv, x_endcoded_1_CLSTM_flatten])

                    # Reshape back again into original shape
                    x_endcoded_4_CLSTM = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM)
                    x_endcoded_3_CLSTM = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM)
                    x_endcoded_2_CLSTM = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM)
                    x_endcoded_1_CLSTM = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM)
                else:
                    ## Attention for x_endcoded_4_CLSTM
                    # Flatten to vector
                    print("config.output_dim[0]: ", config.output_dim[0], config.output_dim[0],config.filter_dimension_encoder[3])
                    x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[0]*config.output_dim[0]*config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
                    x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
                    #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
                    x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
                    #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
                    x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
                    x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[0]*config.output_dim[0]*config.filter_dimension_encoder[3], name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
                    x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
                    x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
                    x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
                    x_endcoded_4_CLSTM = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

                    ## Attention for x_endcoded_3_CLSTM
                    # Flatten to vector
                    x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[1]*config.output_dim[1]*config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
                    x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_3")(
                        x_endcoded_3_CLSTM_last_time_step)
                    x_endcoded_3_CLSTM_scores = tf.reduce_sum(
                        tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                                      keepdims=True), axis=2)
                    x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
                    x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2], name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
                    x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
                    x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
                    x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
                    x_endcoded_3_CLSTM = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

                    ## Attention for x_endcoded_2_CLSTM
                    # Flatten to vector
                    x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[2]*config.output_dim[2]*config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
                    x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
                    x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                                      keepdims=True), axis=2)
                    x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
                    x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1], name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
                    x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
                    x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
                    x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
                    x_endcoded_2_CLSTM = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

                    ## Attention for x_endcoded_1_CLSTM
                    # Flatten to vector
                    x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[3]*config.output_dim[3]*config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
                    x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
                    x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                                      keepdims=True), axis=2)
                    x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
                    x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0], name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
                    x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
                    x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
                    x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
                    x_endcoded_1_CLSTM = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
            else:
                print("Use Attention: ",use_attention,", instead last ConvLSTM Sequence output is used for reconstruction")
            if use_memory_restriction:
                print("Memory restriction is used on the encoder!")
                ### Memory ###
                if config.use_filter_for_memory:
                    print("Memory stores channels/filters, config.use_filter_for_memory: ",config.use_filter_for_memory)
                    memory4 = Memory(config.memory_size, config.filter_dimension_encoder[3])
                    memory3 = Memory(config.memory_size, config.filter_dimension_encoder[2])
                    memory2 = Memory(config.memory_size, config.filter_dimension_encoder[1])
                    memory1 = Memory(config.memory_size, config.filter_dimension_encoder[0])

                    x_endcoded_4_CLSTM, w4 = memory4(x_endcoded_4_CLSTM)
                    x_endcoded_3_CLSTM, w3 = memory3(x_endcoded_3_CLSTM)
                    x_endcoded_2_CLSTM, w2 = memory2(x_endcoded_2_CLSTM)
                    x_endcoded_1_CLSTM, w1 = memory1(x_endcoded_1_CLSTM)
                else:
                    print("Memory stores complete feature maps, config.use_filter_for_memory: ",
                          config.use_filter_for_memory)
                    memory4 = Memory2(config.memory_size, config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3])
                    memory3 = Memory2(config.memory_size, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2])
                    memory2 = Memory2(config.memory_size, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1])
                    memory1 = Memory2(config.memory_size, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0])

                    hidden_rep4 = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3],),
                                                          name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
                    x_endcoded_4_CLSTM_flatten, w4 = memory4(hidden_rep4)
                    x_endcoded_4_CLSTM = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_4_Mem")(
                        x_endcoded_4_CLSTM_flatten)

                    hidden_rep3 = tf.keras.layers.Reshape((config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2],),
                                                          name="Flatten_Conv3_HiddenRep")(x_endcoded_3_CLSTM)
                    x_endcoded_3_CLSTM_flatten, w3 = memory3(hidden_rep3)
                    x_endcoded_3_CLSTM = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_3_Mem")(
                        x_endcoded_3_CLSTM_flatten)

                    hidden_rep2 = tf.keras.layers.Reshape((config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1],),
                                                          name="Flatten_Conv2_HiddenRep")(x_endcoded_2_CLSTM)
                    x_endcoded_2_CLSTM_flatten, w2 = memory2(hidden_rep2)
                    x_endcoded_2_CLSTM = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_2_Mem")(
                        x_endcoded_2_CLSTM_flatten)

                    hidden_rep1 = tf.keras.layers.Reshape((config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0],),
                                                          name="Flatten_Conv1_HiddenRep")(x_endcoded_1_CLSTM)
                    x_endcoded_1_CLSTM_flatten, w1 = memory1(hidden_rep1)
                    x_endcoded_1_CLSTM = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_1_Mem")(
                        x_endcoded_1_CLSTM_flatten)

            if use_encoded_output:
                hidden_rep = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * 256,), name="Flatten_Conv4_HiddenRep_Sim")(x_endcoded_4_CLSTM)
            # Decoder
            # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)
            x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)

            con3 = tf.keras.layers.Concatenate(axis=3, name="Con3")([x_dedcoded_3, x_endcoded_3_CLSTM])
            x_dedcoded_2 = conv2d_trans_layer3(con3)

            con2 = tf.keras.layers.Concatenate(axis=3, name="Con2")([x_dedcoded_2, x_endcoded_2_CLSTM])
            x_dedcoded_1 = conv2d_trans_layer2(con2)

            con1 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_1, x_endcoded_1_CLSTM])
            x_dedcoded_0 = conv2d_trans_layer1(con1)
        else:
            # Reconstruction without using ConvLSTM and Attention
            x_endcoded_4_last_step = x_endcoded_4[:, config.step_max - 1]
            x_endcoded_3_last_step = x_endcoded_3[:, config.step_max - 1]
            x_endcoded_2_last_step = x_endcoded_2[:, config.step_max - 1]
            x_endcoded_1_last_step = x_endcoded_1[:, config.step_max - 1]

            if use_memory_restriction:
                print("Memory restriction is used on the encoder! (WO CONVLSTM Version)")

                if config.use_filter_for_memory:
                    print("Memory stores channels/filters, config.use_filter_for_memory: ",
                          config.use_filter_for_memory)
                    memory4 = Memory(config.memory_size, config.filter_dimension_encoder[3])
                    memory3 = Memory(config.memory_size, config.filter_dimension_encoder[2])
                    memory2 = Memory(config.memory_size, config.filter_dimension_encoder[1])
                    memory1 = Memory(config.memory_size, config.filter_dimension_encoder[0])

                    x_endcoded_4_last_step, w4 = memory4(x_endcoded_4_last_step)
                    x_endcoded_3_last_step, w3 = memory3(x_endcoded_3_last_step)
                    x_endcoded_2_last_step, w2 = memory2(x_endcoded_2_last_step)
                    x_endcoded_1_last_step, w1 = memory1(x_endcoded_1_last_step)
                else:
                    print("Memory stores complete feature maps, config.use_filter_for_memory: ",
                          config.use_filter_for_memory)

                    ### Memory ###
                    memory4 = Memory2(config.memory_size, config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3])
                    memory3 = Memory2(config.memory_size, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2])
                    memory2 = Memory2(config.memory_size, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1])
                    memory1 = Memory2(config.memory_size, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0])
                    hidden_rep4 = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3],),
                                                          name="Flatten_Conv4_HiddenRep")(x_endcoded_4_last_step)
                    x_endcoded_4_flatten, w4 = memory4(hidden_rep4)
                    x_endcoded_4_last_step = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_4_Mem")(
                        x_endcoded_4_flatten)

                    hidden_rep3 = tf.keras.layers.Reshape((config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2],),
                                                          name="Flatten_Conv3_HiddenRep")(x_endcoded_3_last_step)
                    x_endcoded_3_flatten, w3 = memory3(hidden_rep3)
                    x_endcoded_3_last_step = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_3_Mem")(
                        x_endcoded_3_flatten)

                    hidden_rep2 = tf.keras.layers.Reshape((config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1],),
                                                          name="Flatten_Conv2_HiddenRep")(x_endcoded_2_last_step)
                    x_endcoded_2_flatten, w2 = memory2(hidden_rep2)
                    x_endcoded_2_last_step = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_2_Mem")(
                        x_endcoded_2_flatten)

                    hidden_rep1 = tf.keras.layers.Reshape((config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0],),
                                                          name="Flatten_Conv1_HiddenRep")(x_endcoded_1_last_step)
                    x_endcoded_1_flatten, w1 = memory1(hidden_rep1)
                    x_endcoded_1_last_step = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_1_Mem")(
                        x_endcoded_1_flatten)

            if use_encoded_output:
                ### Regularized siamese neural network for unsupervised outlier detection on brain multiparametric magenetic
                # resonance imaging: application to epilepsy lesion screening
                hidden_rep = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * 256,), name="Flatten_Conv4_HiddenRep_Sim")(x_endcoded_4_last_step)
            # Decoder
            x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_last_step)
            con3 = tf.keras.layers.Concatenate(axis=3, name="Con3")([x_dedcoded_3, x_endcoded_3_last_step])
            x_dedcoded_2 = conv2d_trans_layer3(con3)
            con2 = tf.keras.layers.Concatenate(axis=3, name="Con2")([x_dedcoded_2, x_endcoded_2_last_step])
            x_dedcoded_1 = conv2d_trans_layer2(con2)
            con1 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_1, x_endcoded_1_last_step])
            x_dedcoded_0 = conv2d_trans_layer1(con1)

        if use_memory_restriction:
            if config.use_MemEntropyLoss:
                # for memory weight shrinkage:
                if config.use_filter_for_memory:
                    flatten = tf.keras.layers.Flatten()
                    w1 = flatten(w1)
                    w2 = flatten(w2)
                    w3 = flatten(w3)
                    w4 = flatten(w4)
                    w_hat_t = tf.keras.layers.Concatenate()([w1, w2, w3, w4])
                else:
                    w_hat_t = tf.keras.layers.Concatenate()([w1, w2, w3, w4])
                model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, w_hat_t])
            else:
                model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)
        elif use_encoded_output:
            model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        else:
            model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

# This layer should act as a memory that restricts reconstruction: https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf
#Source: https://github.com/YeongHyeon/MemAE-TF2/blob/78f374d2a63089713717276359c4896508bb4aeb/source/neuralnet.py
class Memory2(tf.keras.layers.Layer):
    # In contrast to original memory: here the complete encoded feature map is used as a memory slot
    def __init__(self, memory_size, input_size, **kwargs):
        super(Memory2, self).__init__()
        #self.memory_storage = None
        self.memory_size = memory_size
        self.input_size = input_size
        # self.num_outputs = num_outputs

    def build(self, input_shape):

        self.memory_storage = self.add_weight(name='memoryStorage',
                                              #shape=(100,16384),
                                              shape=(self.memory_size,self.input_size),
                                              initializer=tf.initializers.glorot_normal(),
                                              trainable=True)
        super(Memory2, self).build(input_shape)

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[0, 1, 3, 2]) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w
    def call(self, input_):
        tf.print("Memory Input shape: ", tf.shape(input_))
        tf.print("self.memory_storage: shape: ", tf.shape(self.memory_storage))
        #input_ = tf.keras.layers.Dropout(rate=0.3)(input_)
        num = tf.linalg.matmul(input_, tf.transpose(self.memory_storage), name='attention_num')
        denom = tf.linalg.matmul(input_ ** 2, tf.transpose(self.memory_storage) ** 2, name='attention_denum')

        tf.print("num shape: ", tf.shape(num))
        tf.print("denom shape: ", tf.shape(denom))
        
        w = (num + 1e-12) / (denom + 1e-12)

        # new distance start:
        '''
        input_new = tf.tile(input_, [1,30])
        input_new = tf.reshape(input_new,(tf.shape(input_)[0],30,tf.shape(input_)[1]))
        #tf.print("input_: ", tf.shape(input_new))
        memory_new = tf.expand_dims(self.memory_storage, 0)
        memory_new = tf.tile(memory_new, [tf.shape(input_)[0], 1, 1])
        #input_new = tf.reshape(memory_new, (tf.shape(input_)[0], 30, tf.shape(input_)[1]))
        abs_diff = tf.abs(input_new- memory_new)
        abs_diff = tf.reduce_mean(abs_diff, axis=2)
        #abs_diff =tf.norm(input_new - memory_new, ord='euclidean', axis=2)
        #tf.print("abs_diff: ", tf.shape(abs_diff))
        tf.print("abs_diff: ", abs_diff)
        w = tf.exp(-(abs_diff))
        #w = 1/(1+abs_diff*1000)#tf.exp(-(abs_diff)) #(num + 1e-12) / (denom + 1e-12)
        tf.print("w: ",w)
        '''
        # new distance end:


        '''
        #adding_noise = tf.keras.layers.GaussianNoise(stddev=0.3)
        adding_noise = tf.keras.layers.Dropout(rate=0.1)
        w = adding_noise(w)
        #values, indices = tf.math.top_k(w, 2)
        '''
        attentiton_w = tf.nn.softmax(w) # Eq.4

        tf.print("attentiton_w: shape: ", tf.shape(attentiton_w))
        tf.print("attentiton_w: ", attentiton_w)
        max_idx = tf.math.argmax(attentiton_w, axis=1, output_type=tf.dtypes.int64, name=None)
        tf.print("max_idx shape:",tf.shape(max_idx), "max_idx: ", max_idx)
        #tf.print("value: ",attentiton_w[:,max_idx])

        # Hard Shrinkage for Sparse Addressing
        lam = 1 / self.memory_size
        addr_num = tf.keras.activations.relu(attentiton_w - lam) * attentiton_w
        #tf.print("addr_num: ", addr_num)
        addr_denum = tf.abs(attentiton_w - lam) + 1e-12
        #tf.print("addr_denum: ", addr_denum)
        memory_addr = addr_num / addr_denum # Eq. 7
        #tf.print("memory_addr: ", memory_addr)

        # Set any values less 1e-12 or above  1-(1e-12) to these values
        w_hat = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))
        #w_hat = memory_addr / tf.linalg.normalize(memory_addr,ord=1,axis=0) # written in text after Eq. 7
        # Eq. 3:
        z_hat = tf.linalg.matmul(w_hat, self.memory_storage)
        #tf.print("z_hat shape: ", tf.shape(z_hat))
        #adding_noise = tf.keras.layers.Dropout(rate=0.1)
        #z_hat = adding_noise(z_hat)
        #
        #tf.print("z_hat shape ", z_hat.shape)
        #tf.print("w_hat shape ", w_hat.shape)
        #flatten = tf.keras.layers.Flatten()
        #w_hat = flatten(w_hat)
        #tf.print("w_hat shape flatten", w_hat.shape)
        return z_hat, w_hat

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'memory_size': self.memory_size,
            'input_size': self.input_size,
        })
        return config
    def compute_output_shape(self, input_shape):
        return (None, self.input_size)


class Memory(tf.keras.layers.Layer):
    def __init__(self, memory_size, input_size, **kwargs):
        super(Memory, self).__init__()
        # self.memory_storage = None
        self.memory_size = memory_size
        self.input_size = input_size
        # self.num_outputs = num_outputs

    def build(self, input_shape):
        self.memory_storage = self.add_weight(name='memoryStorage',
                                              # shape=(100,16384),
                                              shape=(1, 1, self.memory_size, self.input_size),
                                              initializer=tf.initializers.glorot_normal(),
                                              trainable=True)
        super(Memory, self).build(input_shape)

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[0, 1, 3, 2]) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w

    def call(self, input_):
        # tf.print("Memory Input shape: ", tf.shape(input_))
        '''
        num = tf.linalg.matmul(input_, tf.transpose(self.memory_storage), name='attention_num')
        denom = tf.linalg.matmul(input_ ** 2, tf.transpose(self.memory_storage) ** 2, name='attention_denum')
        '''
        num = tf.linalg.matmul(input_, tf.transpose(self.memory_storage, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(input_ ** 2, tf.transpose(self.memory_storage, perm=[0, 1, 3, 2]) ** 2,
                                 name='attention_denum')

        w = (num + 1e-12) / (denom + 1e-12)
        # tf.print("Cosine Sim Matrix w: ", tf.shape(w))
        '''
        #adding_noise = tf.keras.layers.GaussianNoise(stddev=0.3)
        adding_noise = tf.keras.layers.Dropout(rate=0.1)
        w = adding_noise(w)
        #values, indices = tf.math.top_k(w, 2)
        '''
        attentiton_w = tf.nn.softmax(w)  # Eq.4

        # tf.print("attentiton_w: ", attentiton_w)
        # tf.print("attentiton_w shape: ",tf.shape(attentiton_w))

        max_idx = tf.math.argmax(attentiton_w, axis=3, output_type=tf.dtypes.int64, name=None)
        # tf.print("max_idx shape:",tf.shape(max_idx), "max_idx: ", max_idx)

        # Hard Shrinkage for Sparse Addressing
        lam = 1 / self.memory_size
        addr_num = tf.keras.activations.relu(attentiton_w - lam) * attentiton_w
        # tf.print("addr_num: ", addr_num)
        addr_denum = tf.abs(attentiton_w - lam) + 1e-12
        # tf.print("addr_denum: ", addr_denum)
        memory_addr = addr_num / addr_denum  # Eq. 7
        # tf.print("memory_addr: ", memory_addr)

        # Set any values less 1e-12 or above  1-(1e-12) to these values
        w_hat = tf.clip_by_value(memory_addr, 1e-12, 1 - (1e-12))
        # w_hat = memory_addr / tf.linalg.normalize(memory_addr,ord=1,axis=0) # written in text after Eq. 7
        # Eq. 3:
        z_hat = tf.linalg.matmul(w_hat, self.memory_storage)
        # tf.print("z_hat shape: ", tf.shape(z_hat))
        # adding_noise = tf.keras.layers.Dropout(rate=0.1)
        # z_hat = adding_noise(z_hat)
        #
        # tf.print("z_hat shape ", z_hat.shape)
        # tf.print("w_hat shape ", w_hat.shape)
        # flatten = tf.keras.layers.Flatten()
        # w_hat = flatten(w_hat)
        # tf.print("w_hat shape flatten", w_hat.shape)
        return z_hat, w_hat

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'memory_size': self.memory_size,
            'input_size': self.input_size,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (None, self.input_size)

# For test purposes
'''
# This layer should act as a memory
class MemoryInstanceBased(tf.keras.layers.Layer):
    def __init__(self, memory_size, input_size, **kwargs):
        super(MemoryInstanceBased, self).__init__()
        #self.memory_storage = None
        self.memory_size = memory_size
        self.input_size = input_size
        # self.num_outputs = num_outputs

    def build(self, input_shape):

        self.memory_storage = self.add_weight(name='memoryStorage',
                                              #shape=(100,16384),
                                              shape=(self.memory_size,self.input_size),
                                              initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05,
                                                                                             seed=42),
                                              trainable=False)

        self.slot_access_counter = self.add_weight(name='slotAccessCounter',
                                              shape=(self.memory_size,self.input_size),
                                              initializer=tf.keras.initializers.Zeros(),
                                              trainable=False)
        self.slot_in_use_counter = self.add_weight(name='slotInUseCounter',
                                              shape=(self.memory_size,self.input_size),
                                              initializer=tf.keras.initializers.Zeros(),
                                              trainable=False)

        super(MemoryInstanceBased, self).build(input_shape)

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[0, 1, 3, 2]) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w
    def call(self, input_):
        #tf.print('inputs', tf.convert_to_tensor(input_))
        num = tf.linalg.matmul(input_, tf.transpose(self.memory_storage), name='attention_num')
        denom = tf.linalg.matmul(input_ ** 2, tf.transpose(self.memory_storage) ** 2, name='attention_denum')

        w = (num + 1e-12) / (denom + 1e-12)
        attentiton_w = tf.nn.softmax(w) # Eq.4

        tf.print("attentiton_w: ", attentiton_w)
        # Hard Shrinkage for Sparse Addressing
        lam = 1 / self.memory_size
        addr_num = tf.keras.activations.relu(attentiton_w - lam) * attentiton_w
        addr_denum = tf.abs(attentiton_w - lam) + 1e-12
        memory_addr = addr_num / addr_denum # Eq. 7

        # Calculate Access

        pos_2_replace = tf.keras.backend.argmax(memory_addr, axis=-1)

        one_hot_pos = tf.zeros(self.memory_size)
        one_hot_pos = tf.Variable(one_hot_pos)
        one = tf.ones(1)
        one_hot_pos.assign(tf.tensor_scatter_nd_update(one_hot_pos, pos_2_replace, one))
        #one_hot_pos = tf.expand_dims(one_hot_pos,axis=-1)
        #one_hot_pos = tf.reshape(one_hot_pos,[None,100])
        print("one_hot_pos shape ", one_hot_pos.shape)

        # Set any values less 1e-12 or above  1-(1e-12) to these values
        w_hat = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))
        # Only retrieve instance with the maximum value
        print("w_hat shape ", w_hat.shape)

        # Eq. 3:
        z_hat = tf.linalg.matmul(w_hat, self.memory_storage)
        #
        print("z_hat shape ", z_hat.shape)

        # Update memory

        def f1(memory_storage, pos_2_replace, input_): return tf.compat.v1.assign(memory_storage,
                                                                                  tf.tensor_scatter_nd_update(
                                                                                      memory_storage, pos_2_replace,
                                                                                      input_))

        def f2(memory_storage): return tf.compat.v1.assign(memory_storage, memory_storage)

        r = tf.cond(tf.keras.backend.max(memory_addr) < 0.9, lambda: f1(self.memory_storage, pos_2_replace, input_),
                    lambda: f2(self.memory_storage))

        return z_hat

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'memory_size': self.memory_size,
            'input_size': self.input_size,
        })
        return config
    def compute_output_shape(self, input_shape):
        return (None, self.input_size)
'''

class MSGCRED(tf.keras.Model):

    def __init__(self):

        super(MSGCRED, self).__init__()


    def create_model(self, guassian_noise_stddev = None, use_attention = True, use_ConvLSTM = True, use_memory_restriction=False, use_encoded_output=False, adj_mat=None):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(config.step_max, config.num_datastreams, config.num_datastreams, config.dim_of_dataset), batch_size=None, name="Input0")
        AdjMatInput = tf.keras.Input(shape=(3721,3721), name="Input1_AdjMat")
        #AdjMatInput = tf.ones((3721,3721))

        if guassian_noise_stddev is not None:
            adding_noise = tf.keras.layers.GaussianNoise(stddev=guassian_noise_stddev)
            adding_noise_dropout = tf.keras.layers.SpatialDropout3D(guassian_noise_stddev)
            a = adding_noise(signatureMatrixInput)
            signatureMatrixInput_ = adding_noise_dropout(a)
        else:
            signatureMatrixInput_ = signatureMatrixInput

        #output = spektral.layers.GCNConv(channels=channels, activation=None)([output, adj_matrix_input_ds])
        '''
        conv2d_layer1 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[0], strides=config.strides_encoder[0], kernel_size=config.kernel_size_encoder[0],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[0],
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[1], strides=config.strides_encoder[1], kernel_size=config.kernel_size_encoder[1],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[1],
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[2], strides=config.strides_encoder[2], kernel_size=config.kernel_size_encoder[2],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[2],
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[3], strides=config.strides_encoder[3], kernel_size=config.kernel_size_encoder[3],
                                               padding='same', activity_regularizer=tf.keras.regularizers.l1(config.l1Reg), dilation_rate=config.dilation_encoder[3],
                                               activation='selu')
        '''
        conv2d_layer1 = spektral.layers.GCNConv(channels=config.filter_dimension_encoder[0], activation='selu')
        conv2d_layer2 = spektral.layers.GCNConv(channels=config.filter_dimension_encoder[1], activation='selu')
        conv2d_layer3 = spektral.layers.GCNConv(channels=config.filter_dimension_encoder[2], activation='selu')
        conv2d_layer4 = spektral.layers.GCNConv(channels=config.filter_dimension_encoder[3], activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[0], strides=1, kernel_size=config.kernel_size_encoder[0], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[1], strides=1, kernel_size=config.kernel_size_encoder[1], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[2], strides=1, kernel_size=config.kernel_size_encoder[2], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[3], strides=1, kernel_size=config.kernel_size_encoder[3], padding='same',
                                                      return_sequences=use_attention, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[2], strides=config.strides_encoder[3],
                                                              kernel_size=config.kernel_size_encoder[3], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[1], strides=config.strides_encoder[2],
                                                              kernel_size=config.kernel_size_encoder[2], padding='same',
                                                              activation='selu')
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[0], strides=config.strides_encoder[1],
                                                              kernel_size=config.kernel_size_encoder[1], padding='same',
                                                              activation='selu')
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=config.strides_encoder[0],
                                                              kernel_size=config.kernel_size_encoder[0], padding='same',
                                                              activation='selu')

        reshapeToGCNFormart = tf.keras.layers.Reshape((5, 61 * 61, 8), name="ReshapeInput")(signatureMatrixInput_)
        '''
        #adj_mat = tf.keras.layers.Reshape((61,61),name="Reshapeadj_mat")(adj_mat)
        F = tf.keras.layers.Lambda(lambda x, i, j : tf.map_fn(conv2d_layer1,[x[:,0,:, :], AdjMatInput]),name="lambda_layer")  # Define your lambda layer
        #tf.keras.layers.Lambda(lambda x: tf.map_fn(conv2d_layer1, x))
        F.arguments = {'i': 0, 'j': 4}
        x_endcoded_1 =  F(reshapeToGCNFormart)
        '''
        #out = tf.keras.layers.Lambda(lambda x: tf.map_fn(conv2d_layer1, x))
        #reshapeToGCNFormart = reshapeToGCNFormart
        x_endcoded_1_0 = conv2d_layer1([reshapeToGCNFormart[:,0, :, :], AdjMatInput])
        x_endcoded_1_1 = conv2d_layer1([reshapeToGCNFormart[:,1, :, :], AdjMatInput])
        x_endcoded_1_2 = conv2d_layer1([reshapeToGCNFormart[:, 2, :, :], AdjMatInput])
        x_endcoded_1_3 = conv2d_layer1([reshapeToGCNFormart[:, 3, :, :], AdjMatInput])
        x_endcoded_1_4 = conv2d_layer1([reshapeToGCNFormart[:, 4, :, :], AdjMatInput])
        x_endcoded_1 = tf.keras.layers.Concatenate(axis=1, name="Concat_GCN_Output_1")([tf.expand_dims(x_endcoded_1_0,1),
                                                                                      tf.expand_dims(x_endcoded_1_1,1),
                                                                                      tf.expand_dims(x_endcoded_1_2,1),
                                                                                      tf.expand_dims(x_endcoded_1_3,1),
                                                                                      tf.expand_dims(x_endcoded_1_4,1)])

        x_endcoded_2_0 = conv2d_layer2([x_endcoded_1[:, 0, :, :], AdjMatInput])
        x_endcoded_2_1 = conv2d_layer2([x_endcoded_1[:, 1, :, :], AdjMatInput])
        x_endcoded_2_2 = conv2d_layer2([x_endcoded_1[:, 2, :, :], AdjMatInput])
        x_endcoded_2_3 = conv2d_layer2([x_endcoded_1[:, 3, :, :], AdjMatInput])
        x_endcoded_2_4 = conv2d_layer2([x_endcoded_1[:, 4, :, :], AdjMatInput])

        x_endcoded_2 = tf.keras.layers.Concatenate(axis=1, name="Concat_GCN_Output_2")([tf.expand_dims(x_endcoded_2_0,1),
                                                                                      tf.expand_dims(x_endcoded_2_1,1),
                                                                                      tf.expand_dims(x_endcoded_2_2,1),
                                                                                      tf.expand_dims(x_endcoded_2_3,1),
                                                                                      tf.expand_dims(x_endcoded_2_4,1)])
        x_endcoded_3_0 = conv2d_layer3([x_endcoded_2[:, 0, :, :], AdjMatInput])
        x_endcoded_3_1 = conv2d_layer3([x_endcoded_2[:, 1, :, :], AdjMatInput])
        x_endcoded_3_2 = conv2d_layer3([x_endcoded_2[:, 2, :, :], AdjMatInput])
        x_endcoded_3_3 = conv2d_layer3([x_endcoded_2[:, 3, :, :], AdjMatInput])
        x_endcoded_3_4 = conv2d_layer3([x_endcoded_2[:, 4, :, :], AdjMatInput])

        x_endcoded_3 = tf.keras.layers.Concatenate(axis=1, name="Concat_GCN_Output_3")([tf.expand_dims(x_endcoded_3_0,1),
                                                                                      tf.expand_dims(x_endcoded_3_1,1),
                                                                                      tf.expand_dims(x_endcoded_3_2,1),
                                                                                      tf.expand_dims(x_endcoded_3_3,1),
                                                                                      tf.expand_dims(x_endcoded_3_4,1)])
        x_endcoded_4_0 = conv2d_layer4([x_endcoded_3[:, 0, :, :], AdjMatInput])
        x_endcoded_4_1 = conv2d_layer4([x_endcoded_3[:, 1, :, :], AdjMatInput])
        x_endcoded_4_2 = conv2d_layer4([x_endcoded_3[:, 2, :, :], AdjMatInput])
        x_endcoded_4_3 = conv2d_layer4([x_endcoded_3[:, 3, :, :], AdjMatInput])
        x_endcoded_4_4 = conv2d_layer4([x_endcoded_3[:, 4, :, :], AdjMatInput])
        x_endcoded_4 = tf.keras.layers.Concatenate(axis=1, name="Concat_GCN_Output_4")([tf.expand_dims(x_endcoded_4_0,1),
                                                                                      tf.expand_dims(x_endcoded_4_1,1),
                                                                                      tf.expand_dims(x_endcoded_4_2,1),
                                                                                      tf.expand_dims(x_endcoded_4_3,1),
                                                                                      tf.expand_dims(x_endcoded_4_4,1)])

        x_endcoded_1 = tf.keras.layers.Reshape((5, 61, 61, 16), name="ReshapeInput1")(
            x_endcoded_1)
        x_endcoded_2 = tf.keras.layers.Reshape((5, 61, 61, 8), name="ReshapeInput2")(
            x_endcoded_2)
        x_endcoded_3 = tf.keras.layers.Reshape((5, 61, 61, 4), name="ReshapeInput3")(
            x_endcoded_3)
        x_endcoded_4 = tf.keras.layers.Reshape((5, 61, 61, 1), name="ReshapeInput4")(x_endcoded_4)
        #F = tf.keras.layers.Lambda(lambda x: tf.map_fn(conv2d_layer1, [x[0][:, 0, :, :], x[1]]), name="lambda_layer")  # Define your lambda layer
        # tf.keras.layers.Lambda(lambda x: tf.map_fn(conv2d_layer1, x))
        #F.arguments = {'i': 0, 'j': 4}
        #x_endcoded_1 = F([reshapeToGCNFormart,AdjMatInput])
        #x_endcoded_1 = conv2d_layer1([x_endcoded_1, AdjMatInput])
        #x_endcoded_2 = conv2d_layer2([x_endcoded_1, AdjMatInput])
        #x_endcoded_3 = conv2d_layer3([x_endcoded_2, AdjMatInput])
        #x_endcoded_4 = conv2d_layer4([x_endcoded_3, AdjMatInput])
        #x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")([reshapeToGCNFormart, AdjMatInput])#
        #x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")([x_endcoded_1, AdjMatInput])
        #x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")([x_endcoded_2, AdjMatInput])
        #x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")([x_endcoded_3, AdjMatInput])

        if use_ConvLSTM:
            x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
            x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
            x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
            x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

            if use_attention:
                if config.keras_attention_layer_instead_of_own_impl:

                    # Flatten to get (batchsize,Tq,dim)
                    x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[0]  * config.output_dim[0]  * config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4_Tq")(x_endcoded_4_CLSTM)
                    x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3_Tq")(x_endcoded_3_CLSTM)
                    x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2_Tq")(x_endcoded_2_CLSTM)
                    x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1_Tq")(x_endcoded_1_CLSTM)
                    # Flatten to get (batchsize,Tv,dim)
                    x_endcoded_4_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[0]  * config.output_dim[0]  * config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4_Tv")(x_endcoded_4_CLSTM_flatten[:, config.step_max - 1, :])
                    x_endcoded_3_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3_Tv")(x_endcoded_3_CLSTM_flatten[:, config.step_max - 1, :])
                    x_endcoded_2_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2_Tv")(x_endcoded_2_CLSTM_flatten[:, config.step_max - 1, :])
                    x_endcoded_1_CLSTM_flatten_Tv = tf.keras.layers.Reshape((1, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1_Tv")(x_endcoded_1_CLSTM_flatten[:, config.step_max - 1, :])

                    # Applying attention after reshaping the input into required format
                    x_endcoded_4_CLSTM = tf.keras.layers.Attention()([x_endcoded_4_CLSTM_flatten_Tv, x_endcoded_4_CLSTM_flatten])
                    x_endcoded_3_CLSTM = tf.keras.layers.Attention()([x_endcoded_3_CLSTM_flatten_Tv, x_endcoded_3_CLSTM_flatten])
                    x_endcoded_2_CLSTM = tf.keras.layers.Attention()([x_endcoded_2_CLSTM_flatten_Tv, x_endcoded_2_CLSTM_flatten])
                    x_endcoded_1_CLSTM = tf.keras.layers.Attention()([x_endcoded_1_CLSTM_flatten_Tv, x_endcoded_1_CLSTM_flatten])

                    # Reshape back again into original shape
                    x_endcoded_4_CLSTM = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM)
                    x_endcoded_3_CLSTM = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM)
                    x_endcoded_2_CLSTM = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM)
                    x_endcoded_1_CLSTM = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM)
                else:
                    ## Attention for x_endcoded_4_CLSTM
                    # Flatten to vector
                    x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[0]*config.output_dim[0]*config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
                    x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
                    #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
                    x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
                    #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
                    x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
                    x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[0]*config.output_dim[0]*config.filter_dimension_encoder[3], name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
                    x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
                    x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
                    x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
                    x_endcoded_4_CLSTM = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

                    ## Attention for x_endcoded_3_CLSTM
                    # Flatten to vector
                    x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[1]*config.output_dim[1]*config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
                    x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_3")(
                        x_endcoded_3_CLSTM_last_time_step)
                    x_endcoded_3_CLSTM_scores = tf.reduce_sum(
                        tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                                      keepdims=True), axis=2)
                    x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
                    x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2], name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
                    x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
                    x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
                    x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
                    x_endcoded_3_CLSTM = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

                    ## Attention for x_endcoded_2_CLSTM
                    # Flatten to vector
                    x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[2]*config.output_dim[2]*config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
                    x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
                    x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                                      keepdims=True), axis=2)
                    x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
                    x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1], name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
                    x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
                    x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
                    x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
                    x_endcoded_2_CLSTM = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

                    ## Attention for x_endcoded_1_CLSTM
                    # Flatten to vector
                    x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,config.output_dim[3]*config.output_dim[3]*config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
                    # x_endcoded_1_CLSTM_flatten: [?,5,16384]
                    x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
                    x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
                    x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                                      keepdims=True), axis=2)
                    x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
                    x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0], name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
                    x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
                    x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
                    x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
                    x_endcoded_1_CLSTM = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
            else:
                print("Use Attention: ",use_attention,", instead last ConvLSTM Sequence output is used for reconstruction")
            if use_memory_restriction:
                print("Memory restriction is used on the encoder!")
                ### Memory ###
                if config.use_filter_for_memory:
                    print("Memory stores channels/filters, config.use_filter_for_memory: ",config.use_filter_for_memory)
                    memory4 = Memory(config.memory_size, config.filter_dimension_encoder[3])
                    memory3 = Memory(config.memory_size, config.filter_dimension_encoder[2])
                    memory2 = Memory(config.memory_size, config.filter_dimension_encoder[1])
                    memory1 = Memory(config.memory_size, config.filter_dimension_encoder[0])

                    x_endcoded_4_CLSTM, w4 = memory4(x_endcoded_4_CLSTM)
                    x_endcoded_3_CLSTM, w3 = memory3(x_endcoded_3_CLSTM)
                    x_endcoded_2_CLSTM, w2 = memory2(x_endcoded_2_CLSTM)
                    x_endcoded_1_CLSTM, w1 = memory1(x_endcoded_1_CLSTM)
                else:
                    print("Memory stores complete feature maps, config.use_filter_for_memory: ",
                          config.use_filter_for_memory)
                    memory4 = Memory2(config.memory_size, config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3])
                    memory3 = Memory2(config.memory_size, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2])
                    memory2 = Memory2(config.memory_size, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1])
                    memory1 = Memory2(config.memory_size, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0])

                    hidden_rep4 = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3],),
                                                          name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
                    x_endcoded_4_CLSTM_flatten, w4 = memory4(hidden_rep4)
                    x_endcoded_4_CLSTM = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_4_Mem")(
                        x_endcoded_4_CLSTM_flatten)

                    hidden_rep3 = tf.keras.layers.Reshape((config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2],),
                                                          name="Flatten_Conv3_HiddenRep")(x_endcoded_3_CLSTM)
                    x_endcoded_3_CLSTM_flatten, w3 = memory3(hidden_rep3)
                    x_endcoded_3_CLSTM = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_3_Mem")(
                        x_endcoded_3_CLSTM_flatten)

                    hidden_rep2 = tf.keras.layers.Reshape((config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1],),
                                                          name="Flatten_Conv2_HiddenRep")(x_endcoded_2_CLSTM)
                    x_endcoded_2_CLSTM_flatten, w2 = memory2(hidden_rep2)
                    x_endcoded_2_CLSTM = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_2_Mem")(
                        x_endcoded_2_CLSTM_flatten)

                    hidden_rep1 = tf.keras.layers.Reshape((config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0],),
                                                          name="Flatten_Conv1_HiddenRep")(x_endcoded_1_CLSTM)
                    x_endcoded_1_CLSTM_flatten, w1 = memory1(hidden_rep1)
                    x_endcoded_1_CLSTM = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),
                                                                 name="Reshape_ToOrignal_ConvLSTM_1_Mem")(
                        x_endcoded_1_CLSTM_flatten)

            if use_encoded_output:
                hidden_rep = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * 256,), name="Flatten_Conv4_HiddenRep_Sim")(x_endcoded_4_CLSTM)
            # Decoder
            # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)
            x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)

            con3 = tf.keras.layers.Concatenate(axis=3, name="Con3")([x_dedcoded_3, x_endcoded_3_CLSTM])
            x_dedcoded_2 = conv2d_trans_layer3(con3)

            con2 = tf.keras.layers.Concatenate(axis=3, name="Con2")([x_dedcoded_2, x_endcoded_2_CLSTM])
            x_dedcoded_1 = conv2d_trans_layer2(con2)

            con1 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_1, x_endcoded_1_CLSTM])
            x_dedcoded_0 = conv2d_trans_layer1(con1)
        else:
            # Reconstruction without using ConvLSTM and Attention
            x_endcoded_4_last_step = x_endcoded_4[:, config.step_max - 1]
            x_endcoded_3_last_step = x_endcoded_3[:, config.step_max - 1]
            x_endcoded_2_last_step = x_endcoded_2[:, config.step_max - 1]
            x_endcoded_1_last_step = x_endcoded_1[:, config.step_max - 1]

            if use_memory_restriction:
                print("Memory restriction is used on the encoder! (WO CONVLSTM Version)")

                if config.use_filter_for_memory:
                    print("Memory stores channels/filters, config.use_filter_for_memory: ",
                          config.use_filter_for_memory)
                    memory4 = Memory(config.memory_size, config.filter_dimension_encoder[3])
                    memory3 = Memory(config.memory_size, config.filter_dimension_encoder[2])
                    memory2 = Memory(config.memory_size, config.filter_dimension_encoder[1])
                    memory1 = Memory(config.memory_size, config.filter_dimension_encoder[0])

                    x_endcoded_4_last_step, w4 = memory4(x_endcoded_4_last_step)
                    x_endcoded_3_last_step, w3 = memory3(x_endcoded_3_last_step)
                    x_endcoded_2_last_step, w2 = memory2(x_endcoded_2_last_step)
                    x_endcoded_1_last_step, w1 = memory1(x_endcoded_1_last_step)
                else:
                    print("Memory stores complete feature maps, config.use_filter_for_memory: ",
                          config.use_filter_for_memory)

                    ### Memory ###
                    memory4 = Memory2(config.memory_size, config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3])
                    memory3 = Memory2(config.memory_size, config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2])
                    memory2 = Memory2(config.memory_size, config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1])
                    memory1 = Memory2(config.memory_size, config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0])
                    hidden_rep4 = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * config.filter_dimension_encoder[3],),
                                                          name="Flatten_Conv4_HiddenRep")(x_endcoded_4_last_step)
                    x_endcoded_4_flatten, w4 = memory4(hidden_rep4)
                    x_endcoded_4_last_step = tf.keras.layers.Reshape((config.output_dim[0], config.output_dim[0], config.filter_dimension_encoder[3]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_4_Mem")(
                        x_endcoded_4_flatten)

                    hidden_rep3 = tf.keras.layers.Reshape((config.output_dim[1] * config.output_dim[1] * config.filter_dimension_encoder[2],),
                                                          name="Flatten_Conv3_HiddenRep")(x_endcoded_3_last_step)
                    x_endcoded_3_flatten, w3 = memory3(hidden_rep3)
                    x_endcoded_3_last_step = tf.keras.layers.Reshape((config.output_dim[1], config.output_dim[1], config.filter_dimension_encoder[2]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_3_Mem")(
                        x_endcoded_3_flatten)

                    hidden_rep2 = tf.keras.layers.Reshape((config.output_dim[2] * config.output_dim[2] * config.filter_dimension_encoder[1],),
                                                          name="Flatten_Conv2_HiddenRep")(x_endcoded_2_last_step)
                    x_endcoded_2_flatten, w2 = memory2(hidden_rep2)
                    x_endcoded_2_last_step = tf.keras.layers.Reshape((config.output_dim[2], config.output_dim[2], config.filter_dimension_encoder[1]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_2_Mem")(
                        x_endcoded_2_flatten)

                    hidden_rep1 = tf.keras.layers.Reshape((config.output_dim[3] * config.output_dim[3] * config.filter_dimension_encoder[0],),
                                                          name="Flatten_Conv1_HiddenRep")(x_endcoded_1_last_step)
                    x_endcoded_1_flatten, w1 = memory1(hidden_rep1)
                    x_endcoded_1_last_step = tf.keras.layers.Reshape((config.output_dim[3], config.output_dim[3], config.filter_dimension_encoder[0]),
                                                                     name="Reshape_ToOrignal_ConvLSTM_1_Mem")(
                        x_endcoded_1_flatten)

            if use_encoded_output:
                ### Regularized siamese neural network for unsupervised outlier detection on brain multiparametric magenetic
                # resonance imaging: application to epilepsy lesion screening
                hidden_rep = tf.keras.layers.Reshape((config.output_dim[0] * config.output_dim[0] * 256,), name="Flatten_Conv4_HiddenRep_Sim")(x_endcoded_4_last_step)
            # Decoder
            x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_last_step)
            con3 = tf.keras.layers.Concatenate(axis=3, name="Con3")([x_dedcoded_3, x_endcoded_3_last_step])
            x_dedcoded_2 = conv2d_trans_layer3(con3)
            con2 = tf.keras.layers.Concatenate(axis=3, name="Con2")([x_dedcoded_2, x_endcoded_2_last_step])
            x_dedcoded_1 = conv2d_trans_layer2(con2)
            con1 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_1, x_endcoded_1_last_step])
            x_dedcoded_0 = conv2d_trans_layer1(con1)

        if use_memory_restriction:
            if config.use_MemEntropyLoss:
                # for memory weight shrinkage:
                if config.use_filter_for_memory:
                    flatten = tf.keras.layers.Flatten()
                    w1 = flatten(w1)
                    w2 = flatten(w2)
                    w3 = flatten(w3)
                    w4 = flatten(w4)
                    w_hat_t = tf.keras.layers.Concatenate()([w1, w2, w3, w4])
                else:
                    w_hat_t = tf.keras.layers.Concatenate()([w1, w2, w3, w4])
                model = tf.keras.Model(inputs=[signatureMatrixInput,AdjMatInput], outputs=[x_dedcoded_0, w_hat_t])
            else:
                model = tf.keras.Model(inputs=[signatureMatrixInput,AdjMatInput], outputs=x_dedcoded_0)
        elif use_encoded_output:
            model = tf.keras.Model(inputs=[signatureMatrixInput,AdjMatInput], outputs=[x_dedcoded_0, hidden_rep])
        else:
            model = tf.keras.Model(inputs=[signatureMatrixInput,AdjMatInput], outputs=x_dedcoded_0)

        return model

class AttentionMSCRED(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(AttentionMSCRED, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
