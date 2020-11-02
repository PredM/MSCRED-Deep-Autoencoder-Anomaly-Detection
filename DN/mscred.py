import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))
from configuration.Configuration import Configuration
config = Configuration()
import tensorflow.keras.backend as K


# Paper "A Deep Neural Network for Unsupervised AnomalyDetection and Diagnosis in Multivariate Time Series Data" by Chuxu Zhang,§∗Dongjin Song,†∗Yuncong Chen,†Xinyang Feng,‡∗Cristian Lumezanu,†Wei Cheng,†Jingchao Ni,†Bo Zong,†Haifeng Chen,†Nitesh V. Chawla
# Implementation of Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED) for Anomaly Detection and Diagnosis in Multivariate Time Series

class MSCRED(tf.keras.Model):

    def __init__(self):

        super(MSCRED, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(config.step_max, config.num_datastreams, config.num_datastreams, config.dim_of_dataset), batch_size=None, name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[0], strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[1], strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[2], strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[3], strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[0], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[1], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[2], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[3], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[2], strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[1], strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[0], strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')


        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,8*8*config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*config.filter_dimension_encoder[3], name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, config.filter_dimension_encoder[3]),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,16*16*config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * config.filter_dimension_encoder[2], name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, config.filter_dimension_encoder[2]),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,31*31*config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * config.filter_dimension_encoder[1], name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, config.filter_dimension_encoder[1]),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,61*61*config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * config.filter_dimension_encoder[0], name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, config.filter_dimension_encoder[0]),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_with_Memory(tf.keras.Model):

    def __init__(self):

        super(MSCRED_with_Memory, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, config.dim_of_dataset), batch_size=None, name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=64, strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=128, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=256, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=32, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=64, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=128, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=256, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

        ### Memory ###

        memory = Memory()
        x_endcoded_4_CLSTM_hidden = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Getting_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        hidden_rep = tf.keras.layers.Reshape(( 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM_hidden)
        x_endcoded_4_CLSTM_flatten = memory(hidden_rep)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)


        ### Attention ###
        ## Attention for x_endcoded_4_CLSTM
        '''
        # Flatten to vector
        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((5,8*8*256), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*256, name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        '''
        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((5,16*16*128), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * 128, name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((5,31*31*64), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * 64, name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((5,61*61*32), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * 32, name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        ### Memory ###
        '''
        memory4 = Memory()
        memory3 = Memory()
        memory2 = Memory()
        memory1 = Memory()

        x_endcoded_4_CLSTM_hidden = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Getting_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        hidden_rep = tf.keras.layers.Reshape(( 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM_hidden)
        x_endcoded_4_CLSTM_flatten = memory4(hidden_rep)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        '''

        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_with_LatentOutput(tf.keras.Model):

    def __init__(self):

        super(MSCRED_with_LatentOutput, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, config.dim_of_dataset), batch_size=config.batch_size, name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=64, strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=128, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=256, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=32, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=64, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=128, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=256, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

        ### Regularized siamese neural network for unsupervised outlier detection on brain multiparametric magenetic
        # resonance imaging: application to epilepsy lesion screening

        #hidden_rep = tf.keras.layers.Reshape((5 * 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4)

        x_endcoded_4_CLSTM_hidden = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Getting_Conv4_HiddenRep")(
            x_endcoded_4_CLSTM)
        hidden_rep = tf.keras.layers.Reshape(( 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM_hidden)
        '''
        indices_a = tf.range(config.batch_size)
        indices_a = tf.tile(indices_a, [config.batch_size])
        hidden_rep_a = tf.gather(hidden_rep, indices_a)
        # a shape: [T*T, C]
        indices_b = tf.range(config.batch_size)
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, config.batch_size])
        indices_b = tf.reshape(indices_b, [-1])
        hidden_rep_b = tf.gather(hidden_rep, indices_b)
        '''
        ###

        '''
        x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_4")(
            x_endcoded_4_CLSTM)
        x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM)
        x_endcoded_2_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_2")(
            x_endcoded_2_CLSTM)
        x_endcoded_1_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_1")(
            x_endcoded_1_CLSTM)
        '''

        ### Attention ###
        ## Attention for x_endcoded_4_CLSTM
        # Flatten to vector
        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((5,8*8*256), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*256, name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((5,16*16*128), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * 128, name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((5,31*31*64), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * 64, name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((5,61*61*32), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * 32, name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_woLSTM_woAttention(tf.keras.Model):

    def __init__(self):

        super(MSCRED_woLSTM_woAttention, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, 8), name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=64, strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=128, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=256, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=8, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        ### Regularized siamese neural network for unsupervised outlier detection on brain multiparametric magenetic
        # resonance imaging: application to epilepsy lesion screening
        #hidden_rep = tf.keras.layers.Reshape((5 * 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4)
        '''
        indices_a = tf.range(config.batch_size)
        indices_a = tf.tile(indices_a, [config.batch_size])
        hidden_rep_a = tf.gather(hidden_rep, indices_a)
        # a shape: [T*T, C]
        indices_b = tf.range(config.batch_size)
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, config.batch_size])
        indices_b = tf.reshape(indices_b, [-1])
        hidden_rep_b = tf.gather(hidden_rep, indices_b)
        '''
        ###
        '''
        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)
        '''
        '''
        x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_4")(
            x_endcoded_4_CLSTM)
        x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM)
        x_endcoded_2_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_2")(
            x_endcoded_2_CLSTM)
        x_endcoded_1_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_1")(
            x_endcoded_1_CLSTM)
        '''


        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4[:,4])
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3[:,4]])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2[:,4]])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1[:,4]])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_woAttention(tf.keras.Model):

    def __init__(self):

        super(MSCRED_woAttention, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, 8), name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=64, strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=128, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=256, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=32, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=False, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=64, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=False, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=128, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=False, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=256, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=False, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=8, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)


        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_w_Noise(tf.keras.Model):

    def __init__(self):

        super(MSCRED_w_Noise, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, config.dim_of_dataset), batch_size=None, name="Input0")

        adding_noise = tf.keras.layers.GaussianNoise(stddev=0.5)

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=64, strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=128, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=256, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=32, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=64, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=128, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=256, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        signatureMatrixInput_noisy = adding_noise(signatureMatrixInput)
        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput_noisy)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        ### Regularized siamese neural network for unsupervised outlier detection on brain multiparametric magenetic
        # resonance imaging: application to epilepsy lesion screening
        #hidden_rep = tf.keras.layers.Reshape((5 * 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4)
        '''
        indices_a = tf.range(config.batch_size)
        indices_a = tf.tile(indices_a, [config.batch_size])
        hidden_rep_a = tf.gather(hidden_rep, indices_a)
        # a shape: [T*T, C]
        indices_b = tf.range(config.batch_size)
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, config.batch_size])
        indices_b = tf.reshape(indices_b, [-1])
        hidden_rep_b = tf.gather(hidden_rep, indices_b)
        '''
        ###

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)
        '''
        x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_4")(
            x_endcoded_4_CLSTM)
        x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM)
        x_endcoded_2_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_2")(
            x_endcoded_2_CLSTM)
        x_endcoded_1_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_ConvLSTM_1")(
            x_endcoded_1_CLSTM)
        '''

        ### Attention ###
        ## Attention for x_endcoded_4_CLSTM
        # Flatten to vector
        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((5,8*8*256), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*256, name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((5,16*16*128), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * 128, name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((5,31*31*64), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * 64, name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((5,61*61*32), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * 32, name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_with_Memory2(tf.keras.Model):

    def __init__(self):

        super(MSCRED_with_Memory2, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, config.dim_of_dataset), batch_size=None, name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=32, strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=64, strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=128, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=256, strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=32, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=64, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=128, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=256, strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

        ### Memory ###
        '''
        memory = Memory()
        x_endcoded_4_CLSTM_hidden = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Getting_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        hidden_rep = tf.keras.layers.Reshape(( 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM_hidden)
        x_endcoded_4_CLSTM_flatten = memory(hidden_rep)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        '''

        ### Attention ###
        ## Attention for x_endcoded_4_CLSTM

        # Flatten to vector
        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((5,8*8*256), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*256, name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((5,16*16*128), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * 128, name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((5,31*31*64), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * 64, name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((5,61*61*32), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * 32, name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        ### Memory ###
        memory4 = Memory(100, 8 * 8 * 256)
        memory3 = Memory(100, 16 * 16 * 128)
        memory2 = Memory(100, 31 * 31 * 64)
        memory1 = Memory(100, 61 * 61 * 32)

        hidden_rep4 = tf.keras.layers.Reshape((8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        x_endcoded_4_CLSTM_flatten = memory4(hidden_rep4)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape((8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4_Mem")(x_endcoded_4_CLSTM_flatten)

        hidden_rep3 = tf.keras.layers.Reshape((16 * 16 * 128,), name="Flatten_Conv3_HiddenRep")(x_endcoded_3_CLSTM)
        x_endcoded_3_CLSTM_flatten = memory3(hidden_rep3)
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3_Mem")(x_endcoded_3_CLSTM_flatten)

        hidden_rep2 = tf.keras.layers.Reshape((31 * 31 * 64,), name="Flatten_Conv2_HiddenRep")(x_endcoded_2_CLSTM)
        x_endcoded_2_CLSTM_flatten = memory2(hidden_rep2)
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2_Mem")(x_endcoded_2_CLSTM_flatten)

        hidden_rep1 = tf.keras.layers.Reshape((61 * 61 * 32,), name="Flatten_Conv1_HiddenRep")(x_endcoded_1_CLSTM)
        x_endcoded_1_CLSTM_flatten = memory1(hidden_rep1)
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1_Mem")(x_endcoded_1_CLSTM_flatten)


        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model

class MSCRED_with_Memory2_Auto(tf.keras.Model):

    def __init__(self):

        super(MSCRED_with_Memory2_Auto, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, config.dim_of_dataset), batch_size=None, name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[0], strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[1], strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[2], strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[3], strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[0], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[1], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[2], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[3], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[2], strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[1], strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[0], strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

        ### Memory ###
        '''
        memory = Memory()
        x_endcoded_4_CLSTM_hidden = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Getting_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        hidden_rep = tf.keras.layers.Reshape(( 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM_hidden)
        x_endcoded_4_CLSTM_flatten = memory(hidden_rep)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        '''

        ### Attention ###
        ## Attention for x_endcoded_4_CLSTM

        # Flatten to vector
        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,8*8*config.filter_dimension_encoder[3]), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*config.filter_dimension_encoder[3], name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, config.filter_dimension_encoder[3]),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,16*16*config.filter_dimension_encoder[2]), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * config.filter_dimension_encoder[2], name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, config.filter_dimension_encoder[2]),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,31*31*config.filter_dimension_encoder[1]), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * config.filter_dimension_encoder[1], name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, config.filter_dimension_encoder[1]),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((config.step_max,61*61*config.filter_dimension_encoder[0]), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,config.step_max-1,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(config.step_max, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * config.filter_dimension_encoder[0], name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, config.filter_dimension_encoder[0]),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        ### Memory ###
        memory4 = Memory(100, 8 * 8 * config.filter_dimension_encoder[3])
        memory3 = Memory(100, 16 * 16 * config.filter_dimension_encoder[2])
        memory2 = Memory(100, 31 * 31 * config.filter_dimension_encoder[1])
        memory1 = Memory(100, 61 * 61 * config.filter_dimension_encoder[0])

        hidden_rep4 = tf.keras.layers.Reshape((8 * 8 * config.filter_dimension_encoder[3],), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        x_endcoded_4_CLSTM_flatten, w4 = memory4(hidden_rep4)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape((8, 8, config.filter_dimension_encoder[3]),name="Reshape_ToOrignal_ConvLSTM_4_Mem")(x_endcoded_4_CLSTM_flatten)

        hidden_rep3 = tf.keras.layers.Reshape((16 * 16 * config.filter_dimension_encoder[2],), name="Flatten_Conv3_HiddenRep")(x_endcoded_3_CLSTM)
        x_endcoded_3_CLSTM_flatten, w3 = memory3(hidden_rep3)
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, config.filter_dimension_encoder[2]),name="Reshape_ToOrignal_ConvLSTM_3_Mem")(x_endcoded_3_CLSTM_flatten)

        hidden_rep2 = tf.keras.layers.Reshape((31 * 31 * config.filter_dimension_encoder[1],), name="Flatten_Conv2_HiddenRep")(x_endcoded_2_CLSTM)
        x_endcoded_2_CLSTM_flatten, w2 = memory2(hidden_rep2)
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, config.filter_dimension_encoder[1]),name="Reshape_ToOrignal_ConvLSTM_2_Mem")(x_endcoded_2_CLSTM_flatten)

        hidden_rep1 = tf.keras.layers.Reshape((61 * 61 * config.filter_dimension_encoder[0],), name="Flatten_Conv1_HiddenRep")(x_endcoded_1_CLSTM)
        x_endcoded_1_CLSTM_flatten, w1 = memory1(hidden_rep1)
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, config.filter_dimension_encoder[0]),name="Reshape_ToOrignal_ConvLSTM_1_Mem")(x_endcoded_1_CLSTM_flatten)


        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])

        #for memory weight shrinkage:
        w_hat_t = tf.keras.layers.Concatenate()([w1, w2, w3, w4])

        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0,w_hat_t])

        return model

class MSCRED_with_Memory2_Auto_InstanceBased(tf.keras.Model):

    def __init__(self):

        super(MSCRED_with_Memory2_Auto_InstanceBased, self).__init__()


    def create_model(self):
        print("create model!")

        signatureMatrixInput = tf.keras.Input(shape=(5, 61, 61, config.dim_of_dataset), batch_size=None, name="Input0")

        conv2d_layer1 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[0], strides=[1, 1], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer2 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[1], strides=[2, 2], kernel_size=[3, 3],
                                               padding='same',
                                               activation='selu')
        conv2d_layer3 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[2], strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        conv2d_layer4 = tf.keras.layers.Conv2D(filters=config.filter_dimension_encoder[3], strides=[2, 2], kernel_size=[2, 2],
                                               padding='same',
                                               activation='selu')
        convLISTM_layer1 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[0], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[1], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[2], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=config.filter_dimension_encoder[3], strides=1, kernel_size=[2, 2], padding='same',
                                                      return_sequences=True, name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[2], strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[1], strides=[2, 2],
                                                              kernel_size=[2, 2], padding='same',
                                                              activation='selu', output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=config.filter_dimension_encoder[0], strides=[2, 2],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu', output_padding=0)
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=config.dim_of_dataset, strides=[1, 1],
                                                              kernel_size=[3, 3], padding='same',
                                                              activation='selu')

        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)
        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)
        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)

        ### Memory ###
        '''
        memory = Memory()
        x_endcoded_4_CLSTM_hidden = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Getting_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        hidden_rep = tf.keras.layers.Reshape(( 8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM_hidden)
        x_endcoded_4_CLSTM_flatten = memory(hidden_rep)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        '''

        ### Attention ###
        ## Attention for x_endcoded_4_CLSTM

        # Flatten to vector
        x_endcoded_4_CLSTM_flatten = tf.keras.layers.Reshape((5,8*8*256), name="Flatten_ConvLSTM_4")(x_endcoded_4_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)
        x_endcoded_4_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM_last_time_step)
        #x_endcoded_4_CLSTM_scores = tf.matmul(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step, transpose_b=True, name="Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM_flatten, x_endcoded_4_CLSTM_last_time_step_5) ,axis=2, keepdims=True),axis=2)
        #x_endcoded_4_CLSTM_scores = tf.squeeze(x_endcoded_4_CLSTM_scores, name="Squeeze_Scores_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention = tf.nn.softmax(x_endcoded_4_CLSTM_scores, name="Attention_ConvLSTM_4")
        x_endcoded_4_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(8*8*256, name="Repeated_Attention_ConvLSTM_4")(x_endcoded_4_CLSTM_attention)
        x_endcoded_4_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_4_CLSTM_attention_repeated, pattern=(0,2, 1))
        x_endcoded_4_CLSTM_flatten = tf.multiply(x_endcoded_4_CLSTM_attention_repeated_T, x_endcoded_4_CLSTM_flatten, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM_flatten = tf.reduce_sum(x_endcoded_4_CLSTM_flatten, axis= 1, name="Apply_Att_ConvLSTM_4")
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape(( 8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4")(x_endcoded_4_CLSTM_flatten)

        ## Attention for x_endcoded_3_CLSTM
        # Flatten to vector
        x_endcoded_3_CLSTM_flatten = tf.keras.layers.Reshape((5,16*16*128), name="Flatten_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)
        x_endcoded_3_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_3")(
            x_endcoded_3_CLSTM_last_time_step)
        x_endcoded_3_CLSTM_scores = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(x_endcoded_3_CLSTM_flatten, x_endcoded_3_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_3_CLSTM_attention = tf.nn.softmax(x_endcoded_3_CLSTM_scores, name="Attention_ConvLSTM_3")
        x_endcoded_3_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(16 * 16 * 128, name="Repeated_Attention_ConvLSTM_3")(x_endcoded_3_CLSTM_attention)
        x_endcoded_3_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_3_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_3_CLSTM_flatten = tf.multiply(x_endcoded_3_CLSTM_attention_repeated_T, x_endcoded_3_CLSTM_flatten,name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM_flatten = tf.reduce_sum(x_endcoded_3_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_3")
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3")(x_endcoded_3_CLSTM_flatten)

        ## Attention for x_endcoded_2_CLSTM
        # Flatten to vector
        x_endcoded_2_CLSTM_flatten = tf.keras.layers.Reshape((5,31*31*64), name="Flatten_ConvLSTM_2")(x_endcoded_2_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)
        x_endcoded_2_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_2")(x_endcoded_2_CLSTM_last_time_step)
        x_endcoded_2_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_2_CLSTM_flatten, x_endcoded_2_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_2_CLSTM_attention = tf.nn.softmax(x_endcoded_2_CLSTM_scores, name="Attention_ConvLSTM_2")
        x_endcoded_2_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(31 * 31 * 64, name="Repeated_Attention_ConvLSTM_2")(x_endcoded_2_CLSTM_attention)
        x_endcoded_2_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_2_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_2_CLSTM_flatten = tf.multiply(x_endcoded_2_CLSTM_attention_repeated_T, x_endcoded_2_CLSTM_flatten,name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM_flatten = tf.reduce_sum(x_endcoded_2_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_2")
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2")(x_endcoded_2_CLSTM_flatten)

        ## Attention for x_endcoded_1_CLSTM
        # Flatten to vector
        x_endcoded_1_CLSTM_flatten = tf.keras.layers.Reshape((5,61*61*32), name="Flatten_ConvLSTM_1")(x_endcoded_1_CLSTM)
        # x_endcoded_1_CLSTM_flatten: [?,5,16384]
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:], name="Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)
        x_endcoded_1_CLSTM_last_time_step_5 = tf.keras.layers.RepeatVector(5, name="Repeated_Last_TS_ConvLSTM_1")(x_endcoded_1_CLSTM_last_time_step)
        x_endcoded_1_CLSTM_scores = tf.reduce_sum(tf.reduce_sum(tf.multiply(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_last_time_step_5), axis=2,
                          keepdims=True), axis=2)
        x_endcoded_1_CLSTM_attention = tf.nn.softmax(x_endcoded_1_CLSTM_scores, name="Attention_ConvLSTM_1")
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * 32, name="Repeated_Attention_ConvLSTM_1")(x_endcoded_1_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)

        ### Memory ###
        memory4 = MemoryInstanceBased(100, 8 * 8 * 256)
        memory3 = MemoryInstanceBased(100, 16 * 16 * 128)
        memory2 = MemoryInstanceBased(100, 31 * 31 * 64)
        memory1 = MemoryInstanceBased(100, 61 * 61 * 32)

        hidden_rep4 = tf.keras.layers.Reshape((8 * 8 * 256,), name="Flatten_Conv4_HiddenRep")(x_endcoded_4_CLSTM)
        x_endcoded_4_CLSTM_flatten = memory4(hidden_rep4)
        x_endcoded_4_CLSTM = tf.keras.layers.Reshape((8, 8, 256),name="Reshape_ToOrignal_ConvLSTM_4_Mem")(x_endcoded_4_CLSTM_flatten)

        hidden_rep3 = tf.keras.layers.Reshape((16 * 16 * 128,), name="Flatten_Conv3_HiddenRep")(x_endcoded_3_CLSTM)
        x_endcoded_3_CLSTM_flatten = memory3(hidden_rep3)
        x_endcoded_3_CLSTM = tf.keras.layers.Reshape((16, 16, 128),name="Reshape_ToOrignal_ConvLSTM_3_Mem")(x_endcoded_3_CLSTM_flatten)

        hidden_rep2 = tf.keras.layers.Reshape((31 * 31 * 64,), name="Flatten_Conv2_HiddenRep")(x_endcoded_2_CLSTM)
        x_endcoded_2_CLSTM_flatten = memory2(hidden_rep2)
        x_endcoded_2_CLSTM = tf.keras.layers.Reshape((31, 31, 64),name="Reshape_ToOrignal_ConvLSTM_2_Mem")(x_endcoded_2_CLSTM_flatten)

        hidden_rep1 = tf.keras.layers.Reshape((61 * 61 * 32,), name="Flatten_Conv1_HiddenRep")(x_endcoded_1_CLSTM)
        x_endcoded_1_CLSTM_flatten = memory1(hidden_rep1)
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1_Mem")(x_endcoded_1_CLSTM_flatten)


        #x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)

        #x_endcoded_4_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_ConvLSTM_4")(x_endcoded_4_CLSTM)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :],  name="Last_TS_ConvLSTM_3")(x_endcoded_3_CLSTM)
        # Last_TS_ConvLSTM_4 (Lambda)  (None, 8, 8, 256)
        x_dedcoded_3 = conv2d_trans_layer4(x_endcoded_4_CLSTM)
        # DeConv4 (Conv2DTranspose)    (None, 16, 16, 128)

        #x_dedcoded_3 = tf.multiply(x_endcoded_3_CLSTM, x_dedcoded_3)
        #x_dedcoded_3 = tf.keras.layers.Lambda(lambda x: x[:, 4,:,:,:], name="Last_TS_DeConv_4")(
        #    x_dedcoded_3)
        #x_endcoded_3_CLSTM = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="Last_TS_DeConv_4")(
        #    x_endcoded_3_CLSTM)
        # Hinweis: wenn rekonstruktion über alle Zeitschritte also DeConv mit timeDistribute, dann axis=4 bei concat
        con2 = tf.keras.layers.Concatenate(axis=3, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        #x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)
        x_dedcoded_2 = conv2d_trans_layer3(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        #x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)
        x_dedcoded_1 = conv2d_trans_layer2(con2)

        con2 = tf.keras.layers.Concatenate(axis=3, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])

        #x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)
        x_dedcoded_0 = conv2d_trans_layer1(con2)

        #Return only the last time step, when time distribute decoding is active
        #x_dedcoded_0 = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_dedcoded_0)

        #model = tf.keras.Model(inputs=signatureMatrixInput, outputs=[x_dedcoded_0, hidden_rep])

        #for memory weight shrinkage:

        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        return model
# This layer should act as a memory
#Source: https://github.com/YeongHyeon/MemAE-TF2/blob/78f374d2a63089713717276359c4896508bb4aeb/source/neuralnet.py
class Memory(tf.keras.layers.Layer):
    def __init__(self, memory_size, input_size, **kwargs):
        super(Memory, self).__init__()
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
                                              trainable=True)
        super(Memory, self).build(input_shape)

    def cosine_sim(self, x1, x2):
        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom = tf.linalg.matmul(x1 ** 2, tf.transpose(x2, perm=[0, 1, 3, 2]) ** 2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w
    def call(self, input_):
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

        # Set any values less 1e-12 or above  1-(1e-12) to these values
        w_hat = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))
        #w_hat = memory_addr / tf.linalg.normalize(memory_addr,ord=1,axis=0) # written in text after Eq. 7
        # Eq. 3:
        z_hat = tf.linalg.matmul(w_hat, self.memory_storage)
        #
        print("z_hat shape ", z_hat.shape)

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
