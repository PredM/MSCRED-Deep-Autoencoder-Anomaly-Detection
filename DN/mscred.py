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


# This layer should act as a memory
class Memory(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Memory, self).__init__()
        self.memory_storage = None
        # self.num_outputs = num_outputs

    def build(self, input_shape):

        self.scaling_weight = self.add_weight(name='scalingWeight',
                                              shape=(1,),
                                              initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.05,
                                                                                             seed=42),
                                              trainable=True)
        super(Memory, self).build(input_shape)

    # noinspection PyMethodOverriding
    def call(self, input_):
        # Add weight
        weight_vector = tf.multiply(input_, self.scaling_weight)
        return weight_vector

    def compute_output_shape(self, input_shape):
        return input_shape
