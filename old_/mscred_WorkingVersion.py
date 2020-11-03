import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))
from configuration.Configuration import Configuration
config = Configuration()
import tensorflow.keras.backend as K


def shape_list(x):
    """Deal with dynamic shape in tensorflow by returning list of integers and tensor slices"""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

# encoder layer
def conv2D(nums_filter, strides, filter_size):
    layers = []
    for num_filter, stride, filter_size in zip(nums_filter, strides, filter_size):
        #print("num filters: ", str(num_filter) + ' Stride: ' +str(stride) + ' Filter size: ' +str(filter_size))
        layer = tf.keras.layers.Conv2D(filters=num_filter, strides=stride, kernel_size=filter_size, padding='same',
                                       activation='selu')
        layers.append(layer)
    return layers

# LSTM layer
def convLSTM2D(nums_filter):
    layers = []
    for num_filter in nums_filter:
        layer = tf.keras.layers.ConvLSTM2D(filters=num_filter, kernel_size=2, padding='same')
        layers.append(layer)
    return layers

# decoder layer
def conv2DTranspose(nums_filter, strides):
    layers = []
    for num_filter, stride in zip(nums_filter, strides):
        layer = tf.keras.layers.Conv2DTranspose(filters=num_filter, strides=stride, kernel_size=2, padding='same',
                                                  activation='selu')
        layers.append(layer)
    return layers


class MSCRED(tf.keras.Model):

    def __init__(self):

        super(MSCRED, self).__init__()

        # creation of endocer layers
        # input: filter numbers, strides, filter size
        self.enc_layers = conv2D(config.filter_dimension_encoder, config.stride_encoder , config.filter_size_encoder)
        def encoder(sigs):
            """
            :param sigs: shape ([batch, timesteps], number of sensors, number of Sensors, number of different time
            series length) :return:
            """
            outs = []
            prev = sigs
            for i, layer in enumerate(self.enc_layers):
                outs.append(layer(prev))
                prev = outs[-1]
            return outs

        def _reshape_batch_to_timestep(encoded):
            return tf.reshape(encoded, [1] + shape_list(encoded))
        reshape_batch_to_timestep = tf.keras.layers.Lambda(_reshape_batch_to_timestep)

        # creation of LSTM layers
        # input: filter numbers, strides, filter size
        self.lstm_layers = convLSTM2D(config.dimension_lstm)
        def attention(h_all):
            h_last = h_all[:, -1:, ...]
            similarities = h_all * h_last
            similarities = tf.reduce_sum(tf.reshape(similarities, (1, tf.shape(h_all)[1], -1)), axis=-1)
            similarities = tf.nn.softmax(similarities, axis=-1)
            similarities = tf.reshape(similarities, shape_list(similarities) + ([1, 1, 1]))
            weighted = h_all * similarities
            ret = tf.reduce_sum(weighted, axis=1)
            return ret
        def lstmer(encodeds):
            """
            :param encoded:  shape([
            :return:
            """
            return [attention(layer(reshape_batch_to_timestep(encoded))) for encoded, layer
                    in zip(encodeds, self.lstm_layers)]

        # creation of decoder layers
        # input: filter numbers, strides
        self.dec_layers = conv2DTranspose(config.filter_dimension_decoder, config.stride_decoder)
        concat = tf.keras.layers.Concatenate(axis=-1)
        def decoder(lstm_outs):
            x = None
            for layer, h in zip(self.dec_layers, lstm_outs[::-1]):
                if x is None:
                    x = layer(h)
                else:
                    x = x[:, :shape_list(h)[1], :shape_list(h)[2], :] # trim convTranspose
                    x = layer(concat([x,h]))
            return x

        self.encoder = encoder
        self.lstmer = lstmer
        self.decoder = decoder

    def create_model():
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
        x_endcoded_1_CLSTM_attention_repeated = tf.keras.layers.RepeatVector(61 * 61 * 32, name="Repeated_Attention_ConvLSTM_1")(x_endcoded_2_CLSTM_attention)
        x_endcoded_1_CLSTM_attention_repeated_T = tf.keras.backend.permute_dimensions(x_endcoded_1_CLSTM_attention_repeated, pattern=(0, 2, 1))
        x_endcoded_1_CLSTM_flatten = tf.multiply(x_endcoded_1_CLSTM_attention_repeated_T, x_endcoded_1_CLSTM_flatten,name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM_flatten = tf.reduce_sum(x_endcoded_1_CLSTM_flatten, axis=1, name="Apply_Att_ConvLSTM_1")
        x_endcoded_1_CLSTM = tf.keras.layers.Reshape((61, 61, 32),name="Reshape_ToOrignal_ConvLSTM_1")(x_endcoded_1_CLSTM_flatten)




        '''

        #x_endcoded_1_CLSTM = tf.keras.layers.dot([x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM_flatten[:, 4]],axes=[2, 2])
        #x_endcoded_1_CLSTM = tf.matmul(x_endcoded_1_CLSTM_flatten, x_endcoded_1_CLSTM, transpose_b=True, name="Scores_ConvLSTM_4")


        #1. get last time steps from ConvLSTM
        x_endcoded_1_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:,4,:,:,:])(x_endcoded_1_CLSTM)
        x_endcoded_1_CLSTM_last_time_step = tf.expand_dims(x_endcoded_1_CLSTM_last_time_step, axis=1)
        x_endcoded_2_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :])(x_endcoded_2_CLSTM)
        x_endcoded_2_CLSTM_last_time_step = tf.expand_dims(x_endcoded_2_CLSTM_last_time_step, axis=1)
        x_endcoded_3_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :])(x_endcoded_3_CLSTM)
        x_endcoded_3_CLSTM_last_time_step = tf.expand_dims(x_endcoded_3_CLSTM_last_time_step, axis=1)
        x_endcoded_4_CLSTM_last_time_step = tf.keras.layers.Lambda(lambda x: x[:, 4, :, :, :], name="LastTimeStepConvLSTM_4")(x_endcoded_4_CLSTM)
        x_endcoded_4_CLSTM_last_time_step = tf.expand_dims(x_endcoded_4_CLSTM_last_time_step, axis=1, name="LastTimeStepConvLSTM_4")

        ### attention nachprogrammiert, keras layer ausgeklammert, geht aber beides###
        print("x_endcoded_4_CLSTM", tf.shape(x_endcoded_4_CLSTM))
        #scores_4 = tf.matmul(x_endcoded_4_CLSTM_last_time_step, x_endcoded_4_CLSTM[:,4], transpose_b=True, name="Scores_ConvLSTM_4")
        # Test. scores_4 = tf.multiply(x_endcoded_4_CLSTM_last_time_step, x_endcoded_4_CLSTM,name="Scores_ConvLSTM_4")
        #scores_4 = tf.reduce_sum(scores_4, name="Scores_Sum_ConvLSTM_4")
        #scores_rescaled_4 = tf.divide(scores_4, 5)
        att_0 = tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[:,0], x_endcoded_4_CLSTM[:,-1])) / 5
        att_1 = tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[:, 1], x_endcoded_4_CLSTM[:, -1])) / 5
        att_2 = tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[:, 2], x_endcoded_4_CLSTM[:, -1])) / 5
        att_3 = tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[:, 3], x_endcoded_4_CLSTM[:, -1])) / 5
        att_4 = tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[:, 4], x_endcoded_4_CLSTM[:, -1])) / 5

        #x_endcoded_4_CLSTM[:, 0] = tf.multiply(x_endcoded_4_CLSTM[:, 0], att_0)

        attention_w = []
        for k in range(5):
            attention_w.append(
                tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[:,k], x_endcoded_4_CLSTM[:,-1])) / 5)
        attention_4 = tf.nn.softmax(attention_w)
        #attention_4 = tf.nn.softmax(tf.stack(scores_rescaled_4), name="Scores_to_Attention_ConvLSTM_4")
        #h = tf.matmul(scores_rescaled_4, x_endcoded_4_CLSTM, name="Compute_h_ConvLSTM_4")
        h = tf.multiply(x_endcoded_4_CLSTM, att_0)
        x_endcoded_4_CLSTM = h
        '''

        # attention based on inner-product between feature representation of last step and other steps

        '''
        step_max =5
        attention_w = []
        for k in range(step_max):
            attention_w.append(tf.reduce_sum(tf.multiply(x_endcoded_4_CLSTM[0][k], x_endcoded_4_CLSTM[0][-1])) / step_max)
        attention_w = tf.nn.softmax(tf.stack(attention_w))
        #attention_w = tf.keras.layers.Reshape(attention_w,(-1, 5))
        shape = K.shape(x_endcoded_4_CLSTM)

        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [shape[0], step_max])
        h = tf.matmul(attention_w, x_endcoded_4_CLSTM, name="Compute_h_ConvLSTM_4")
        x_endcoded_4_CLSTM = h
        '''
        #outputs = tf.reshape(x_endcoded_4_CLSTM[0], [step_max, -1])
        #x_endcoded_4_CLSTM = tf.matmul(attention_w, outputs)
        #outputs = tf.reshape(outputs,
        #                     [1, int(math.ceil(float(sensor_n) / 8)), int(math.ceil(float(sensor_n) / 8)), 256])
        '''
        scores = tf.matmul(x_endcoded_1_CLSTM_last_time_step, x_endcoded_1_CLSTM, transpose_b=True)
        scores_rescaled = tf.divide(scores,5)
        attention = tf.nn.softmax(scores_rescaled)
        h = tf.matmul(attention, x_endcoded_1_CLSTM)
        x_endcoded_1_CLSTM = h
        '''

        '''
        x_endcoded_1_CLSTM = tf.keras.layers.Attention()(
            [x_endcoded_1_CLSTM, x_endcoded_1_CLSTM_last_time_step])
        
        x_endcoded_2_CLSTM = tf.keras.layers.Attention()(
            [x_endcoded_2_CLSTM, x_endcoded_2_CLSTM_last_time_step])
        x_endcoded_3_CLSTM = tf.keras.layers.Attention()(
            [x_endcoded_3_CLSTM, x_endcoded_3_CLSTM_last_time_step])
        
        x_endcoded_4_CLSTM = tf.keras.layers.Attention()(
            [x_endcoded_4_CLSTM, x_endcoded_4_CLSTM_last_time_step])
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

        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        # tf.keras.utils.plot_model(model, to_file='mscred.png', show_shapes=True)
        # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        '''
        ### MSCRED START:
        signatureMatrixInput = tf.keras.Input(shape=(5,30,30,3), name="Input0")

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
                                                     return_sequences=True,name="ConvLSTM1")
        convLISTM_layer2 = tf.keras.layers.ConvLSTM2D(filters=64, strides=1, kernel_size=[2, 2], padding='same',
                                                     return_sequences=True,name="ConvLSTM2")

        convLISTM_layer3 = tf.keras.layers.ConvLSTM2D(filters=128, strides=1, kernel_size=[2, 2], padding='same',
                                                     return_sequences=True,name="ConvLSTM3")

        convLISTM_layer4 = tf.keras.layers.ConvLSTM2D(filters=256, strides=1, kernel_size=[2, 2], padding='same',
                                                     return_sequences=True,name="ConvLSTM4")

        conv2d_trans_layer4 = tf.keras.layers.Conv2DTranspose(filters=128, strides=[2, 2],
                                                             kernel_size=[2, 2], padding='same',
                                                             activation='selu', name="DeConv4")
        conv2d_trans_layer3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=[2, 2],
                                                             kernel_size=[2, 2], padding='same',
                                                             activation='selu',output_padding=1)
        conv2d_trans_layer2 = tf.keras.layers.Conv2DTranspose(filters=32, strides=[2, 2],
                                                             kernel_size=[3, 3], padding='same',
                                                             activation='selu')
        conv2d_trans_layer1 = tf.keras.layers.Conv2DTranspose(filters=3, strides=[1, 1],
                                                             kernel_size=[3, 3], padding='same',
                                                             activation='selu')


        x_endcoded_1 = tf.keras.layers.TimeDistributed(conv2d_layer1, name="Conv1")(signatureMatrixInput)
        x_endcoded_2 = tf.keras.layers.TimeDistributed(conv2d_layer2, name="Conv2")(x_endcoded_1)
        x_endcoded_3 = tf.keras.layers.TimeDistributed(conv2d_layer3, name="Conv3")(x_endcoded_2)
        x_endcoded_4 = tf.keras.layers.TimeDistributed(conv2d_layer4, name="Conv4")(x_endcoded_3)

        x_endcoded_1_CLSTM = convLISTM_layer1(x_endcoded_1)

        x_endcoded_2_CLSTM = convLISTM_layer2(x_endcoded_2)


        x_endcoded_4_CLSTM = convLISTM_layer4(x_endcoded_4)
        x_endcoded_3_CLSTM = convLISTM_layer3(x_endcoded_3)

        x_dedcoded_3 = tf.keras.layers.TimeDistributed(conv2d_trans_layer4, name="DeConv4")(x_endcoded_4_CLSTM)
        con2 = tf.keras.layers.Concatenate(axis=4, name="Con4")([x_dedcoded_3, x_endcoded_3_CLSTM])
        x_dedcoded_2 = tf.keras.layers.TimeDistributed(conv2d_trans_layer3, name="DeConv3")(con2)

        con2 = tf.keras.layers.Concatenate(axis=4, name="Con1")([x_dedcoded_2, x_endcoded_2_CLSTM])
        x_dedcoded_1 = tf.keras.layers.TimeDistributed(conv2d_trans_layer2, name="DeConv2")(con2)

        con2 = tf.keras.layers.Concatenate(axis=4, name="Con0")([x_dedcoded_1, x_endcoded_1_CLSTM])
        x_dedcoded_0 = tf.keras.layers.TimeDistributed(conv2d_trans_layer1, name="DeConv1")(con2)

        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=x_dedcoded_0)

        #tf.keras.utils.plot_model(model, to_file='mscred.png', show_shapes=True)
        #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        ### MSCRED STOP
        '''

        '''
        # CONV2D Encoder
        cnn2d_enc_outputs = []
        convLSTM_enc_outputs = []
        cnn2dTran_enc_outputs = []
        i=0
        for num_filter, stride, filter_size, num_filter_decoder, stride_decoder in zip(config.filter_dimension_encoder, config.stride_encoder, config.filter_size_encoder, config.filter_dimension_decoder, config.stride_decoder):
            print("Conv: ",i,"num filters: ", str(num_filter) + ' Stride: ' +str(stride) + ' Filter size: ' +str(filter_size))
            print("LSTM: ",i,"num filters: ", str(num_filter) + ' Stride: ' +str(stride) + ' Filter size: ' +str(filter_size))

            conv2d_layer = tf.keras.layers.Conv2D(filters=num_filter, strides=stride, kernel_size=filter_size, padding='same',
                                           activation='selu')
            convLISTM_layer = tf.keras.layers.ConvLSTM2D(filters=num_filter, strides=1, kernel_size=[2, 2], padding='same', return_sequences=True)

            #conv2d_trans_layer = tf.keras.layers.Conv2DTranspose(filters=num_filter_decoder, strides=stride_decoder, kernel_size=2, padding='same',
            #                                        activation='selu')
            if i == 0:
                x = tf.keras.layers.TimeDistributed(conv2d_layer)(signatureMatrixInput)
                cnn2d_enc_outputs.append(x)
                y = convLISTM_layer(cnn2d_enc_outputs[i])
                convLSTM_enc_outputs.append(y)

            else:
                x = tf.keras.layers.TimeDistributed(conv2d_layer)(x)
                cnn2d_enc_outputs.append(x)
                y1 = convLISTM_layer(cnn2d_enc_outputs[i])
                convLSTM_enc_outputs.append(y1)
            i = i +1

        # Build decoder
        d = 0
        for num_filter_decoder, stride_decoder, filter_size_decoder in zip(config.filter_dimension_decoder, config.stride_decoder, config.filter_size_decoder):
            print("Deconv: ",d," Filter: ", num_filter_decoder, " Strides: ", stride_decoder, " Filter size: ", filter_size_decoder)
            conv2d_trans_layer = tf.keras.layers.Conv2DTranspose(filters=num_filter_decoder, strides=stride_decoder,
                                                                 kernel_size=filter_size_decoder, padding='same',
                                                                 activation='selu')
            #out = tf.keras.layers.Lambda(lambda y1: y1[9, :,:,:])(y1)
            if d == 0:
                print("d==0")
                z = tf.keras.layers.TimeDistributed(conv2d_trans_layer)(y1)
                #z = conv2d_trans_layer(out)
            if d == 2:
                z = tf.keras.layers.TimeDistributed(conv2d_trans_layer)(o)
            else:
                con = tf.keras.layers.Concatenate(axis=4)([cnn2d_enc_outputs[0],z])
                o = tf.keras.layers.TimeDistributed(conv2d_trans_layer)(con)
                #z = conv2d_trans_layer([convLSTM_enc_outputs[d],cnn2d_enc_outputs[d]])
            d = d + 1


        # CONV LSTM Layers
        for num_filter in config.dimension_lstm:
            convLISTM_layer = tf.keras.layers.ConvLSTM2D(filters=num_filter, kernel_size=2, padding='same')

        for num_filter, stride in zip(config.filter_dimension_decoder, config.stride_decoder):
            layer = tf.keras.layers.Conv2DTranspose(filters=num_filter, strides=1, kernel_size=[2,2], padding='same',
                                                    activation='selu')
        model = tf.keras.Model(inputs=signatureMatrixInput, outputs=z)
        '''
        return model

    # run NN with data
    def call (self, inputs):
        tf.print(inputs)
        encodeds = self.encoder(inputs)
        lstmouts = self.lstmer(encodeds)
        reconstructed = self.decoder(lstmouts)
        rms = inputs - reconstructed
        return reconstructed, rms

