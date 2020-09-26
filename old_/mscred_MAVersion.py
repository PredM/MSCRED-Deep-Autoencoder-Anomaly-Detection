import tensorflow.compat.v1 as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# run able server
sys.path.append(os.path.abspath("."))
from configuration.Configuration import Configuration
config = Configuration()


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



    # run NN with data
    def call (self, inputs):
        tf.print(inputs)
        encodeds = self.encoder(inputs)
        lstmouts = self.lstmer(encodeds)
        reconstructed = self.decoder(lstmouts)
        rms = inputs - reconstructed
        return reconstructed, rms

