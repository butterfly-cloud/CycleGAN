#coding=utf-8
import tensorflow as tf 
from utils import _norm

class Discriminator(object):
    def __init__(self, name, training, sigmod):
        self.name = name
        self.training = training
        self.sigmod = sigmod
        self.reuse = False

    def __call__(self, raw):
        """
        Args:
            input: batch_size, img_size, img_size, 3
        Returns:
            output: 4D tensor batch_size, output_size, output_size, 1
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            d64 = self.layer(raw, 64, training=self.training, name='d64') # (?, w/2, h/2, 64)
            d128 = self.layer(d64, 128, training=self.training, name='d128') # (?, w/4, h/4, 128)
            d256 = self.layer(d128, 256, training=self.training, name='d256') # (?, w/8, h/8, 256)
            d512 = self.layer(d256, 512, training=self.training, name='d512') # (?, w/16, h/16, 512)

            output = tf.layers.conv2d(d512, 1, 4, strides=1, padding='SAME')
            bias = tf.get_variable('biases', [1],
                        initializer=tf.constant_initializer(0.0))

            output = output + bias
            if self.sigmod:
                output = tf.sigmoid(output)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output


    def layer(self, raw, filters, slope=0.02, stride=2, training=True, name=None):
        """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
        Args:
            input: 4D tensor
            filters: integer, number of filters (output depth)
            slope: LeakyReLU's slope
            stride: integer
            training: boolean or BoolTensor
        Returns:
            4D tensor
        """
        with tf.variable_scope(name):
            layer = tf.layers.conv2d(raw, filters, 4, strides=stride, padding='SAME')
            layer = _norm(layer)
            output = self.leaky_relu(layer, slope)
        
            return output

    def leaky_relu(self, input, slope=0.02):
        return tf.maximum(slope*input, input)
