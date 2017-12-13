#coding=utf-8
import tensorflow as tf
from utils import _norm, batch_convert2int
import config

class Generator(object):
    """docstring for Generator"""
    def __init__(self, name, is_train, filters, img_size):
        self.training = is_train
        self.name = name
        self.filters = filters
        self.img_size = img_size
        self.reuse = False

    def __call__(self, raw):
        with tf.variable_scope(self.name, reuse=self.reuse):
            g32 = self.generator_layer(raw, self.filters, 7, 1, self.training, config.GEN_FIRST_LAYER, 'g32') # (?, w, h, 32)
            g64 = self.generator_layer(g32, 2 * self.filters, 3, 2, self.training, config.GEN_OTHER_LAYER, 'g64') # (?, w/2, h/2, 64)
            g128 = self.generator_layer(g64, 4 * self.filters, 3, 2, self.training, config.GEN_OTHER_LAYER, 'g128') # (?, w/4, h/4, 128)
            
            if self.img_size <= 128:
                # use 6 residual blocks
                rb_layer = self.residual_blocks(g128, training=self.training, n=6) # (?, w/4, h/4, 128)
            else:
                # use 9 residual blocks
                rb_layer = self.residual_blocks(g128, training=self.training, n=9) # (?, w/4, h/4, 128)
            
            # fractional-strided convolution
            f64 = self.generator_layer(rb_layer, 2 * self.filters, 3, 2, self.training, config.GEN_FS_LAYER, 'f64')  # (?, w/2, h/2, 64)
            f32 = self.generator_layer(f64, self.filters, 3, 2, self.training, config.GEN_FS_LAYER, 'f32')  # (?, w, h, 32)
            output = self.generator_layer(f32, 3, 7, 1, training=self.training, layer=config.GEN_LAST_LAYER, name='output_g') # (?, w, h, 3)
        
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
     
        return output

   
    def generator_layer(self, raw, filters, out_channels, strides, training=True, layer=None, name=None):
        with tf.variable_scope(name):
            if layer == config.GEN_FIRST_LAYER or layer == config.GEN_LAST_LAYER:
                padded = tf.pad(raw, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
                # weights = self.get_variable('weights', [7, 7, raw.get_shape()[3], filters])
                raw = tf.layers.conv2d(padded, filters, out_channels, strides=strides, padding='VALID')
                # raw = tf.nn.conv2d(padded, weights, strides=[1, 1, 1, 1], padding='VALID')


            elif layer == config.GEN_OTHER_LAYER:
                raw = tf.layers.conv2d(raw, filters, out_channels, strides=strides, padding='SAME')
            elif layer == config.GEN_FS_LAYER:
                raw = tf.layers.conv2d_transpose(raw, filters, out_channels, strides=strides, padding='SAME')

            # at last layer , it may not need norm
            # raw = tf.contrib.layers.batch_norm(raw, is_training=training)
            raw = _norm(raw)
            
            if layer == config.GEN_LAST_LAYER:
                raw = tf.tanh(raw)
            else:
                raw = tf.nn.relu(raw)
            
            return raw

    def residual_blocks(self, raw, training, n=6):
        depth = raw.get_shape()[3]
        for i in range(n):
            output = self.rb_layer(raw, depth, training, 'RB_{}_{}'.format(depth, i))
            raw = output
        return output

    def rb_layer(self, raw, k, training=True, name=None):
        """
            residual blocks: 2 3 * 3 conv layers
        """
        with tf.variable_scope(name):
            # weights1 = self.get_variable('weights1', shape=[3, 3, input.get_shape()[3], k])
            with tf.variable_scope('res1'):
                padded1 = tf.pad(raw, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
                conv1 = tf.layers.conv2d(padded1, k, 3, strides=1, padding='VALID')
                normalized1 = _norm(conv1)
                # normalized1 = conv1
                relu1 = tf.nn.relu(normalized1)

            # weights2 = self.get_variable("weights2",shape=[3, 3, relu1.get_shape()[3], k])
            with tf.variable_scope('res2'):
                padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
                conv2 = tf.layers.conv2d(padded2, k, 3, strides=1, padding='VALID')
                normalized2 = _norm(conv2)
                # normalized2 = conv2
            output = raw+normalized2
            return output

    def get_variable(self, name, shape, mean=0.0, stddev=0.02):
        vars = tf.get_variable('weights',shape,
                    initializer=tf.random_normal_initializer(
                      mean=mean, stddev=stddev, dtype=tf.float32))
        return vars

    def sample(self, input):
        image = batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image

















            
