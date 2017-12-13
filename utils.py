#coding=utf-8
import tensorflow as tf
import config
import scipy.misc
import random

def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image/127.5) - 1.

def batch_convert2int(images):
    """
    Args:
      images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)

def generator_layer(raw, filters, out_channels, strides, training=True, layer=None, batch_size=64):
    if layer == config.GEN_FIRST_LAYER:
        padded = tf.pad(raw, [[[0,0], [3,3],[3,3],[0,0]]])
        tf.get_variable('weights', [7, 7, raw.get_shape()[3], filters],
                initializer=tf.random_normal_initializer(
                  mean=0.0, stddev=0.02, dtype=tf.float32))

        raw = tf.nn.conv2d(padded, weights, strides=[1, 1, 1, 1], padding='VALID')


    elif layer == config.GEN_OTHER_LAYER:
        raw = tf.layers.conv2d(raw, filters, out_channels, strides=strides, padding='SAME')
        # layer = tf.reshape(layer, [batch_size, int(raw.get_shape()[1]), int(raw.get_shape()[2]), int(raw.get_shape()[3])])

    raw = tf.contrib.layers.batch_norm(raw, is_training=training)
    
    if last_layer:
        raw = tf.tanh(raw)
    else:
        raw = tf.nn.relu(raw)
    

    return raw

def discrimator_layer(layer, filters, out_channels, strides, normalization=True, keep_prob=0.8):
    new_layer = tf.layers.conv2d(layer, filters, out_channels, strides, padding='SAME')
    if normalization:
        new_layer = tf.contrib.layers.batch_norm(new_layer, is_training=True)
    new_layer = leakyRelu(new_layer)
    new_layer = tf.nn.dropout(new_layer, keep_prob=keep_prob)
    return new_layer

def get_img(tfrecords_file, input_height, input_width, batch_size=1, num_threads=4,
    min_queue_examples=100):
    # file initial
  
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'image/file_name': tf.FixedLenFeature([], tf.string),
        'image/encoded_image': tf.FixedLenFeature([], tf.string),
      })

    image_buffer = features['image/encoded_image']
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    # image = self._preprocess(image)
    image = tf.image.resize_images(image, size=(input_height, input_width))
    image = convert2float(image)
    image.set_shape([input_height, input_width, 3])

    images = tf.train.shuffle_batch(
        [image], batch_size=batch_size, num_threads=num_threads,
        capacity=min_queue_examples + 3*batch_size,
        min_after_dequeue=min_queue_examples
      )

    return images


def _norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm", reuse=False):
    depth = input.get_shape()[3]
    scale = tf.get_variable("scale", [depth],initializer=tf.random_normal_initializer(
              mean=1.0, stddev=0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth],
              initializer=tf.constant_initializer(0.0))

    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.



class ImagePool(object):
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image
