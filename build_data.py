#coding=utf-8
#这个目前是针对非对称数据量来做处理的

import tensorflow as tf 
from glob import glob
import os

tf.flags.DEFINE_string('X_input_dir', 'test/real_man/b_resized/', 'file dir to use')
tf.flags.DEFINE_string('Y_input_dir', '/Users/yixin/sunhao25/myself/DCGAN-tensorflow/data/faces/', 'file dir to use')
tf.flags.DEFINE_string('X_output_file', 'data/tfrecords/real2cartoon/real.tfrecords', 'tf file dir to use')
tf.flags.DEFINE_string('Y_output_file', 'data/tfrecords/real2cartoon/cartoon.tfrecords', 'tf file dir to use')
FLAGS = tf.app.flags.FLAGS

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def parse_data(input_dir, output_file):
    """convert image to tfrecords
    Args:
        input_dir: input file path
        output_dir: output file path
    Return: None
    """
    images = glob(input_dir + '*.jpg')

    writer = tf.python_io.TFRecordWriter(output_file)

    img_num = len(images)

    for i in range(img_num):
        img = images[i]
        with tf.gfile.FastGFile(img, 'rb') as f:
            image_data = f.read()
            example = _convert_to_example(img, image_data)
            writer.write(example.SerializeToString())

        if i % 500 == 0:
            print "Processed {}/{}".format(i, img_num)
    print 'Done'
    writer.close()


def main(unused_argv):
    print 'parse data X'
    parse_data(FLAGS.X_input_dir, FLAGS.X_output_file)
    print 'parse data Y'
    parse_data(FLAGS.Y_input_dir, FLAGS.Y_output_file)

if __name__ == '__main__':
    tf.app.run()