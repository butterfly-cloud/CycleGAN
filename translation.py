#coding=utf-8
import tensorflow as tf
import os
from model import CycleGAN
import utils

flags = tf.app.flags
flags.DEFINE_string('model', 'pretrained/man2woman.pb', 'model path .pb')
flags.DEFINE_string('input_img', '/Users/yixin/Desktop/sunhao/others/a_resized/000003.jpg', 'input img path')
flags.DEFINE_string('output_img', 'files/man2woman/006696.jpg', 'output img path')
flags.DEFINE_integer('img_size', 96, 'input img size path')

FLAGS = flags.FLAGS

def translate():
    graph = tf.Graph()

    with graph.as_default():
        with tf.gfile.FastGFile(FLAGS.input_img, 'rb') as f:
            img = f.read()
            input_img = tf.image.decode_jpeg(img, channels=3)
            input_img = tf.image.resize_images(input_img, size=(FLAGS.img_size, FLAGS.img_size))
            input_img = utils.convert2float(input_img)
            input_img.set_shape([FLAGS.img_size, FLAGS.img_size, 3])

        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model.read())

        [output_img] = tf.import_graph_def(graph_def, input_map={'input_image': input_img},
                                            return_elements=['output_image:0'],
                                            name='output')

    with tf.Session(graph=graph) as sess:
        generated = output_img.eval()
        with open(FLAGS.output_img, 'wb') as f:
            f.write(generated)

def main(unused_argv):
    translate()

if __name__ == '__main__':
    tf.app.run()