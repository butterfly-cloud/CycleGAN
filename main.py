#coding=utf-8
import tensorflow as tf 
from model import CycleGAN
import logging
import utils
import os
import re

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2e5, "how many step or epoch")
flags.DEFINE_float("learning_rate", 0.0002, "")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 1, "")
flags.DEFINE_integer("input_height", 108, "")
flags.DEFINE_integer("input_width", None, "")
flags.DEFINE_integer("output_height", 256,"")
flags.DEFINE_integer("output_width", 256, "")
flags.DEFINE_string("file_x", 'data/tfrecords/real2cartoon/real.tfrecords', "file x path")
flags.DEFINE_string("file_y", 'data/tfrecords/real2cartoon/cartoon.tfrecords', "file y path")
flags.DEFINE_integer('lambda1', 10.0, 'weight for forward cycle loss (X->Y->X), default: 10.0')
flags.DEFINE_integer('lambda2', 10.0, 'weight for backward cycle loss (Y->X->Y), default: 10.0')
flags.DEFINE_integer('filters', 64, 'number of gen filters in first conv layer, default: 64')
flags.DEFINE_bool('use_mse', True, 'mse loss or cross entropy loss')
flags.DEFINE_float('mse_label', 1.0, 'real y value in mse')
flags.DEFINE_integer('pool_size', 50, 'size of image buffer that stores previously generated images')

FLAGS = flags.FLAGS

def main(unused_argv):
    total_step = 0
    checkpoints_dir = './models/real2cartoon'
    summary_dir = './summary'



    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            batch_size=FLAGS.batch_size, 
            image_size=256, 
            use_mse=FLAGS.use_mse, 
            lambda1=FLAGS.lambda1, 
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            filters=FLAGS.filters,
            beta1=FLAGS.beta1,
            mse_label=FLAGS.mse_label,
            file_x=FLAGS.file_x,
            file_y=FLAGS.file_y
            )

        G_loss, F_loss, D_X_loss, D_Y_loss, fake_y, fake_x = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, F_loss, D_X_loss, D_Y_loss)

        summarys = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoints_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            total_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            logger.info('load model success' + ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            logger.info('start new model')

        # img_x = utils.get_img(FLAGS.file_x, FLAGS.output_height, FLAGS.output_width, FLAGS.batch_size)
        # img_y = utils.get_img(FLAGS.file_y, FLAGS.output_height, FLAGS.output_width, FLAGS.batch_size)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            fake_X_pool = utils.ImagePool(FLAGS.pool_size)
            fake_Y_pool = utils.ImagePool(FLAGS.pool_size)

            while not coord.should_stop():
                # img_x, img_y = read_file()
                fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
                        sess.run(
                                [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summarys],
                                feed_dict={cycle_gan.x: fake_X_pool.query(fake_x_val), 
                                            cycle_gan.y: fake_Y_pool.query(fake_y_val)}
                            )
                    )

                train_writer.add_summary(summary, total_step)
                train_writer.flush()

                logger.info('step: {}'.format(total_step))
                if total_step > 1e5:
                    sess.run(cycle_gan.learning_rate_decay_op())

                if total_step % 100 == 0:
                    logger.info('-----------Step %d:-------------' % total_step)
                    logger.info('  G_loss   : {}'.format(G_loss_val))
                    logger.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                    logger.info('  F_loss   : {}'.format(F_loss_val))
                    logger.info('  D_X_loss : {}'.format(D_X_loss_val))
                    logger.info('  learning_rate : {}'.format(cycle_gan.learning_rate))

                if total_step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=total_step)
                    logger.info("Model saved in file: %s" % save_path)

                total_step += 1
        except KeyboardInterrupt:
            logger.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=total_step)
            logger.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    logger = logging.getLogger('trainlogger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
            fmt='%(levelname)s\t%(asctime)s\t%(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S')
    handler = logging.FileHandler('./logs/train.log','a')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    tf.app.run()






