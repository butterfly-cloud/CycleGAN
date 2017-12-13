#coding=utf-8
import tensorflow as tf 
import utils
from discriminator import Discriminator
from generator import Generator
# from gen import Generator
# from dis import Discriminator
# from reader import Reader


class CycleGAN(object):
    """docstring for CycleGAN"""
    def __init__(self, 
                batch_size=1, 
                image_size=256, 
                use_mse=True, 
                lambda1=10, 
                lambda2=10,
                learning_rate=2e-4,
                filters=64,
                beta1=0.5,
                mse_label=1.0,
                file_x='',
                file_y=''):
        """
        batch_size: batch size, default 1 in paper
        use_mse: whether using mse loss or cross entropy
        lambda1: weight of cycle consistency loss(F(G(x)) -> x)
        lambda2: weight of cycle consistency loss(G(F(y)) -> y)
        learning_rate: learning rate
        filters: number of filter
        beta1: adam optimize params
        mse_label: using mse loss, true label value
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_mse = use_mse
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.file_y = file_y
        self.file_x = file_x
        
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        # self.learning_rate = learning_rate

        self.beta1 = beta1
        self.label = mse_label
        self.filters = filters

        self.training = tf.placeholder_with_default(True, shape=[], name='is_training')
        
        self.G = Generator('G', self.training, self.filters, image_size)
        self.F = Generator('F', self.training, self.filters, image_size)

        self.D_X = Discriminator('D_X', self.training, not self.use_mse)
        self.D_Y = Discriminator('D_Y', self.training, not self.use_mse)

        self.x = tf.placeholder(tf.float32,
            shape=[batch_size, image_size, image_size, 3])
        self.y = tf.placeholder(tf.float32,
            shape=[batch_size, image_size, image_size, 3])



    def model(self):
        # X_reader = Reader('data/tfrecords/man2woman/man.tfrecords', name='X',
        #     image_size=self.image_size, batch_size=self.batch_size)
        # Y_reader = Reader('data/tfrecords/man2woman/woman.tfrecords', name='Y',
        #     image_size=self.image_size, batch_size=self.batch_size)

        # x = X_reader.feed()
        # y = Y_reader.feed()
        x = utils.get_img(self.file_x, self.image_size, self.image_size, self.batch_size)
        y = utils.get_img(self.file_y, self.image_size, self.image_size, self.batch_size)

        fake_y = self.G(x)
        fake_x = self.F(y)

        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)
        G_gan_loss, F_gan_loss, D_loss_x, D_loss_y = self.gan_loss(self.D_Y, self.D_X, 
                            self.x, self.y, x, y, fake_y, fake_x, self.use_mse)
        # G_gan_loss = tf.reduce_mean(tf.squared_difference(self.D_Y(fake_y), self.label))
        # F_gan_loss = tf.reduce_mean(tf.squared_difference(self.D_X(fake_x), self.label))
        # D_loss_x = self.discriminator_loss(self.D_X, x, self.x)
        # D_loss_y = self.discriminator_loss(self.D_Y, y, self.y)


        
        G_loss = G_gan_loss + cycle_loss
        F_loss = F_gan_loss + cycle_loss

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(self.y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(self.x)))
        tf.summary.histogram('D_X/true', self.D_X(self.x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(self.y)))

        tf.summary.scalar('loss/G', G_loss)
        tf.summary.scalar('loss/D_Y', D_loss_y)
        tf.summary.scalar('loss/F', F_loss)
        tf.summary.scalar('loss/D_X', D_loss_x)
        tf.summary.scalar('loss/cycle', cycle_loss)

        tf.summary.image('X/generated', utils.batch_convert2int(self.G(self.x)))
        tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(self.x))))
        tf.summary.image('Y/generated', utils.batch_convert2int(self.F(self.y)))
        tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(self.y))))

        return G_loss, F_loss, D_loss_x, D_loss_y, fake_y, fake_x

    

    def optimize(self, G_loss, F_loss, D_loss_x, D_loss_y):
        """ Adam optimizer with learning rate 0.0002 for the first 100 epochs
            and a linearly decaying rate that goes to zero over the next 100 epochs
        """
        g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, name='g_opt') \
                    .minimize(G_loss, var_list=self.G.variables)
        d_y_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, name='d_y_opt') \
                    .minimize(D_loss_y, var_list=self.D_Y.variables)
        f_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, name='f_opt') \
                    .minimize(F_loss, var_list=self.F.variables)        
        d_x_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, name='d_x_opt') \
                    .minimize(D_loss_x, var_list=self.D_X.variables)

        
        with tf.control_dependencies([g_optimizer, d_y_optimizer, f_optimizer, d_x_optimizer]):
            return tf.no_op(name='optimizers')

    

    def cycle_consistency_loss(self, G, F, x, y):
        forward_loss = tf.reduce_mean(tf.abs(F(G(x) - x)))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y) - y)))
        cycle_loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return cycle_loss

    def gan_loss(self, D_Y, D_X, x, y, x_real, y_real, fake_y, fake_x, use_mse):
        
        D_real_loss_y = tf.reduce_mean(tf.squared_difference(D_Y(y_real), self.label))
        D_fake_loss_y = tf.reduce_mean(tf.square(D_Y(y)))

        D_real_loss_x = tf.reduce_mean(tf.squared_difference(D_X(x_real), self.label))
        D_fake_loss_x = tf.reduce_mean(tf.square(D_X(x)))

        G_loss = tf.reduce_mean(
            tf.squared_difference(D_Y(fake_y), self.label))
        F_loss = tf.reduce_mean(
            tf.squared_difference(D_X(fake_x), self.label))


        D_loss_y = (D_real_loss_y + D_fake_loss_y) / 2

        D_loss_x = (D_real_loss_x + D_fake_loss_x) / 2


        return G_loss, F_loss, D_loss_x, D_loss_y

    def learning_rate_decay_op(self, decay=0.999999999):
        return self.learning_rate.assign(self.learning_rate * decay)


    









        