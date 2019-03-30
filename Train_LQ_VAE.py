from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras.datasets import mnist


def lrelu(x , alpha = 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def de_conv(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def fully_connect(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))

    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:

      return tf.matmul(input_, matrix) + bias




class LQVAE(object):

    #build model
    def __init__(self, batch_size, max_iters, model_path, latent_dim, learnrate_init):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.saved_model_path = model_path
        self.latent_dim = latent_dim
        self.learn_rate_init = learnrate_init

        self.log_vars = []

        self.channel = 1
        self.output_size = 28

        self.x_input = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.x_true = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.ep1 = tf.random_normal(shape=[self.batch_size, self.latent_dim])

 
        print('Data Loading Begins')
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.X_Real_Train = X_train.reshape(-1,28,28,1)
        self.X_Real_Train = (self.X_Real_Train/255)*2 - 1
        print('Data Loading Completed')


    def build_model_lqvae(self):
        self.quant_thresh = 0.1
        self.z_mean, self.z_sigm = self.Encode1(self.x_input)
        self.z_x = tf.add(self.z_mean, self.quant_thresh*0.5*tf.sqrt(tf.exp(self.z_sigm))*self.ep1)
        self.z_softsign_x = tf.nn.softsign((self.quant_thresh*self.quant_thresh - tf.square(self.z_x))*500)
        self.x1 = self.generate1(self.z_softsign_x, reuse=False)

        self.z_hardsign_x = tf.sign(self.quant_thresh*self.quant_thresh -  tf.square(self.z_x))
        self.x2 = self.generate2(self.z_hardsign_x, reuse=False)


        self.z_filt_x = tf.sign(self.quant_thresh*self.quant_thresh -  tf.square(self.z_mean))
        self.x_filt = self.generate2(self.z_filt_x, reuse=True)


        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)/(self.latent_dim*self.batch_size)


        self.x1_mse = tf.reduce_mean(tf.square(tf.subtract(self.x1, self.x_true)))
        self.x2_mse = tf.reduce_mean(tf.square(tf.subtract(self.x2, self.x_true)))
        self.grad = tf.gradients(self.z_mean, self.x_input)[0]
        self.EncoderGradPenality = tf.reduce_mean(tf.square(0.1 - tf.sqrt(tf.reduce_sum(tf.square(self.grad),[1,2,3]))))



        #For encode
        self.E_loss = 1*self.kl_loss + 10*self.x1_mse + 10*self.EncoderGradPenality
        #for Gen
        self.G1_loss =  10*self.x1_mse 
        self.G2_loss =  10*self.x2_mse


        t_vars = tf.trainable_variables()

        self.log_vars.append(("encode_loss", self.E_loss))
        self.log_vars.append(("generator1_loss", self.G1_loss))
        self.log_vars.append(("generator2_loss", self.G2_loss))



        self.g1_vars = [var for var in t_vars if 'VAE_gen1' in var.name]
        self.e1_vars = [var for var in t_vars if 'VAE_e1_' in var.name]
        self.g2_vars = [var for var in t_vars if 'VAE_gen2' in var.name]
        self.cl_vars = [var for var in t_vars if 'sequential_1' in var.name]


        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)
        print('Model is Built')






    #do train
    def train(self):

        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=10000,
                                                   decay_rate=0.98)


        #for G1
        trainer_G1 = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_G1 = trainer_G1.compute_gradients(self.G1_loss, var_list=self.g1_vars)
        opti_G1 = trainer_G1.apply_gradients(gradients_G1)

        #for G2
        trainer_G2 = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_G2 = trainer_G2.compute_gradients(self.G2_loss, var_list=self.g2_vars)
        opti_G2 = trainer_G2.apply_gradients(gradients_G2)




        #for E1
        trainer_E1 = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_E1 = trainer_E1.compute_gradients(self.E_loss, var_list=self.e1_vars)
        opti_E1 = trainer_E1.apply_gradients(gradients_E1)




        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            print('******************')
            print(' ')
            print(' ')
            print('LQ-VAE Training Begins')
            print(' ')
            print(' ')
            print('******************')

            step = 0
            batchNum = 0
            NumBatches = np.int(50000/self.batch_size)
            while step <= self.max_iters:

                next_x_images = self.X_Real_Train[batchNum*self.batch_size:(batchNum+1)*self.batch_size]
                batchNum = (batchNum +1)%(NumBatches)
                new_learn_rate = sess.run(new_learning_rate)

                if new_learn_rate > 0.00005:
                    sess.run(add_global)
                fd ={self.x_input: next_x_images, self.x_true: next_x_images}

                
                sess.run(opti_E1, feed_dict=fd)
                sess.run(opti_G1, feed_dict=fd)
                sess.run(opti_G2, feed_dict=fd)

                
                if np.mod(step , 200) == 0 and step != 0:
                    self.saver.save(sess , self.saved_model_path)
                    k1_loss, enc_loss, x1_mse, x2_mse = sess.run([self.kl_loss , self.E_loss,self.x1_mse, self.x2_mse],feed_dict=fd)

                    print('step', step)
                    print('model saved: ', self.saved_model_path)
                    print('lr:', new_learn_rate)
                    print('KL_Loss: ', k1_loss)
                    print('Encoder Loss: ', enc_loss)
                    print('x1_mse: ', x1_mse)
                    print('x2_mse: ', x2_mse)


                step += 1


 

    def generate1(self, z_var, reuse=False):

        with tf.variable_scope('generator1') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = lrelu(fully_connect(z_var , output_size=1024, scope='VAE_gen1_fully1'))
            d2 = lrelu(fully_connect(d1 , output_size=128*7*7, scope='VAE_gen1_fully2'))
            d3 = tf.reshape(d2, [self.batch_size, 7, 7, 128])
            d4 = lrelu(de_conv(d3 , output_shape=[self.batch_size, 7, 7, 128],d_h=1, d_w=1, k_h=3, k_w=3, name='VAE_gen1_deconv2'))
            d5 = lrelu(de_conv(d4, output_shape=[self.batch_size, 14, 14, 128],  k_h=3, k_w=3,name='VAE_gen1_deconv3'))
            d6 = lrelu(de_conv(d5, output_shape=[self.batch_size, 28, 28, 64],  k_h=3, k_w=3,name='VAE_gen1_deconv4'))
            d7 = de_conv(d6, output_shape=[self.batch_size, 28, 28, 1] ,d_h=1, d_w=1,  k_h=3, k_w=3,name='VAE_gen1_deconv5')

            return tf.nn.tanh(d7)


    def generate2(self, z_var, reuse=False):

        with tf.variable_scope('generator2') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = lrelu(fully_connect(z_var , output_size=1024, scope='VAE_gen2_fully1'))
            d2 = lrelu(fully_connect(d1 , output_size=128*7*7, scope='VAE_gen2_fully2'))
            d3 = tf.reshape(d2, [self.batch_size, 7, 7, 128])
            d4 = lrelu(de_conv(d3 , output_shape=[self.batch_size, 7, 7, 128],d_h=1, d_w=1,  k_h=3, k_w=3,name='VAE_gen2_deconv2'))
            d5 = lrelu(de_conv(d4, output_shape=[self.batch_size, 14, 14, 128],  k_h=3, k_w=3,name='VAE_gen2_deconv3'))
            d6 = lrelu(de_conv(d5, output_shape=[self.batch_size, 28, 28, 64],  k_h=3, k_w=3,name='VAE_gen2_deconv4'))
            d7 = de_conv(d6, output_shape=[self.batch_size, 28, 28, 1] ,d_h=1, d_w=1,  k_h=3, k_w=3,name='VAE_gen2_deconv5')

            return tf.nn.tanh(d7)





    def Encode1(self, x, reuse=False):

        with tf.variable_scope('encode1') as scope:

            if reuse == True:
                scope.reuse_variables()
            conv1 = lrelu(conv2d(x, output_dim=64, d_h=1, d_w=1, k_h=3, k_w=3, name='VAE_e1_c1'))
            conv2 = lrelu(conv2d(conv1, output_dim=64,  k_h=3, k_w=3,name='VAE_e1_c2'))
            conv3 = lrelu(conv2d(conv2, output_dim=128,  k_h=3, k_w=3,name='VAE_e1_c3'))
            conv4 = lrelu(conv2d(conv3, output_dim=128, d_h=1, d_w=1,  k_h=3, k_w=3,name='VAE_e1_c4'))
            conv5 = tf.reshape(conv4, [self.batch_size, 128 * 7 * 7])
            fc1   = lrelu(fully_connect(conv5, output_size= 1024, scope='VAE_e1_f1'))
            z_mean  = fully_connect(fc1, output_size=self.latent_dim, scope='VAE_e1_f2')
            z_sigma = fully_connect(fc1, output_size=self.latent_dim, scope='VAE_e1_f3')

            return z_mean, z_sigma


    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))






model_path =  "./model/model.ckpt"
batch_size = 64
max_iters = 10000
latent_dim = 64
learn_rate_init = 0.0003

LQVAE = LQVAE(batch_size= batch_size, max_iters= max_iters, model_path= model_path, latent_dim= latent_dim,learnrate_init= learn_rate_init)

LQVAE.build_model_lqvae()

LQVAE.train()











