
from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
np.random.seed(2018)
from scipy.io import savemat
from ops import *
from scipy.io import loadmat
from vgg_face import vgg_face
import pickle
from time import gmtime, strftime


class ExprGAN(object):
    def __init__(self,
                 session,
                 size_image=128,
                 size_kernel=5,
                 size_batch=64,
                 num_encoder_channels=64,
                 num_z_channels=50,
                 y_dim=7,
                 num_gen_channels=1024,
                 enable_tile_label=True,
                 tile_ratio=1.0,
                 is_training=True,
                 save_dir='./save',
                 dataset_name='OULU',
                 is_flip=True,
                 checkpoint_dir='./checkpoint',
                 content_layer='relu4_2',
                 is_stage_one=True,
                 rb_dim=3,
                 vgg_coeff=1,
                 q_coeff=1,
                 fm_coeff=1,
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.label_value_range = (0, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.y_dim = y_dim
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.is_flip = is_flip
        self.CONTENT_LAYER = content_layer
        self.vgg_weights = loadmat('utils/vgg-face.mat')
        self.checkpoint_dir = checkpoint_dir
        self.is_stage_one = is_stage_one
        self.rb_dim = rb_dim
        self.vgg_coeff = vgg_coeff
        self.q_coeff = q_coeff
        self.fm_coeff = fm_coeff
        if self.is_stage_one:
            self.conv1_2_coeff = 1
            self.conv2_2_coeff = 1
            self.conv3_2_coeff = 1
            self.conv4_2_coeff = 1
            self.conv5_2_coeff = 1
        else:
            self.conv1_2_coeff = 1
            self.conv2_2_coeff = 1
            self.conv3_2_coeff = 1
            self.conv4_2_coeff = 1
            self.conv5_2_coeff = 1


        print "\n\tLoading data"
        self.data_X, self.data_y = self.load_anno('../split/' + self.dataset_name.lower() + '_anno.pickle')
        self.data_X = [os.path.join("../data", self.dataset_name, x) for x in self.data_X]

        imreadImg = imread(self.data_X[0])
        if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
            self.num_input_channels = imread(self.data_X[0]).shape[-1]
        else:
            self.num_input_channels = 1
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )
        self.emo = tf.placeholder(
            tf.float32,
            [self.size_batch, self.y_dim],
            name='emotion_labels'
        )
        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        self.rb = tf.placeholder(
            tf.float32,
            [self.size_batch, self.y_dim * self.rb_dim],
            name='rb')
        print '\n\tBuilding graph ...'
        self.z = self.encoder(
            image=self.input_image
        )
        self.G = self.generator(
            z=self.z,
            y=self.rb,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio
        )
        self.D_z, self.D_z_logits = self.discriminator_z(
            z=self.z,
            is_training=self.is_training
        )
        self.D_G, self.D_G_logits, self.D_G_feats, self.D_cont_G = self.discriminator_img(
            image=self.G,
            y=self.emo,
            is_training=self.is_training
        )
        self.D_z_prior, self.D_z_prior_logits = self.discriminator_z(
            z=self.z_prior,
            is_training=self.is_training,
            reuse_variables=True
        )
        self.D_input, self.D_input_logits, self.D_input_feats, self.D_cont_input = self.discriminator_img(
            image=self.input_image,
            y=self.emo,
            is_training=self.is_training,
            reuse_variables=True
        )

        self.real_conv1_2, self.real_conv2_2, self.real_conv3_2, self.real_conv4_2, self.real_conv5_2 = self.face_embedding(self.input_image)
        self.fake_conv1_2, self.fake_conv2_2, self.fake_conv3_2, self.fake_conv4_2, self.fake_conv5_2 = self.face_embedding(self.G)

        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss
        self.conv1_2_loss = tf.reduce_mean(tf.abs(self.real_conv1_2 - self.fake_conv1_2)) / 224. / 224.
        self.conv2_2_loss = tf.reduce_mean(tf.abs(self.real_conv2_2 - self.fake_conv2_2)) / 112. / 112.
        self.conv3_2_loss = tf.reduce_mean(tf.abs(self.real_conv3_2 - self.fake_conv3_2)) / 56. / 56.
        self.conv4_2_loss = tf.reduce_mean(tf.abs(self.real_conv4_2 - self.fake_conv4_2)) / 28. / 28.
        self.conv5_2_loss = tf.reduce_mean(tf.abs(self.real_conv5_2 - self.fake_conv5_2)) / 14. / 14.
        self.vgg_loss = self.conv1_2_coeff * self.conv1_2_loss + self.conv2_2_coeff * self.conv2_2_loss + \
                        self.conv3_2_coeff * self.conv3_2_loss + self.conv4_2_coeff * self.conv4_2_loss + self.conv5_2_coeff * self.conv5_2_loss

        self.D_z_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_prior_logits, tf.ones_like(self.D_z_prior_logits))
        )
        self.D_z_loss_z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_logits, tf.zeros_like(self.D_z_logits))
        )
        self.E_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_logits, tf.ones_like(self.D_z_logits))
        )

        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_input_logits, tf.ones_like(self.D_input_logits))
        )
        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.zeros_like(self.D_G_logits))
        )
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.ones_like(self.D_G_logits))
        )
        self.fm_loss = tf.reduce_mean(tf.abs(self.D_input_feats - self.D_G_feats))

        tv_y_size = self.size_image
        tv_x_size = self.size_image
        self.tv_loss = (
            (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
            (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1, :]) / tv_x_size)) / self.size_batch

        self.D_cont_loss_fake_array = []
        for i in range(self.y_dim):
            self.label_per_class = self.rb[:, i * self.rb_dim:(i + 1) * self.rb_dim]
            loss = tf.reduce_mean(
                tf.square(self.D_cont_G[i] - self.label_per_class)
            )
            self.D_cont_loss_fake_array.append(loss)
        self.D_cont_loss_fake = tf.reduce_sum(self.D_cont_loss_fake_array)

        trainable_variables = tf.trainable_variables()
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        self.z_summary = tf.summary.histogram('z', self.z)
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.D_z_loss_z_summary = tf.summary.scalar('D_z_loss_z', self.D_z_loss_z)
        self.D_z_loss_prior_summary = tf.summary.scalar('D_z_loss_prior', self.D_z_loss_prior)
        self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
        self.D_z_logits_summary = tf.summary.histogram('D_z_logits', self.D_z_logits)
        self.D_z_prior_logits_summary = tf.summary.histogram('D_z_prior_logits', self.D_z_prior_logits)
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)

        self.D_input_feats_summary = tf.summary.histogram('D_input_feats', self.D_input_feats)
        self.D_G_feats_summary = tf.summary.histogram('D_G_feats', self.D_G_feats)
        self.fm_loss_summary = tf.summary.scalar('fm_loss', self.fm_loss)
        self.vgg_loss_summary = tf.summary.scalar('VGG_loss', self.vgg_loss)
        self.conv5_2_loss_summary = tf.summary.scalar('conv5_2_loss', self.conv5_2_loss)
        self.conv4_2_loss_summary = tf.summary.scalar('conv4_2_loss', self.conv4_2_loss)
        self.conv3_2_loss_summary = tf.summary.scalar('conv3_2_loss', self.conv3_2_loss)
        self.conv2_2_loss_summary = tf.summary.scalar('conv2_2_loss', self.conv2_2_loss)
        self.conv1_2_loss_summary = tf.summary.scalar('conv1_2_loss', self.conv1_2_loss)
        self.D_cont_loss_fake_summary = tf.summary.scalar('D_cont_loss_fake', self.D_cont_loss_fake)
        self.D_cont_G_summary = tf.summary.histogram('D_cont_G', self.D_cont_G)

        self.saver = tf.train.Saver(max_to_keep=10)
        if self.is_stage_one:
            self.ft_saver = tf.train.Saver(self.G_variables + self.D_img_variables)

    def train(self,
              num_epochs=200,
              learning_rate=0.0002,
              beta1=0.5,
              decay_rate=1.0,
              enable_shuffle=True,
              use_trained_model=True,
              ):


        size_data = len(self.data_X)
        if self.is_stage_one:
            print '\n\tStage One'
            print '\n\tVGG_coeff: %f' % self.vgg_coeff
            self.loss_EG = self.EG_loss + self.vgg_coeff * self.vgg_loss + self.fm_coeff * self.fm_loss + \
                           0.000 * self.G_img_loss + 0.000 * self.E_z_loss + 0.000 * self.tv_loss + \
                           self.q_coeff * self.D_cont_loss_fake  # slightly increase the params
        else:
            print '\n\tStage Two'
            print '\n\tVGG_coeff: %f' % self.vgg_coeff
            self.loss_EG = self.EG_loss + self.vgg_coeff * self.vgg_loss + self.fm_coeff * self.fm_loss + \
                           0.01 * self.G_img_loss + 0.01 * self.E_z_loss + 0.001 * self.tv_loss + \
                            self.q_coeff * self.D_cont_loss_fake  # slightly increase the params
        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G + \
                        self.D_cont_loss_fake

        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )
        self.D_z_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Dz,
            var_list=self.D_z_variables
        )
        self.D_img_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Di,
            var_list=self.D_img_variables
        )
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.z_summary, self.z_prior_summary,
            self.D_z_loss_z_summary, self.D_z_loss_prior_summary,
            self.D_z_logits_summary, self.D_z_prior_logits_summary,
            self.EG_loss_summary, self.E_z_loss_summary,
            self.D_img_loss_input_summary, self.D_img_loss_G_summary,
            self.G_img_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary,
            self.fm_loss_summary, self.D_G_feats_summary,
            self.D_input_feats_summary, self.vgg_loss_summary,
            self.conv1_2_loss_summary,self.conv2_2_loss_summary,self.conv3_2_loss_summary,
            self.conv4_2_loss_summary,self.conv5_2_loss_summary,
            self.D_cont_loss_fake_summary, self.D_cont_G_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        sample_files = self.data_X[0:self.size_batch]
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
            is_flip=self.is_flip
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        sample_label_emo = self.data_y[0:self.size_batch]
        sample_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim), dtype=np.float32)
        for i in range(self.size_batch):
            sample_label_rb[i] = self.y_to_rb_label(sample_label_emo[i])

        print '\n\tPreparing for training ...'
        tf.global_variables_initializer().run()

        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        num_batches = len(self.data_X) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                seed = 2017
                np.random.seed(seed)
                np.random.shuffle(self.data_X)
                np.random.seed(seed)
                np.random.shuffle(self.data_y)
            for ind_batch in range(num_batches):
                start_time = time.time()
                batch_files = self.data_X[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                    is_flip=self.is_flip
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_label_emo = self.data_y[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim),
                                           dtype=np.float32)
                for i in range(self.size_batch):
                    batch_label_rb[i] = self.y_to_rb_label(batch_label_emo[i])

                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, TV, fm_err, vgg_err, conv1_2_err, conv2_2_err, \
                conv3_2_err, conv4_2_err, conv5_2_err, DG_cont_err= self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        self.EG_loss,
                        self.E_z_loss,
                        self.D_z_loss_z,
                        self.D_z_loss_prior,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input,
                        self.tv_loss,
                        self.fm_loss,
                        self.vgg_loss,
                        self.conv1_2_loss,
                        self.conv2_2_loss,
                        self.conv3_2_loss,
                        self.conv4_2_loss,
                        self.conv5_2_loss,
                        self.D_cont_loss_fake
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.emo: batch_label_emo,
                        self.z_prior: batch_z_prior,
                        self.rb: batch_label_rb
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f\tvgg_err=%.4f\tfmm_err=%.4f" %
                    (epoch+1, num_epochs, ind_batch+1, num_batches, EG_err, TV, vgg_err, fm_err))
                print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))
                print("\tDG_cont=%.4f" % (DG_cont_err))
                print("\tconv1_2=%.4f\tconv2_2=%.4f\tconv3_2=%.4f") % (conv1_2_err, conv2_2_err, conv3_2_err)
                print("\tconv4_2=%.4f\tconv5_2=%.4f") % (conv4_2_err, conv5_2_err)

                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.emo: batch_label_emo,
                        self.z_prior: batch_z_prior,
                        self.rb: batch_label_rb
                    }
                )

                self.writer.add_summary(summary, self.EG_global_step.eval())

            name = '{:02d}.png'.format(epoch+1)
            self.sample(sample_images, sample_label_rb, name=name)
            self.test(sample_images, name=name)

            if np.mod(epoch, 10) == 1:
                self.save_checkpoint()

        self.save_checkpoint()
        self.writer.close()

    def encoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name
                )
            current = tf.nn.relu(current)
        name = 'E_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=name
        )
        return tf.nn.tanh(current)

    def generator(self, z, y, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.y_dim)
        else:
            duplicate = 1
        z = concat_label(z, y, duplicate=duplicate)

        size_mini_map = int(self.size_image / 2 ** num_layers)
        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name
        )
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)
        current = concat_label(current, y)

        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = tf.image.resize_nearest_neighbor(current, [size_mini_map * 2 ** (i + 1), size_mini_map * 2 ** (i + 1)])
            current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 1)), name=name)
            current = tf.nn.relu(current)
            current = concat_label(current, y)

        name = 'G_deconv' + str(i + 1)
        current = tf.image.resize_nearest_neighbor(current, [self.size_image, self.size_image])
        current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 2)), name=name)
        current = tf.nn.relu(current)
        current = concat_label(current, y)

        name = 'G_deconv' + str(i + 2)
        current = custom_conv2d(input_map=current, num_output_channels=self.num_input_channels, name=name)
        return tf.nn.tanh(current)

    def discriminator_z(self, z, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        current = z
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_z_fc' + str(i)
            current = fc(
                    input_vector=current,
                    num_output_length=num_hidden_layer_channels[i],
                    name=name
                )
            if enable_bn:
                name = 'D_z_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
        name = 'D_z_fc' + str(i+1)
        current = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        return tf.nn.sigmoid(current), current

    def discriminator_img(self, image, y, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        current = concat_label(current, y)
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name
                )
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
            current = concat_label(current, y)

        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        name = 'D_img_fc1_bn'
        current = tf.contrib.layers.batch_norm(
            current,
            scale=False,
            is_training=is_training,
            scope=name,
            reuse=reuse_variables
        )
        current = lrelu(current)
        shared = concat_label(current, y)

        name = 'D_img_fc2'
        disc = fc(
            input_vector=shared,
            num_output_length=1,
            name=name
        )
        name = 'D_img_q_shared'
        q_shared = fc(
            input_vector=shared,
            num_output_length=128,
            name=name
        )
        name = 'D_img_q_shared_bn'
        q_shared = tf.contrib.layers.batch_norm(
            q_shared,
            scale=False,
            is_training=is_training,
            scope=name,
            reuse=reuse_variables
        )
        q_shared = lrelu(q_shared)

        cats = []
        for i in range(self.y_dim):
            name = 'D_img_q_fc' + str(i)
            cat_fc = fc(q_shared, 64, name=name)

            name = 'D_img_q_fc_bn' + str(i)
            cat_fc = tf.contrib.layers.batch_norm(
                cat_fc,
                scale=False,
                is_training=is_training,
                scope=name,
                reuse=reuse_variables
            )
            cat_fc = lrelu(cat_fc)

            name = 'D_img_q_cat' + str(i)
            cat = tf.nn.tanh(fc(cat_fc, self.rb_dim, name=name))
            cats.append(cat)

        return tf.nn.sigmoid(disc), disc, q_shared, cats

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'checkpoint')
        print checkpoint_dir
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            if self.is_stage_one:
                self.ft_saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            else:
                self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, labels, name=None):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: images,
                self.rb: labels
            }
        )

        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[self.y_dim, int(self.size_batch/self.y_dim)]
        )

    def test(self, images, name=None):
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        images = images[:int(self.size_batch / self.y_dim), :, :, :]

        size_sample = images.shape[0]
        labels = np.arange(self.y_dim)
        labels = np.repeat(labels, size_sample)
        query_labels = np.zeros(
            shape=(size_sample * self.y_dim, self.y_dim * self.rb_dim),
            dtype=np.float32
        )

        for i in range(query_labels.shape[0]):
            one_hot = np.zeros(self.y_dim, dtype=np.float32)
            one_hot[labels[i]] = 1.0
            query_labels[i] = self.y_to_rb_label(one_hot)

        query_images = np.tile(images, [self.y_dim, 1, 1, 1])

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.rb: query_labels
            }
        )

        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'test_input.png'),
            image_value_range=self.image_value_range,
            size_frame=[self.y_dim, size_sample]

        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[self.y_dim, size_sample]

        )

    def custom_test(self, testing_samples_dir, random_seed):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(self.size_batch / self.y_dim)
        if testing_samples_dir:
            file_names = glob(testing_samples_dir)
        else:
            np.random.seed(random_seed)
            file_names = self.data_X[num_samples:2*num_samples]
        if len(file_names) < num_samples:
            print 'The number of testing images must be larger than %d' % num_samples
            exit(0)
        sample_files = file_names[0:num_samples]
        print sample_files
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
            is_flip=False
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        print images.shape

        self.test(images, name='test.png')

    def face_embedding(self, images):
        images = (images+1)/2 * 255
        if self.dataset_name == 'CK' or self.dataset_name == 'TFD':
            images = tf.tile(images, [1, 1, 1, 3])
        net = vgg_face(self.vgg_weights, images)
        return net['conv1_2'], net['conv2_2'], net['conv3_2'], net['conv4_2'], net['conv5_2']

    def load_anno(self, anno_file):
        print "\n\tLoading anno"
        anno = pickle.load(open(anno_file, 'rb'))
        if self.is_training:
            anno = anno['train']
        else:
            anno = anno['test']
        X = anno.keys()
        y = anno.values()
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        y_vec = np.ones(shape=(len(y), self.y_dim), dtype=np.float32) * self.label_value_range[0]
        for i, label in enumerate(y):
            y_vec[i, label] = self.label_value_range[-1]
        return X, y_vec

    def y_to_rb_label(self, label):
        number = np.argmax(label)
        one_hot = np.random.uniform(-1, 1, self.rb_dim)
        rb = np.tile(-1*np.abs(one_hot), self.y_dim)
        rb[number * self.rb_dim:(number + 1) * self.rb_dim] = np.abs(one_hot)
        return rb
