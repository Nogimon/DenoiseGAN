#Copyright: LD

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import math
import keras

# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg

def generator(gen_inputs, gen_output_channels, reuse = False):
    #parameters from flag:
    is_training = True
    num_resblock = 16

    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = slim.conv2d(inputs, 64, [3, 3], stride, 'SAME', data_format = 'NHWC', 
                            activation_fn = None, weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer=None,
                            scope = 'conv_1')
            net = slim.batch_norm(net, decay = 0.9, epsilon = 0.001, updates_collections = tf.GraphKeys.UPDATE_OPS, scale = False, fused = True, is_training = is_training)
            net = prelu_tf(net)

            #need to check the scope here
            net = slim.conv2d(net, 64, [3, 3], stride, 'SAME', data_format = 'NHWC', 
                            activation_fn = None, weights_initializer = tf.contrib.layers.xavier_initializer(), biases_initializer=None,
                            scope = 'conv_2')

            net = slim.batch_norm(net, decay = 0.9, epsilon = 0.001, updates_collections = tf.GraphKeys.UPDATE_OPS, scale = False, fused = True, is_training = is_training)
            net = net + inputs
        return net

    with tf.variable_scope('generator_unit', reuse = reuse):
        #input layer
        with tf.variable_scope('input_stage'):
            net = slim.conv2d(gen_inputs, 64, [9, 9], 1, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            scope = 'conv')

            net = prelu_tf(net)

        stage1_output = net

        #residual block
        for i in range(1, num_resblock+1, 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = slim.conv2d(net, 64, [3, 3], 1, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None, scope = 'conv')
            net = slim.batch_norm(net, decay = 0.9, epsilon = 0.001, updates_collections = tf.GraphKeys.UPDATE_OPS, scale = False, fused = True, is_training = is_training)

        net = net + stage1_output
        '''
        with tf.variable_scope('subpixelconv_stage1'):
            net = slim.conv2d(net, 256, [3, 3], 1, 'SAME', data_format='NCWH',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            scope = 'conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = slim.conv2d(net, 256, [3, 3], 1, 'SAME', data_format='NCWH',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            scope = 'conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)
        '''
        with tf.variable_scope('output_stage'):
            net = slim.conv2d(net, gen_output_channels, [9, 9], 1, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            scope = 'conv')
    return net


def discriminator(dis_inputs):

    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = slim.conv2d(inputs, output_channel, [kernel, kernel], stride,  'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None, scope = 'conv1')
            net = slim.batch_norm(net, decay = 0.9, epsilon = 0.001, updates_collections = tf.GraphKeys.UPDATE_OPS, scale = False, fused = True, is_training = is_training)
            net = keras.layers.LeakyReLu(alpha = 0.2).call(net)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):

            #input
            with tf.variable_scope('input_stage'):
                net = slim.conv2d(dis_inputs, 64, [3, 3], 1, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               scope = 'conv')
                net = keras.layers.LeakyReLu(alpha = 0.2).call(net)

            #discriminator block
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            with tf.variable_scope('dense_layer_1'):
                net = slim.flatten(net)
                net = tf.layers.dense(net, 1024, activation = None, kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = keras.layers.LeakyReLu(alpha = 0.2).call(net)

            with tf.variable_scope('dense_layer_2'):
                net = slim.flatten(net)
                net = tf.layers.dense(net, 1, activation = None, kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.nn.sigmoid(net)
    return net

def VGG19_slim(input, type, reuse, scope):
    target_layer = scope + 'vgg_19/conv5/conv5_4'
    _, output = vgg_19(input, is_training = False, reuse = reuse)
    output = output[target_layer]
    return output

def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
               fc_conv_padding='VALID'):
    """Oxford Net VGG 19-Layers version E Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
    Returns:
    the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # Use conv2d instead of fully_connected layers.
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


# whole network
def SRGAN(inputs, targets):
    #from the FLAGS:
    batch_size = 16
    crop_size = 24
    vgg_scaling = 0.0061
    EPS = 1e-12
    ratio = 0.001
    decay_step = 100000
    decay_rate = 0.1
    stair = True
    beta = 0.9
    flags_learning_rate = 0.0001

    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'discrim_real_output, discrim_fake_output, discrim_loss, \
        discrim_grads_and_vars, adversarial_loss, content_loss, gen_grads_and_vars, gen_output, train, global_step, \
        learning_rate')

            
    #generator
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse = False)
        gen_output.set_shape([batch_size, crop_size*4, crop_size*4, 3])

    #fake discriminator
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse = False):
            discrim_fake_output = discriminator(gen_output)

    #real discriminator
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse = False):
            discrim_real_output = discriminator(targets)


    # Use the VGG54 feature
    with tf.name_scope('vgg19_1') as scope:
        extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
    with tf.name_scope('vgg19_2') as scope:
        extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)


    #calculate generator loss
    with tf.variable_scope('generator_loss'):
        with tf.variable_scope('content_loss'):
            diff = extracted_feature_gen - extracted_feature_target
            content_loss = vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis = [3]))
        with tf.variable_scope('adversarial_loss'):
            adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + EPS))
        gen_loss = content_loss + ratio * adversarial_loss
        print(adversarial_loss.get_shape())
        print(content_loss.get_shape())
        print("first print function in SRGAN function")


    #calculate discriminator loss
    with tf.variable_scope('discriminator_loss'):
        discrim_fake_loss = tf.log(1 - discrim_fake_ouput + EPS)
        discrim_real_loss = tf.log(discrim_real_output + EPS)
        discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

    #define learning rate and global stetp
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags_learning_rate, global_step, decay_step, decay_rate, staircase = stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('discriminator_train'):
        discrim_tvars = tf.get_collection(tf.GraphKeys.TRANABLE_VARIABLES, scope = 'discriminator')
        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = beta)
        discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
        discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

    with tf.variable_scope('generator_train'):
        with tf.control_dependencies([discrim_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRANABLE_VARIABLES, scope = 'generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1 = beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    exp_averager = tf.train.ExponentialMovingAverage(decay = 0.99)
    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])

    return Network(
        discrim_real_output = discrim_real_output,
        discrim_fake_output = discrim_fake_output,
        discrim_loss = exp_averager.average(discrim_loss),
        discrim_loss_and_vars = discrim_grads_and_vars,
        adversarial_loss = exp_averager.average(adversarial_loss),
        content_loss = exp_averager.average(content_loss),
        gen_grads_and_vars = gen_grads_and_vars,
        train = tf.group(update_loss, incr_global_step, gen_train),
        global_step = global_step,
        learning_rate = learning_rate)

    
