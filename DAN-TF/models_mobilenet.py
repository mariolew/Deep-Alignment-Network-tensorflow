from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib as tc

from layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer

IMGSIZE = 112
N_LANDMARK = 68

def NormRmse(GroudTruth, Prediction):
    Gt = tf.reshape(GroudTruth, [-1, N_LANDMARK, 2])
    Pt = tf.reshape(Prediction, [-1, N_LANDMARK, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
    # norm = tf.sqrt(tf.reduce_sum(((tf.reduce_mean(Gt[:, 36:42, :],1) - \
    #     tf.reduce_mean(Gt[:, 42:48, :],1))**2), 1))
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
    # cost = tf.reduce_mean(loss / norm)

    return loss/norm


def DAN(MeanShapeNumpy):

    MeanShape = tf.constant(MeanShapeNumpy, dtype=tf.float32)
    InputImage = tf.placeholder(tf.float32,[None, IMGSIZE,IMGSIZE,1])
    GroundTruth = tf.placeholder(tf.float32,[None, N_LANDMARK * 2])
    S1_isTrain = tf.placeholder(tf.bool)
    S2_isTrain = tf.placeholder(tf.bool)
    Ret_dict = {}
    Ret_dict['InputImage'] = InputImage
    Ret_dict['GroundTruth'] = GroundTruth
    Ret_dict['S1_isTrain'] = S1_isTrain
    Ret_dict['S2_isTrain'] = S2_isTrain
    bn_params_s1 = {'is_training': S1_isTrain}
    bn_params_s2 = {'is_training': S2_isTrain}

    with tf.variable_scope('Stage1'):
        s1_conv1a = tc.layers.conv2d(InputImage, num_outputs=32, kernel_size=3, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        # First depth wise separable conv block
        s1_dconv1 = tc.layers.separable_conv2d(s1_conv1a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv1 = tc.layers.conv2d(s1_dconv1, num_outputs=64, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        # Second depth wise separable conv block
        s1_dconv2a = tc.layers.separable_conv2d(s1_pconv1, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=1,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv2a = tc.layers.conv2d(s1_dconv2a, num_outputs=128, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        s1_dconv2b = tc.layers.separable_conv2d(s1_pconv2a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv2b = tc.layers.conv2d(s1_dconv2b, num_outputs=128, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        # Third depth wise separable conv block
        s1_dconv3a = tc.layers.separable_conv2d(s1_pconv2b, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=1,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv3a = tc.layers.conv2d(s1_dconv3a, num_outputs=256, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        s1_dconv3b = tc.layers.separable_conv2d(s1_pconv3a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv3b = tc.layers.conv2d(s1_dconv3b, num_outputs=256, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        # Forth depth wise separable conv block
        s1_dconv4a = tc.layers.separable_conv2d(s1_pconv3b, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=1,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv4a = tc.layers.conv2d(s1_dconv4a, num_outputs=512, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)

        s1_dconv4b = tc.layers.separable_conv2d(s1_pconv4a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s1)
        s1_pconv4b = tc.layers.conv2d(s1_dconv4b, num_outputs=512, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s1)


        # s1_pconv4b_flat = tc.layers.flatten(s1_pconv4b)
        # s1_dropout = tc.layers.dropout(s1_pconv4b_flat,0.5,is_training=S1_isTrain)

        # s1_fc1 = tc.layers.fully_connected(s1_dropout, num_outputs=256,
        #                                    normalizer_fn=tc.layers.batch_norm, \
        #                                    weights_initializer=tf.glorot_uniform_initializer(),
        #                                    normalizer_params=bn_params_s1)
        s1_fc1 = tc.layers.max_pool2d(s1_pconv4b, kernel_size=7)
        s1_fc1_flat = tc.layers.flatten(s1_fc1)
        s1_dropout = tc.layers.dropout(s1_fc1_flat,0.5,is_training=S1_isTrain)
        s1_fc2 = tc.layers.fully_connected(s1_dropout, num_outputs=N_LANDMARK * 2, activation_fn=None)

        S1_Ret = s1_fc2 + MeanShape
        S1_Cost = tf.reduce_mean(NormRmse(GroundTruth, S1_Ret))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))

    Ret_dict['S1_Ret'] = S1_Ret
    Ret_dict['S1_Cost'] = S1_Cost
    Ret_dict['S1_Optimizer'] = S1_Optimizer
    # Ret_dict['debug'] = s1_pconv4b_flat


    with tf.variable_scope('Stage2'):

        S2_AffineParam = TransformParamsLayer(S1_Ret, MeanShape)
        S2_InputImage = AffineTransformLayer(InputImage, S2_AffineParam)
        S2_InputLandmark = LandmarkTransformLayer(S1_Ret, S2_AffineParam)
        S2_InputHeatmap = LandmarkImageLayer(S2_InputLandmark)

        S2_Feature = tf.reshape(tf.layers.dense(s1_fc1,int((IMGSIZE / 2) * (IMGSIZE / 2)),\
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),(-1,int(IMGSIZE / 2),int(IMGSIZE / 2),1))
        S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(IMGSIZE,IMGSIZE),1)

        S2_ConcatInput = tf.layers.batch_normalization(tf.concat([S2_InputImage,S2_InputHeatmap,S2_FeatureUpScale],3),\
            training=S2_isTrain)


        s2_conv1a = tc.layers.conv2d(S2_ConcatInput, num_outputs=32, kernel_size=3, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        # First depth wise separable conv block
        s2_dconv1 = tc.layers.separable_conv2d(s2_conv1a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv1 = tc.layers.conv2d(s2_dconv1, num_outputs=64, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        # Second depth wise separable conv block
        s2_dconv2a = tc.layers.separable_conv2d(s2_pconv1, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=1,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv2a = tc.layers.conv2d(s2_dconv2a, num_outputs=128, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        s2_dconv2b = tc.layers.separable_conv2d(s2_pconv2a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv2b = tc.layers.conv2d(s2_dconv2b, num_outputs=128, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        # Third depth wise separable conv block
        s2_dconv3a = tc.layers.separable_conv2d(s2_pconv2b, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=1,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv3a = tc.layers.conv2d(s2_dconv3a, num_outputs=256, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        s2_dconv3b = tc.layers.separable_conv2d(s2_pconv3a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv3b = tc.layers.conv2d(s2_dconv3b, num_outputs=256, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        # Forth depth wise separable conv block
        s2_dconv4a = tc.layers.separable_conv2d(s2_pconv3b, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=1,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv4a = tc.layers.conv2d(s2_dconv4a, num_outputs=512, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)

        s2_dconv4b = tc.layers.separable_conv2d(s2_pconv4a, \
                                               num_outputs=None, kernel_size=3, depth_multiplier=1, stride=2,
                                                     normalizer_fn=tc.layers.batch_norm,
                                                     weights_initializer=tf.glorot_uniform_initializer(),
                                                      normalizer_params=bn_params_s2)
        s2_pconv4b = tc.layers.conv2d(s2_dconv4b, num_outputs=512, kernel_size=1, stride=1,
                                           normalizer_fn=tc.layers.batch_norm, \
                                           weights_initializer=tf.glorot_uniform_initializer(),
                                           normalizer_params=bn_params_s2)


        # s2_pconv4b_flat = tc.layers.flatten(s2_pconv4b)
        # s2_dropout = tc.layers.dropout(s2_pconv4b_flat,0.5,is_training=S2_isTrain)

        # s2_fc1 = tc.layers.fully_connected(s2_dropout, num_outputs=256,
        #                                    normalizer_fn=tc.layers.batch_norm, \
        #                                    weights_initializer=tf.glorot_uniform_initializer(),
        #                                    normalizer_params=bn_params_s2)
        # s2_fc1 = tc.layers.max_pool2d(s2_dropout, kernel_size=7)
        # s2_fc2 = tc.layers.fully_connected(s2_fc1, num_outputs=N_LANDMARK * 2, activation_fn=None)
        s2_fc1 = tc.layers.max_pool2d(s2_pconv4b, kernel_size=7)
        s2_fc1_flat = tc.layers.flatten(s2_fc1)
        s2_dropout = tc.layers.dropout(s2_fc1_flat,0.5,is_training=S2_isTrain)
        s2_fc2 = tc.layers.fully_connected(s2_dropout, num_outputs=N_LANDMARK * 2, activation_fn=None)

        S2_Ret = s2_fc2 + MeanShape
        S2_Cost = tf.reduce_mean(NormRmse(GroundTruth, S2_Ret))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage2')):
            S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage2"))


    Ret_dict['S2_Cost'] = S2_Cost
    Ret_dict['S2_Optimizer'] = S2_Optimizer

    Ret_dict['S2_InputImage'] = S2_InputImage
    Ret_dict['S2_InputLandmark'] = S2_InputLandmark
    Ret_dict['S2_InputHeatmap'] = S2_InputHeatmap
    Ret_dict['S2_FeatureUpScale'] = S2_FeatureUpScale
    
    return Ret_dict
