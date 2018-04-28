#coding=utf-8

# 模型定义

import os
import time
import datetime

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

from ops import *

# from layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer


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



def CNN():

    #MeanShape = tf.constant(MeanShapeNumpy, dtype=tf.float32)
    InputImage = tf.placeholder(tf.float32,[None, IMGSIZE,IMGSIZE,1])
    GroundTruth = tf.placeholder(tf.float32,[None, N_LANDMARK * 2])
    S1_isTrain = tf.placeholder(tf.bool)
    Ret_dict = {}
    Ret_dict['InputImage'] = InputImage
    Ret_dict['GroundTruth'] = GroundTruth
    Ret_dict['S1_isTrain'] = S1_isTrain
    exp = 6

    with tf.variable_scope('Stage1'):

        net = conv2d_block(InputImage, 32, 3, 2, S1_isTrain, name='conv1_1')  # size/2

        net = res_block(net, 1, 16, 1, S1_isTrain, name='res2_1')

        net = res_block(net, exp, 24, 2, S1_isTrain, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, S1_isTrain, name='res3_2')

        net = res_block(net, exp, 32, 2, S1_isTrain, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, S1_isTrain, name='res4_2')
        net = res_block(net, exp, 32, 1, S1_isTrain, name='res4_3')

        net = res_block(net, exp, 64, 1, S1_isTrain, name='res5_1')
        net = res_block(net, exp, 64, 1, S1_isTrain, name='res5_2')
        net = res_block(net, exp, 64, 1, S1_isTrain, name='res5_3')
        net = res_block(net, exp, 64, 1, S1_isTrain, name='res5_4')

        net = res_block(net, exp, 96, 2, S1_isTrain, name='res6_1')  # size/16
        net = res_block(net, exp, 96, 1, S1_isTrain, name='res6_2')
        net = res_block(net, exp, 96, 1, S1_isTrain, name='res6_3')

        # net = res_block(net, exp, 160, 2, S1_isTrain, name='res7_1')  # size/32
        # net = res_block(net, exp, 160, 1, S1_isTrain, name='res7_2')
        # net = res_block(net, exp, 160, 1, S1_isTrain, name='res7_3')

        net = res_block(net, exp, 128, 1, S1_isTrain, name='res7_1', shortcut=False)

        net = pwise_block(net, 256, S1_isTrain, name='conv8_1')
        net = global_avg(net)
        net = tf.layers.dropout(net, 0.5, training=S1_isTrain)
        logits = flatten(conv_1x1(net, N_LANDMARK * 2, name='logits'))


        #S1_Ret = S1_Fc2 + MeanShape
        S1_Cost = tf.reduce_mean(NormRmse(GroundTruth, logits))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))
        
    Ret_dict['S1_Ret'] = logits
    Ret_dict['S1_Cost'] = S1_Cost
    Ret_dict['S1_Optimizer'] = S1_Optimizer


    
    return Ret_dict
