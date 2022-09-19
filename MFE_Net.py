import motion
import tensorflow as tf
from MC_net import *



def Feature(input):

    en_out, x_1, x_3, x_5 = Encode(input, "Fe")
    x = tf.layers.conv2d(inputs=en_out, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                         activation=tf.nn.relu)

    x = res_block(x, 64, 27)
    x = up_sample(x, scale_factor=2)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                         activation=tf.nn.relu)

    x = x + x_5

    x = res_block(x, 64, 28)
    x = up_sample(x, scale_factor=2)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                         activation=tf.nn.relu)

    x = x + x_3
    x = res_block(x, 64, 29)
    x = up_sample(x, scale_factor=2)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                         activation=tf.nn.relu)

    x = x + x_1
    x = res_block(x, 64, 30)
    state_1_out = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                   activation=tf.nn.relu)
    stage1_output = tf.layers.conv2d(inputs=state_1_out, filters=3, kernel_size=[3, 3], padding="same", strides=1,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    # ---------stage_2---------#
    name = "stage_2"
    stage_2 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               activation=tf.nn.relu)
    stage_2 = tf.layers.conv2d(inputs=stage_2, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               activation=tf.nn.relu)
    stage_2 = RCB(stage_2, 31)
    stage_2 = FF12(x_1, stage_2, 0, 64)
    stage_2 = RCB(stage_2, 32)
    stage_2 = FF12(x_3, stage_2, 1, 64)
    stage_2 = RCB(stage_2, 33)
    stage_2 = FF12(x_5, stage_2, 2, 64)
    stage_2 = tf.layers.conv2d(inputs=stage_2, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    stage_2 = res_block(stage_2, 64, 41)
    stage_2_out = tf.layers.conv2d(inputs=stage_2, filters=3, kernel_size=[3, 3], padding="same", strides=1,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    # ----------weight----------#
    W = weight_01(stage1_output, 64)

    output = W * stage1_output + (1 - W) * stage_2_out

    return output



def MFE_Net(Y0_com, Y1_com, batch_size, Height, Width):
    with tf.variable_scope("MFE_Net"):
        flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_com, batch_size, Height, Width)
        Y1_flow = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)
    with tf.variable_scope("stage1"):
        F_stage1 = Feature(Y1_com)
    with tf.variable_scope("stage2"):
        F_stage2 = Feature(Y1_flow)
    with tf.variable_scope("Weight1"):
        W1 = weight_01(Y1_com, 64)
    with tf.variable_scope("Weight2"):
        W2 = weight_01(Y1_flow, 64)
    output = W1 * F_stage1 + W2 * F_stage2 + (1 - W1 - W2) * Y1_com
    return output

