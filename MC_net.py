import tensorflow as tf
from cell import ConvLSTMCell
import tensorflow.contrib as tf_contrib
from ops import *


def res_block(input, num_filters, layer):
    with tf.variable_scope("res_block_" + str(layer), reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                 activation=tf.nn.relu)
        output = input + conv2
        return output

def down_sample( input, numfilters, layer):
    with tf.variable_scope("down_sample_" + str(layer), reuse=tf.AUTO_REUSE):
        x = tf.layers.average_pooling2d(input, pool_size=2, strides=2, padding='same')
        x = tf.layers.conv2d(inputs=x, filters=numfilters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation=tf.nn.relu)
        return x

def CA(input, num_filters):
    GAP = tf.reduce_mean(input, axis=[1, 2])
    GAP = tf.expand_dims(GAP, axis=1)
    GAP = tf.expand_dims(GAP, axis=1)
    conv1 = tf.layers.conv2d(inputs=GAP, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation=tf.nn.sigmoid)
    return input * conv2

def CAB(input, num_filters):
    conv1 = tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    CA_out = CA(conv2, 64)
    return CA_out + input
def weight_01(input,num_filters):
    conv1 = tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation=tf.nn.relu)
    conv1 = res_block(conv1, num_filters, 45)
    conv1 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    conv1 = res_block(conv1, num_filters, 46)
    conv1 = tf.layers.conv2d(inputs=conv1, filters=3, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation = tf.nn.sigmoid)
    #out = tf.clip_by_value(conv1,0,1)
    return conv1
def RCB(stage_2, j):
    stage_21 = res_block(stage_2, 64, j)
    stage_21 = CAB(stage_21, 64)
    stage_21 = res_block(stage_21, 64, j + 4)
    stage_2 = stage_21 + stage_2
    return stage_2

def FF12(stage_1, stage_2, num_up, num_fil):
    x = stage_1
    for i in range(num_up):
        x = up_sample(x, scale_factor=2)
        x = tf.layers.conv2d(inputs=x, filters=num_fil, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    x = tf.concat([x, stage_2], 3)
    out = tf.layers.conv2d(inputs=x, filters=num_fil, kernel_size=[3, 3], padding="same", strides=1,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    return out

def Encode(input, name):
    with tf.variable_scope("Encode" + str(name), reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                             activation=tf.nn.relu)
        num_filters = 64
        x_1 = res_block(x, 64, str(name) + str(20))
        x_2 = down_sample(x_1, 64, str(name) + str(21))
        x_3 = res_block(x_2, 64, str(name) + str(22))
        x_4 = down_sample(x_3, 64, str(name) + str(23))
        x_5 = res_block(x_4, 64, str(name) + str(24))
        x_6 = down_sample(x_5, 64, str(name) + str(25))
        x_7 = res_block(x_6, 64, str(name) + str(26))
        return x_7, x_1, x_3, x_5

def one_LSTM(tensor, state_c, state_h, Height, Width, num_filters):

    tensor = tf.expand_dims(tensor, axis=1)
    print("aaa", tensor)
    kernal = [3, 3]
    act = tf.tanh
    state_c = tf.squeeze(state_c, axis=0)
    state_h = tf.squeeze(state_h, axis=0)
    cell = ConvLSTMCell(shape=[Height // 8, Width // 8], activation=act,
                                  filters=num_filters, kernel=kernal)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    tensor, state = tf.nn.dynamic_rnn(cell, tensor, initial_state=state, dtype=tensor.dtype)
    state_c, state_h = state

    tensor = tf.squeeze(tensor, axis=1)

    return tensor, state_c, state_h

def conv_LSTM(inputs, state_c, state_h):
    _, H, W, channels = inputs.get_shape()

    kernel = [3, 3]
    shape = [H, W]
    filters = 64
    LSTM_input = tf.expand_dims(inputs, axis=1)
    cell = ConvLSTMCell(shape, filters, kernel)
    state_c = tf.squeeze(state_c, axis=0)
    state_h = tf.squeeze(state_h, axis=0)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    # state = tf.squeeze(state, axis=1)
    print('LSTM_input', state)
    outputs, state = tf.nn.dynamic_rnn(cell, LSTM_input, initial_state=state, dtype=LSTM_input.dtype)
    outputs = tf.squeeze(outputs, axis=1)

    state_c, state_h = state
    return outputs, state_c, state_h

def Generator(Y1_warp, Y0_com, state_c, state_h, Height, Width):
    with tf.variable_scope("genreator", reuse=tf.AUTO_REUSE):
        input = tf.concat([Y1_warp, Y0_com], axis=-1)
        x = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1)
        x = tf_contrib.layers.batch_norm(x, scope="BN")
        x = tf.nn.relu(x)
        num_filters = 64
        # Down-Sampling
        for i in range(2):
            num_filters *= 2
            x = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=[3, 3], padding="same", strides=2)
            x = tf_contrib.layers.batch_norm(x, scope="BN" + str(i))
            x = tf.nn.relu(x)

        for j in range(4):
            x = res_block(x, num_filters, j)
            x = tf_contrib.layers.batch_norm(x, scope="BN" + str(j + 2))
            x = tf.nn.relu(x)

        num_filters = 64
        act = tf.nn.relu
        # LSTM_output,state_c,state_h= self.conv_LSTM(x,state_c,state_h)
        LSTM_output, state_c, state_h = one_LSTM(x, state_c, state_h, Height, Width, num_filters)
        x = tf.layers.conv2d(inputs=LSTM_output, filters=num_filters, kernel_size=[3, 3], padding="same",
                             strides=1)

        for j in range(4):
            x = res_block(x, num_filters, j)
            x = tf_contrib.layers.batch_norm(x, scope="BN" + str(j + 6))
            x = tf.nn.relu(x)
        # Up-Sampling
        for i in range(2):
            num_filters = num_filters // 2
            x = up_sample(x, scale_factor=2)
            x = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=[3, 3], padding="same", strides=1)
            x = tf_contrib.layers.batch_norm(x, scope="BN" + str(i + 10))
            x = tf.nn.relu(x)

        x = tf.layers.conv2d(inputs=x, filters=3, kernel_size=[3, 3], padding="same", strides=1)
        output = tf.tanh(x)
        return output, state_c, state_h

def MC_Generator_1(Y1_warp, Y0_com, state_c, state_h, Height, Width):
    en_out, x_1, x_3, x_5 = Encode(Y1_warp, "Encode_Y1_warp")
    num_filters = 64
    LSTM_output, state_c, state_h = one_LSTM(en_out, state_c, state_h, Height, Width, num_filters)
    x = tf.layers.conv2d(inputs=LSTM_output, filters=64, kernel_size=[3, 3], padding="same", strides=1,
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
    stage_2 = tf.layers.conv2d(inputs=Y1_warp, filters=64, kernel_size=[3, 3], padding="same", strides=1,
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
    #----------weight----------#
    W = weight_01(stage1_output,64)
    output = W * stage1_output + (1 - W)*stage_2_out
    return output, state_c, state_h

def discriminator(input, name, sn):
    # with tf.variable_scope('discriminator', reuse=False):
    x = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1)
    x = tf_contrib.layers.batch_norm(x, scope="D_BN" + name)
    x = tf.nn.relu(x)
    num_filters = 64
    # Down-Sampling
    for i in range(4):
        num_filters *= 2
        x = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=[3, 3], padding="same", strides=2)
        x = tf_contrib.layers.batch_norm(x, scope="D_BN" + name + str(i))
        x = tf.nn.relu(x)
    num_filters *= 2
    x = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=[3, 3], padding="same", strides=2)
    cam_x = tf.reduce_mean(x, axis=[1, 2])
    cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=sn, scope='CAM_logit' + name)
    x_gap = tf.multiply(x, cam_x_weight)

    cam_x = tf.reduce_max(x, axis=[1, 2])
    cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=sn, reuse=True, scope='CAM_logit' + name)
    x_gmp = tf.multiply(x, cam_x_weight)

    cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
    x = tf.concat([x_gap, x_gmp], axis=-1)

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=1)
    x = tf_contrib.layers.batch_norm(x, scope="D1_BN" + name)
    x = tf.nn.relu(x)
    output = tf.layers.conv2d(inputs=x, filters=1, kernel_size=[3, 3], padding="same", strides=1)
    return output, cam_logit






