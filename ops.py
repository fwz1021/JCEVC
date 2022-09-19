import tensorflow as tf
import tensorflow.contrib as tf_contrib
import os
from scipy import misc
import numpy as np
import cv2
def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)
def flatten(x) :
    return tf.layers.flatten(x)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
def read_img(path_int,path_float):
    F1_com_int = misc.imread(path_int)
    F1_com_float = misc.imread(path_float)
    F1_float = F1_com_float.astype(np.float32) / 255.0
    F1_int = F1_com_int.astype(np.float32)
    F1_com = F1_int + F1_float
    return F1_com

def write_img(F0_com,p_int,p_float):
    F0_int = F0_com.astype(np.uint8)
    F0_float = (F0_com - F0_int) * 255
    F0_float = F0_float.astype(np.uint8)
    Y1_int = cv2.cvtColor(F0_int, cv2.COLOR_RGB2BGR)
    Y1_float = cv2.cvtColor(F0_float, cv2.COLOR_RGB2BGR)
    cv2.imwrite(p_int, Y1_int)
    cv2.imwrite(p_float, Y1_float)

def fully_connected_with_w(x, use_bias=True, sn=False, reuse=False, scope='linear'):
    with tf.variable_scope(scope, reuse=reuse):
        x = tf.layers.flatten(x)
        bias = 0.0
        shape = x.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable("kernel", [channels, 1], tf.float32,
                            initializer=weight_init, regularizer=weight_regularizer)

        if sn :
            w = spectral_norm(w)  #该技术从每层神经网络的参数矩阵的谱范数角度，引入正则约束，
            # 使神经网络对输入扰动具有较好的非敏感性，从而使训练过程更稳定，更容易收敛

        if use_bias :
            bias = tf.get_variable("bias", [1],
                                   initializer=tf.constant_initializer(0.0))

            x = tf.matmul(x, w) + bias
        else :
            x = tf.matmul(x, w)

        if use_bias :
            weights = tf.gather(tf.transpose(tf.nn.bias_add(w, bias)), 0)     #将张量中对应索引的向量提取出来
        else :
            weights = tf.gather(tf.transpose(w), 0)

        return x, weights

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

def discriminator_loss(real, fake):

    real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))

    return real_loss + fake_loss

def generator_loss(real,fake,fake_out,fake_cam):

    fake_loss = tf.reduce_mean(tf.squared_difference(fake_out, 1.0))
    fake_cam_loss = tf.reduce_mean(tf.squared_difference(fake_cam, 1.0))
    loss = tf.reduce_mean(tf.squared_difference(real,fake))

    return loss*1000 + fake_loss + fake_cam_loss

def L2_loss(real,fake):

    loss = tf.reduce_mean(tf.squared_difference(real, fake))
    return loss
def configure(args):

    path = args.path + '/'
    path_com = args.path + '_' + args.mode + '_' + str(args.l) + '/frames/'
    path_bin = args.path + '_' + args.mode + '_' + str(args.l) + '/bitstreams/'
    path_lat = args.path + '_' + args.mode + '_' + str(args.l) + '/latents/'

    os.makedirs(path_com, exist_ok=True)
    os.makedirs(path_bin, exist_ok=True)
    os.makedirs(path_lat, exist_ok=True)

    F1 = misc.imread(path + 'f001.png')
    Height = np.size(F1, 0)
    Width = np.size(F1, 1)
    batch_size = 1
    Channel = 3

    if (Height % 16 != 0) or (Width % 16 != 0):
        raise ValueError('Height and Width must be a mutiple of 16.')

    activation = tf.nn.relu

    GOP_size = args.f_P + args.b_P + 1
    GOP_num = int(np.floor((args.frame - 1)/GOP_size))

    return Height, Width, batch_size, \
           Channel, activation, GOP_size, GOP_num, \
           path, path_com, path_bin, path_lat
class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=False, peephole=False, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state

