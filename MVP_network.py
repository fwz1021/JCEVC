import tensorflow as tf
def res_block(input,layer):
    with tf.variable_scope("res_block_"+ str(layer),reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                                 activation=tf.nn.relu)
        output = input + conv2
        return output
def res_block1(input,layer):
    with tf.variable_scope("res_block1_"+ str(layer),reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                                 activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same", strides=1,
                                 activation=tf.nn.relu)
        output = conv3 + conv2
        return output
def MVP_net(pre_MV):

    with tf.variable_scope("mvp_net", reuse=tf.AUTO_REUSE):
        mv1,mv2,mv3 = tf.unstack(pre_MV, axis=4)
        mv12 = (mv1 + mv2)/2
        mv23 = (mv2 + mv3)/2
        input = tf.concat([mv1, mv2, mv3,mv12,mv23], axis=-1)
        conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[3, 3], padding="same",strides=1,
                                 activation=tf.nn.relu)
        conv2 = res_block(conv1,1)
        conv3 = res_block(conv2,2)
        conv4 = res_block1(conv3,1)
        conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[3, 3], padding="same",strides=1,
                                 activation=tf.nn.relu)
        conv6 = tf.layers.conv2d(inputs=conv5, filters=64, kernel_size=[3, 3], padding="same",strides=1,
                                 activation=tf.nn.relu)
        output = tf.layers.conv2d(inputs=conv6, filters=2 , kernel_size=[3, 3], padding="same",strides=1,
                                 activation=None)

    return output
