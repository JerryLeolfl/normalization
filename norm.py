import tensorflow as tf
from __future__ import division

  def group_norm(input, group_num):
    # input: input features with shape [batch, rows, cols, channels]
    # group_num: number of groups for GN (Layers norm when group_num = 1, Instance norm when group_num = channels)

    batch, rows, cols, channels = [i.value for i in input.get_shape()]
    var_shape = [group_num]

    input = tf.reshape(input, [batch, group_num, rows, cols, channels // group_num])
    mean, variance = tf.nn.moments(input, [2,3,4], keep_dims=True)
    beta = tf.Variable(tf.zeros(var_shape))
    gamma = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (input-mean)/(variance + epsilon)**(.5)
    normalized = gamma * normalized + beta
    normalized = tf.reshape(normalized, [batch, rows, cols, channels])

    return normalized

def block_norm(input, block_size):
    # input: input features with shape [batch, rows, cols, channels]
    # block_size: size of the block for block norm

    batch, rows, cols, channels = [i.value for i in input.get_shape()]
    var_shape = [(rows // block_size), (cols // block_size)]
    input = tf.reshape(input, [batch, rows // block_size, cols // block_size, block_size, block_size, channels])

    mean, variance = tf.nn.moments(input, [3,4,5], keep_dims=True)
    beta = tf.Variable(tf.zeros(var_shape))
    gamma = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (input-mean)/(variance + epsilon)**(.5)
    normalized = gamma * normalized + beta
    normalized = tf.reshape(normalized, [batch, rows, cols, channels])

    return normalized

def instance_norm(input):

    batch, rows, cols, channels = [i.value for i in input.get_shape()]
    var_shape = [channels]

    mean, variance = tf.nn.moments(input, [1,2], keep_dims=True)
    beta = tf.Variable(tf.zeros(var_shape))
    gamma = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (input-mean)/(variance + epsilon)**(.5)

    return gamma * normalized + beta


