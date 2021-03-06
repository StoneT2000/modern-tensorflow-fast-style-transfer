import tensorflow as tf
from tensorflow.keras import layers

STDDEV = .1

def residual_layer(net, filters, kernel_size):
    """
    Residual layer
    """
    x = conv_layer(net, filters, kernel_size)
    x = conv_layer(x, filters, kernel_size)
    return net + x

def conv_layer(net, filters = 32, kernel_size = 3, strides = 1, with_relu = True):
    """
    Conv2D layer with batch norm and optional ReLU (default is applied)
    """
    net = layers.Conv2D(filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding='SAME',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=STDDEV, seed=None)
    )(net)
    net = _instance_norm(net)
    if with_relu:
        net = tf.nn.relu(net)
    return net

def conv_transpose_layer(net, filters=32, kernel_size=3, strides=1):
    """
    Conv2D transpose layer that upsamples
    """
    net = layers.Conv2DTranspose(
        filters, 
        kernel_size=kernel_size, 
        strides=strides, 
        padding='SAME',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=STDDEV, seed=None)
    )(net)
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _instance_norm(net):
    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(x=net, axes=[1,2], keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def transform_net():
    """
    feed forward transformation network
    """
    inputs = tf.keras.Input((256, 256, 3))
    x = conv_layer(inputs, 32, 9, 1)
    x = conv_layer(x, 64, 3, 2)
    x = conv_layer(x, 128, 3, 2)
    x = residual_layer(x, 128, 3)
    x = residual_layer(x, 128, 3)
    x = residual_layer(x, 128, 3)
    x = residual_layer(x, 128, 3)
    x = residual_layer(x, 128, 3)
    x = conv_transpose_layer(x, 64, 3, 2)
    x = conv_transpose_layer(x, 32, 3, 2)
    x = conv_layer(x, 3, 9, 1, False)
    outputs = tf.nn.tanh(x) * 150 + 255./2
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def restore_from_checkpoint(path):
    """
    restores transform net model from checkpoint for
    """
    restored_model = transform_net()
    restored_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    checkpoint = tf.train.Checkpoint(opt=restored_optimizer, net=restored_model)
    checkpoint.restore(path)
    return restored_model, restored_optimizer