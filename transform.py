import tensorflow as tf
from tensorflow.keras import layers
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
    net = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=None, padding='SAME')(net)
    net = layers.BatchNormalization()(net)
    if with_relu:
        net = layers.ReLU()(net)
    return net

def conv_transpose_layer(net, filters=32, kernel_size=3, strides=1):
    """
    Conv2D transpose layer that upsamples
    """
     # TODO instance norm it later
    net = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='SAME')(net)
    net = layers.BatchNormalization()(net)
    net = layers.ReLU()(net)
    return net

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
