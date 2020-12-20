from loss import loss
from model import get_vgg_model
import transform
import tensorflow as tf
import matplotlib.pyplot as plt

transform_net = transform.transform_net()
vgg_model = get_vgg_model()
vgg_model_outputs_dir = dict([(layer.name, layer.output) for layer in vgg_model.layers])
extractor = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model_outputs_dir)
wave = plt.imread("images/style/wave.jpg");
content_1 = plt.imread("images/content/chicago.jpg");
loss(transform_net=transform_net, model_feature_extractor=extractor, content_targets=[content_1], style_targets=wave, batch_size=1)