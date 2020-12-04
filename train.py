from loss import grad
from model import get_vgg_model
import transform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def parse_image(img, size=(256, 256)):
    return tf.image.resize(img, size)

wave = plt.imread("wave.jpg")
content_1 = plt.imread("chicago.jpg")
content_1 = parse_image(content_1, (256, 256))
content_2 = plt.imread('stata.jpg')
content_2 = parse_image(content_2, (256, 256))
wave = parse_image(wave, (256, 256))

vgg_model = get_vgg_model()
vgg_model_outputs_dir = dict([(layer.name, layer.output) for layer in vgg_model.layers])
extractor = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model_outputs_dir)

content_targets=np.array([content_1, content_2])
style_target=np.expand_dims(wave, axis=0)

def train(content_targets, style_target, transform_net, epochs=1000, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        losses, grads = grad(
        transform_net=transform_net, 
        model_feature_extractor=extractor, 
        content_targets=content_targets, 
        style_target=style_target,
        batch_size=2)
        optimizer.apply_gradients(zip(grads, transform_net.trainable_variables))
        tl = losses[0] + losses[1] + losses[2]
        print("Step: {}, TL: {} FR: {} Style: {} TV: {}".format(optimizer.iterations.numpy(),tl,
                                                  losses[0].numpy(), losses[1].numpy(), losses[2].numpy()))

transform_net = transform.transform_net()
train(content_targets, style_target, transform_net=transform_net, lr=1e-4, epochs=int(1e4))
res = transform_net(content_targets)
plt.imshow(res[0] / 255)
plt.imshow(res[1] / 255)