import tensorflow as tf
from tensorflow.keras import layers

def get_vgg_model():
    # TODO don't include top, we don't use it anyway
    vgg_model = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax'
    )

    # create VGG model from the keras one but with explicit relu layers
    inputs = layers.Input((224,224 ,3))
    x = inputs
    for vgg_layer in vgg_model.layers:
        layer_name = vgg_layer.name
        if 'input' in layer_name:
            continue
        if 'conv' in layer_name:
            new_layer = layers.Conv2D(
                filters=vgg_layer.filters,
                kernel_size=vgg_layer.kernel_size,
                padding=vgg_layer.padding,
                name = layer_name
            )
            relu = layers.ReLU(name=layer_name + '_relu')
            x = new_layer(x)
            x = relu(x)
        elif 'pool' in layer_name:
            new_layer = layers.MaxPool2D(pool_size=vgg_layer.pool_size, strides=vgg_layer.strides, padding=vgg_layer.padding, name=layer_name)
            x = new_layer(x)
    
    new_vgg_model = tf.keras.Model(inputs=inputs, outputs=x, name="VGG19")

    # import weights from original vgg model
    for vgg_layer in vgg_model.layers:
        layer_name = vgg_layer.name
        if 'input' in layer_name:
            continue
        new_vgg_model.get_layer(layer_name).set_weights(vgg_layer.get_weights())
    
    del vgg_model
    return new_vgg_model

def parse_image(img, size=(224, 224)):
    return tf.image.resize(img, size)
def process_image_for_vgg_model(img):
    processed = tf.keras.applications.vgg19.preprocess_input(img)
    return parse_image(processed)