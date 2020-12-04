import tensorflow as tf

STYLE_LAYERS = ['block1_conv1_relu', 'block2_conv1_relu', 'block3_conv1_relu', 'block4_conv1_relu', 'block5_conv1_relu']
CONTENT_LAYER = 'block4_conv2_relu'

@tf.function
def loss(transform_net, model_feature_extractor, content_targets, style_target, batch_size=1):

    style_weight = 1e-8
    feature_reconstruction_weight = 1e-5
    tv_weight = 1e-6

    # calculate loss
    

    # speed up by transforming on original content first instead of using noise
    y_h = transform_net(content_targets)
    
    # prepare content targets, transformed targets (y_h) and style target for vgg19 feature extractor
    content_targets = tf.keras.applications.vgg19.preprocess_input(content_targets)
    content_targets = tf.image.resize(content_targets, (224, 224))
    
    y_h = tf.keras.applications.vgg19.preprocess_input(y_h)
    y_h = tf.image.resize(y_h, (224, 224))

    style_target = tf.keras.applications.vgg19.preprocess_input(style_target)
    style_target = tf.image.resize(style_target, (224, 224))
    
    # get the relevant feature maps
    features_y = model_feature_extractor(content_targets)
    features_y_h = model_feature_extractor(y_h)
    style_target_features_y = model_feature_extractor(style_target)

    # calculate feature reconstruction loss
    content_feature_y = features_y[CONTENT_LAYER]
    content_feature_y_h = features_y_h[CONTENT_LAYER]
    feature_map_size = content_feature_y.shape[1] * content_feature_y.shape[2] * content_feature_y.shape[3]
    feature_reconstruction_loss = tf.nn.l2_loss(content_feature_y - content_feature_y_h) / feature_map_size
    feature_reconstruction_loss *= feature_reconstruction_weight
    feature_reconstruction_loss /= batch_size

    # calculate style reconstruction loss
    style_losses = []
    for style_layer in STYLE_LAYERS:
        style_feature_y = style_target_features_y[style_layer]
        style_feature_y_h = features_y_h[style_layer]
        gram_mat_y = compute_gram_mat(style_feature_y)
        gram_mat_y_h = compute_gram_mat(style_feature_y_h)
        this_style_loss = tf.nn.l2_loss(gram_mat_y_h - gram_mat_y)
        style_losses.append(this_style_loss)
    style_loss = tf.reduce_sum(style_losses)/ len(STYLE_LAYERS) / batch_size
    style_loss *= style_weight

    # calculate total variation loss
    tv_loss = tv_weight* total_variation_loss(y_h) / batch_size
    return feature_reconstruction_loss, style_loss, tv_loss 

@tf.function
def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

@tf.function
def total_variation_loss(image):
    """ 
    compute total variation loss
    """
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

@tf.function
def _tensor_size(tensor):
    return tf.reduce_prod(tensor.get_shape())

@tf.function
def compute_gram_mat(feature_map):
    """
    computes gram matrix
    """
    filters = feature_map.shape[3]
    kernel_w = feature_map.shape[1]
    kernel_h = feature_map.shape[2]
    psi = tf.reshape(feature_map, (feature_map.shape[0], filters, kernel_w * kernel_h))
    gram =  tf.matmul(psi, tf.transpose(psi, perm=[0, 2, 1])) / (filters * kernel_w * kernel_h)
    return gram

@tf.function
def grad(transform_net, model_feature_extractor, content_targets, style_target, batch_size=1):
    """
    given the transform net and feature extractor, content targets and a style target, 
    compute the various losses and the gradient of the sum of the loss values with respect to variables in transform net
    """
    with tf.GradientTape() as tape:
        feature_reconstruction_loss, style_loss, tv_loss = loss(transform_net, model_feature_extractor, content_targets, style_target, batch_size)
        loss_value = feature_reconstruction_loss + style_loss + tv_loss 
    return (feature_reconstruction_loss, style_loss, tv_loss), tape.gradient(loss_value, transform_net.trainable_variables)