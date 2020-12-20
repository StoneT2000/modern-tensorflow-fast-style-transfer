import tensorflow as tf

STYLE_LAYERS = ['block1_conv1_relu', 'block2_conv1_relu', 'block3_conv1_relu', 'block4_conv1_relu', 'block5_conv1_relu']
CONTENT_LAYER = 'block4_conv2_relu'

@tf.function
def loss(transform_net, model_feature_extractor, content_targets, style_target, batch_size=1):
    """
    Calculate the losses of the transformation network using the vgg19 feature extractor and given content images and style image
    """


    style_weight = 1e0
    feature_reconstruction_weight = 1e-1
    tv_weight = 2e0    

    # speed up by transforming on original content first instead of using noise
    y_h = transform_net(content_targets / 255.0) # preds
    content_targets = tf.keras.applications.vgg19.preprocess_input(content_targets)
    
    y_h = tf.keras.applications.vgg19.preprocess_input(y_h)

    style_target = tf.keras.applications.vgg19.preprocess_input(style_target)
    
    features_y = model_feature_extractor(content_targets)
    features_y_h = model_feature_extractor(y_h) 
    style_target_features_y = model_feature_extractor(style_target)

    # calculate feature reconstruction loss
    content_feature_y = features_y[CONTENT_LAYER]
    content_feature_y_h = features_y_h[CONTENT_LAYER]
    feature_map_size = content_feature_y.shape[1] * content_feature_y.shape[2] * content_feature_y.shape[3]
    feature_reconstruction_loss = feature_reconstruction_weight * (2 * tf.nn.l2_loss(content_feature_y_h - content_feature_y)) / feature_map_size
    feature_reconstruction_loss /= batch_size
    


    # calculate style reconstruction loss
    style_losses = []
    for style_layer in STYLE_LAYERS:
        style_feature_y = style_target_features_y[style_layer]
        style_feature_y_h = features_y_h[style_layer]
        gram_mat_y = gram_matrix(style_feature_y)
        gram_mat_y_h = gram_matrix(style_feature_y_h)
        this_style_loss = tf.nn.l2_loss(gram_mat_y_h - gram_mat_y)
        style_losses.append(this_style_loss)
    style_loss = 2 * tf.reduce_sum(style_losses) * style_weight / batch_size

    tv_loss = tv_weight * 2 * total_variation_loss(y_h) / batch_size
    return feature_reconstruction_loss, style_loss, tv_loss 

@tf.function
def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return tf.nn.l2_loss(x_var), tf.nn.l2_loss(y_var)

@tf.function
def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return x_deltas / _tensor_size(image[:,:,1:,:]) + y_deltas / _tensor_size(image[:,1:,:,:])

@tf.function
def _tensor_size(tensor):
    """
    Returns size of tensor, nice function borrowed from https://github.com/lengstrom/fast-style-transfer
    """
    return tf.cast(tf.reduce_prod(tensor.get_shape()), 'float32')

@tf.function
def gram_matrix(input_tensor):
    """
    compute gram matrix
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    return result / tf.cast(input_shape[1] * input_shape[2] * input_shape[3], 'float32')

@tf.function
def grad(transform_net, model_feature_extractor, content_targets, style_target, batch_size=1):
    """
    Using TF gradient tape to compute the graddients and return them, along with the 3 losses
    """
    with tf.GradientTape() as tape:
        feature_reconstruction_loss, style_loss, tv_loss = loss(transform_net, model_feature_extractor, content_targets, style_target, batch_size)
        loss_value = feature_reconstruction_loss + style_loss + tv_loss 
    return (feature_reconstruction_loss, style_loss, tv_loss), tape.gradient(loss_value, transform_net.trainable_variables)

