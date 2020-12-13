import tensorflow as tf

STYLE_LAYERS = ['block1_conv1_relu', 'block2_conv1_relu', 'block3_conv1_relu', 'block4_conv1_relu', 'block5_conv1_relu']
CONTENT_LAYER = 'block4_conv2_relu'

@tf.function
def loss(transform_net, model_feature_extractor, content_targets, style_target, batch_size=1):

    style_weight = 1e2
    feature_reconstruction_weight = 7.5e0
    tv_weight = 2e2

    # calculate loss
    

    # speed up by transforming on original content first instead of using noise
    y_h = transform_net(content_targets / 255.0) # preds
    content_targets = tf.keras.applications.vgg19.preprocess_input(content_targets)
    # content_targets = tf.image.resize(content_targets, (224, 224))
    
    y_h = tf.keras.applications.vgg19.preprocess_input(y_h) # preds_pre
    # y_h = tf.image.resize(y_h, (224, 224))

    style_target = tf.keras.applications.vgg19.preprocess_input(style_target) # style_image_pre
    # style_target = tf.image.resize(style_target, (224, 224))
    
    features_y = model_feature_extractor(content_targets) # content_features
    features_y_h = model_feature_extractor(y_h) # net
    style_target_features_y = model_feature_extractor(style_target) # line 34 net

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
        # why does lngstrom divide by gram_may_y.size?
        this_style_loss = tf.nn.l2_loss(gram_mat_y_h - gram_mat_y) / _tensor_size(gram_mat_y)
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
    # return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

@tf.function
def _tensor_size(tensor):
    return tf.cast(tf.reduce_prod(tensor.get_shape()), 'float32')

@tf.function
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    return result / tf.cast(input_shape[1] * input_shape[2] * input_shape[3], 'float32')

@tf.function
def grad(transform_net, model_feature_extractor, content_targets, style_target, batch_size=1):
    with tf.GradientTape() as tape:
        feature_reconstruction_loss, style_loss, tv_loss = loss(transform_net, model_feature_extractor, content_targets, style_target, batch_size)
        loss_value = feature_reconstruction_loss + style_loss + tv_loss 
    return (feature_reconstruction_loss, style_loss, tv_loss), tape.gradient(loss_value, transform_net.trainable_variables)

