import tensorflow as tf

STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER = 'relu4_2'

@tf.function
def loss(transform_net, model_feature_extractor, x):

    style_weight = 1e-4
    feature_reconstruction_weight = 1e-7

    # calculate loss

    features_y = model_feature_extractor(x)

    # speed up by transforming on original content first instead of using noise
    y_h = transform_net(x)
    features_y_h = model_feature_extractor(y_h)

    # calculate feature reconstruction loss
    content_feature_y = features_y[CONTENT_LAYER]
    content_feature_y_h = features_y_h[CONTENT_LAYER]
    feature_map_size = content_feature_y.shape[1] * content_feature_y.shape[2] * content_feature_y.shape[3]
    feature_reconstruction_loss = tf.nn.l2_loss(content_feature_y - content_feature_y_h) / feature_map_size
    feature_reconstruction_loss *= feature_reconstruction_weight

    # calculate style reconstruction loss
    style_losses = []
    for style_layer in STYLE_LAYERS:
        style_feature_y = features_y[style_layer]
        style_feature_y_h = features_y_h[style_layer]
        gram_mat_y = compute_gram_mat(style_feature_y)
        gram_mat_y_h = compute_gram_mat(style_feature_y_h)
        this_style_loss = tf.nn.l2_loss(gram_mat_y_h - gram_mat_y)
        style_losses.append(this_style_loss)
    style_loss = tf.reduce_sum(style_losses)
    style_loss *= style_weight

    tv_loss = 0

    return feature_reconstruction_loss + style_loss + tv_loss 

@tf.function
def compute_gram_mat(feature_map):
    filters = feature_map.shape[3]
    kernel_w = feature_map.shape[1]
    kernel_h = feature_map.shape[2]
    psi = tf.reshape(feature_map, (filters, kernel_w * kernel_h))
    gram =  tf.matmul(psi, tf.transpose(psi)) / (filters * kernel_w * kernel_h)
    return gram