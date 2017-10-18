import tensorflow as tf

def AdaIN(content_features, style_features, style_weight):
    with tf.variable_scope("adaptive_instance_normalization"):
        style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
        epsilon = 1e-5
        normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                content_variance, style_mean,
                                                                tf.sqrt(style_variance), epsilon)
        normalized_content_features = style_weight * normalized_content_features + (1 - style_weight) * content_features
        return normalized_content_features
# end
