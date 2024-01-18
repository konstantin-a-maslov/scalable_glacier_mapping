import tensorflow as tf
import layers.general


def ResNeSt(
    input_shape, n_layers, n_filters=[64, 128, 256, 512], radix=2, cardinality=1, width=64, 
    use_stem=True, return_low_level=False, stem_width=64, activation=tf.keras.layers.LeakyReLU, 
    dropout=0.0, inference_dropout=False, name="ResNeSt_backbone", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)

    x = inputs
    if use_stem:
        for strides, filters in zip([(2, 2), (1, 1), (1, 1)], [stem_width, stem_width, 2 * stem_width]):
            x = tf.keras.layers.Conv2D(
                filters, 3, strides=strides, padding="same", use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = activation()(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x) 

    for _layers, _filters in zip(n_layers, n_filters):
        for block_idx in range(_layers):
            x = layers.general.ResNeStBlock(
                _filters, 
                radix=radix, 
                cardinality=cardinality, 
                width=width, 
                strides=(2, 2) if block_idx == _layers - 1 and _filters != n_filters[-1] else (1, 1), 
                activation=activation,
                dropout=dropout, 
                inference_dropout=inference_dropout
            )(x)

            if return_low_level and _filters == n_filters[1] and block_idx == _layers - 2:
                low_level_features = x

    outputs = x if not return_low_level else [low_level_features, x]
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def ResNeSt50(input_shape, name="ResNeSt-50", **kwargs):
    return ResNeSt(input_shape, [3, 4, 6, 3], stem_width=32, name=name, **kwargs)


def ResNeSt101(input_shape, name="ResNeSt-101", **kwargs):
    return ResNeSt(input_shape, [3, 4, 23, 3], name=name, **kwargs)
