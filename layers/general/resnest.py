"""
This submodule implements a slightly modified version of the ResNeSt architecture based on the 
concepts presented in the following papers and code repository:
- "ResNeSt: Split-Attention Networks" by Zhang et al. (https://arxiv.org/pdf/2004.08955.pdf)
- Official ResNeSt code repository: https://github.com/zhanghang1989/ResNeSt/tree/master

The implementation follows the overall concept presented in the referenced papers and code repository, 
but with minor deviations to maintain consistency with our other models.
Note that in this submodule, the implementation is based on functions rather than inheriting from 
tf.keras.layers.Layer. We found this approach to be more readable and easier to follow here because of
the complexity of the ResNeSt building blocks.

Please refer to the provided references for more detailed information about the ResNeSt architecture 
and its design principles.
"""
import tensorflow as tf


def ResNeStBlock(
    filters, radix=2, cardinality=1, width=64, expansion=4, strides=None, upsampling=None,
    activation=tf.keras.layers.LeakyReLU, dropout=0.0, inference_dropout=False
):
    if radix < 2:
        # We have not used this version of the model and have not tested it properly
        raise NotImplementedError()

    def dropout(x):
        y = x
        if dropout:
            y = tf.keras.layers.SpatialDropout2D(dropout)(y, training=inference_dropout)
        return y

    def block(x):
        residual = x

        if strides and strides != (1, 1) and strides != 1:
            residual = tf.keras.layers.AveragePooling2D(pool_size=strides, padding="same")(residual)
        if upsampling and upsampling != (1, 1) and upsampling != 1:
            residual = tf.keras.layers.UpSampling2D(size=upsampling, interpolation="bilinear")(residual)
        if strides and strides != (1, 1) and strides != 1 or \
            upsampling and upsampling != (1, 1) and upsampling != 1 or \
            filters * expansion != x.shape[-1]:
            residual = tf.keras.layers.Conv2D(
                filters * expansion, 1, padding="same", use_bias=False
            )(residual)

        group_width = width * filters * cardinality // 64 

        conv1 = tf.keras.layers.Conv2D(group_width, 1, use_bias=False)(x)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        act1 = activation()(bn1)
        drop1 = dropout(act1)

        conv2 = tf.keras.layers.Conv2D(
            group_width * radix, 3, padding="same", groups=cardinality * radix, use_bias=False
        )(drop1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        act2 = activation()(bn2)
        drop2 = dropout(act2)

        if strides and strides != (1, 1) and strides != 1:
            drop2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=strides, padding="same")(drop2)
        if upsampling and upsampling != (1, 1) and upsampling != 1:
            drop2 = tf.keras.layers.UpSampling2D(size=upsampling, interpolation="bilinear")(drop2)

        radix_blocks = tf.split(drop2, num_or_size_splits=radix, axis=-1) 
        radix_blocks_sum = tf.add_n(radix_blocks)

        pooled = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(radix_blocks_sum)

        ffn_channels = max(32, group_width * radix // 4)
        conv3 = tf.keras.layers.Conv2D(ffn_channels, 1, groups=cardinality)(pooled)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        act3 = activation()(bn3)

        conv4 = tf.keras.layers.Conv2D(group_width * radix, 1, groups=cardinality)(act3)

        _, _, _, r_channels = conv4.shape
        rsoftmax = tf.reshape(conv4, (-1, cardinality, radix, r_channels // radix // cardinality))
        rsoftmax = tf.transpose(rsoftmax, perm=[0, 2, 1, 3])
        rsoftmax = tf.nn.softmax(rsoftmax, axis=1)
        rsoftmax = tf.reshape(rsoftmax, (-1, 1, 1, r_channels))

        attentions = tf.split(rsoftmax, num_or_size_splits=radix, axis=-1) 
        scaled = tf.add_n([attention * block for attention, block in zip(attentions, radix_blocks)])

        conv5 = tf.keras.layers.Conv2D(filters * expansion, 1, use_bias=False)(scaled)
        bn5 = tf.keras.layers.BatchNormalization()(conv5)
        act5 = activation()(bn5)
        drop5 = dropout(act5)

        return tf.keras.layers.Add()([drop5, residual])
    
    return block
