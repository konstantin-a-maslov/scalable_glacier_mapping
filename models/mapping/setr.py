import tensorflow as tf
import models.misc
import layers.general


def SETRPUPDecoder(
    input_shape, n_outputs, last_activation="softmax", n_filters=256, name="SETRPUPDecoder", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)

    conv1 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    upsampling1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(bn1)
    conv2 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")(upsampling1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    upsampling2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(bn2)
    conv3 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")(upsampling2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    upsampling3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(bn3)
    conv4 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")(upsampling3)
    bn4 = tf.keras.layers.BatchNormalization()(conv4)
    conv5 = tf.keras.layers.Conv2D(n_outputs, 1, activation=last_activation)(bn4)
    outputs = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(conv5)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def SETR(
    input_shape, n_outputs, last_activation="softmax", patch_size=16, embedding_size=768, mlp_size=3072, 
    n_blocks=12, n_heads=12, dropout=0.1, inference_dropout=False, n_filters=256, name="SETR", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)
    encoded = models.misc.ViTEncoder(
        inputs.shape[1:], 
        use_class_token=False, 
        patch_size=patch_size, 
        embedding_size=embedding_size, 
        mlp_size=mlp_size,
        n_blocks=n_blocks, 
        n_heads=n_heads, 
        dropout=dropout, 
        inference_dropout=inference_dropout
    )(inputs)
    outputs = SETRPUPDecoder(
        encoded.shape[1:],
        n_outputs,
        last_activation=last_activation,
        n_filters=n_filters
    )(encoded)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def FusionSETR(
    input_shapes, n_outputs, last_activation="softmax", fused_size=64, patch_size=16, embedding_size=768, mlp_size=3072, 
    n_blocks=12, n_heads=12, dropout=0.1, inference_dropout=False, n_filters=256, name="FusionSETR", **kwargs
):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = tf.keras.layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    fused = layers.general.FusionBlock(
        fused_size, 
        spatial_dropout=dropout, 
        inference_dropout=inference_dropout
    )(inputs)

    outputs = SETR(
        fused.shape[1:], 
        n_outputs, 
        last_activation=last_activation, 
        patch_size=patch_size, 
        embedding_size=embedding_size, 
        mlp_size=mlp_size, 
        n_blocks=n_blocks, 
        n_heads=n_heads, 
        dropout=dropout, 
        inference_dropout=inference_dropout, 
        n_filters=n_filters
    )(fused)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def SETR_B16(
    input_shape, n_outputs, last_activation="softmax", dropout=0.1, inference_dropout=False, 
    name="SETR-B/16", **kwargs
):
    return SETR(
        input_shape,
        n_outputs,
        last_activation=last_activation,
        patch_size=16,
        embedding_size=768,
        mlp_size=3072,
        n_blocks=12,
        n_heads=12,
        dropout=dropout,
        inference_dropout=inference_dropout,
        n_filters=256,
        name=name,
        **kwargs
    )


def FusionSETR_B16(
    input_shapes, n_outputs, last_activation="softmax", dropout=0.1, inference_dropout=False,
    name="FusionSETR-B/16", **kwargs
):
    return FusionSETR(
        input_shapes,
        n_outputs,
        last_activation=last_activation,
        fused_size=64,
        patch_size=16,
        embedding_size=768,
        mlp_size=3072,
        n_blocks=12,
        n_heads=12,
        dropout=dropout,
        inference_dropout=inference_dropout,
        n_filters=256,
        name=name,
        **kwargs
    )


def SETR_LN16(
    input_shape, n_outputs, last_activation="softmax", dropout=0.1, inference_dropout=False,
    name="SETR-LN/16", **kwargs
):
    return SETR(
        input_shape,
        n_outputs,
        last_activation=last_activation,
        patch_size=16,
        embedding_size=64,
        mlp_size=256,
        n_blocks=24,
        n_heads=4,
        dropout=dropout,
        inference_dropout=inference_dropout,
        n_filters=64,
        name=name,
        **kwargs
    )


def FusionSETR_LN16(
    input_shapes, n_outputs, last_activation="softmax", dropout=0.1, inference_dropout=False,
    name="FusionSETR-LN/16", **kwargs
):
    return FusionSETR(
        input_shapes,
        n_outputs,
        last_activation=last_activation,
        fused_size=32,
        patch_size=16,
        embedding_size=64,
        mlp_size=256,
        n_blocks=24,
        n_heads=4,
        dropout=dropout,
        inference_dropout=inference_dropout,
        n_filters=64,
        name=name,
        **kwargs
    )
