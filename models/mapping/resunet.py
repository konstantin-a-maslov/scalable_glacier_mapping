import tensorflow as tf
import layers.general


def ResUNetEncoder(
    input_shape, n_steps=4, start_n_filters=64, activation=tf.nn.leaky_relu, 
    dropout=0, inference_dropout=False, name="ResUNetEncoder", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)

    outputs = []
    x = inputs
    n_filters = start_n_filters
    for _ in range(n_steps):
        x = layers.general.ResidualWithProjection(
            layers.general.ConvBatchNormAct_x2(
                n_filters, activation=activation, 
                spatial_dropout=dropout, inference_dropout=inference_dropout
            ), 
            n_filters
        )(x)
        outputs.append(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        n_filters *= 2

    last_step = layers.general.ResidualWithProjection(
        layers.general.ConvBatchNormAct_x2(
            n_filters, activation=activation,
            spatial_dropout=dropout, inference_dropout=inference_dropout
        ), 
        n_filters
    )(x)
    outputs.append(last_step)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def ResUNetDecoder(
    input_shape, n_outputs, n_steps=4, activation=tf.nn.leaky_relu, last_activation="softmax", 
    dropout=0, inference_dropout=False, name="ResUNetDecoder", **kwargs
):
    encoded_height, encoded_width, encoded_depth = input_shape
    inputs = []

    input1 = tf.keras.layers.Input(input_shape)
    inputs.append(input1)

    x = input1
    n_filters = encoded_depth // 2
    input_height, input_width = encoded_height * 2, encoded_width * 2
    for _ in range(n_steps):
        upsampling = layers.general.UpConv(n_filters)(x)
        input_i = tf.keras.layers.Input((input_height, input_width, n_filters))
        inputs.append(input_i)
        concat = tf.keras.layers.Concatenate()([upsampling, input_i])
        x = layers.general.ResidualWithProjection(
            layers.general.ConvBatchNormAct_x2(
                n_filters, activation=activation,
                spatial_dropout=dropout, inference_dropout=inference_dropout
            ), 
            n_filters
        )(concat)
        n_filters //= 2
        input_height *= 2
        input_width *= 2
    
    outputs = tf.keras.layers.Dense(n_outputs, activation=last_activation)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def ResUNet(
    input_shape, n_outputs, n_steps=4, start_n_filters=64, activation=tf.nn.leaky_relu, last_activation="softmax", 
    dropout=0, inference_dropout=False, name="ResUNet", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape)
    encoded = ResUNetEncoder(
        input_shape, 
        n_steps=n_steps, 
        start_n_filters=start_n_filters,
        activation=activation,
        dropout=dropout, 
        inference_dropout=inference_dropout
    )(inputs)
    decoded = ResUNetDecoder(
        encoded[-1].shape[1:],
        n_outputs,
        n_steps=n_steps, 
        activation=activation,
        last_activation=last_activation,
        dropout=dropout,
        inference_dropout=inference_dropout
    )(encoded[::-1])

    model = tf.keras.models.Model(inputs=inputs, outputs=decoded, name=name, **kwargs)
    return model


def LightResUNet(
    input_shape, n_outputs, last_activation="softmax", dropout=0, inference_dropout=False, 
    name="LightResUNet", **kwargs
):
    return ResUNet(
        input_shape,
        n_outputs,
        n_steps=3,
        start_n_filters=32,
        last_activation=last_activation,
        dropout=dropout,
        inference_dropout=inference_dropout,
        name=name,
        **kwargs
    )


def FusionResUNet(
    input_shapes, n_outputs, last_activation="softmax", dropout=0, inference_dropout=False, 
    name="FusionResUNet", **kwargs
):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = tf.keras.layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    fused = layers.general.FusionBlock(
        64, 
        spatial_dropout=dropout, 
        inference_dropout=inference_dropout
    )(inputs)

    outputs = ResUNet(
        fused.shape[1:],
        n_outputs,
        last_activation=last_activation,
        dropout=dropout,
        inference_dropout=inference_dropout
    )(fused)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def LightFusionResUNet(
    input_shapes, n_outputs, last_activation="softmax", dropout=0, inference_dropout=False, 
    name="LightFusionResUNet", **kwargs
):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = tf.keras.layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    fused = layers.general.FusionBlock(
        32, 
        spatial_dropout=dropout, 
        inference_dropout=inference_dropout
    )(inputs)

    outputs = LightResUNet(
        fused.shape[1:],
        n_outputs,
        last_activation=last_activation,
        dropout=dropout,
        inference_dropout=inference_dropout
    )(fused)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
