import tensorflow as tf
import models.misc
import layers.general


def DeepLabV3Plus(
    input_shapes, n_outputs, last_activation="softmax", dropout=0, inference_dropout=False, 
    backbone_builder=models.misc.ResNeSt101, name="DeepLabv3+", **kwargs
):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = tf.keras.layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    fused = layers.general.FusionBlock_design2(
        64, 
        spatial_dropout=dropout, 
        inference_dropout=inference_dropout,
    )(inputs)

    backbone = backbone_builder(
        fused.shape[1:],
        use_stem=False,
        return_low_level=True,
        dropout=dropout,
        inference_dropout=inference_dropout,
    )

    low_level_features, high_level_features = backbone(fused)
    high_level_features = layers.general.DilatedSpatialPyramidPooling(256)(high_level_features)

    outputs = layers.general.DeepLabv3PlusHead(256)([low_level_features, high_level_features])
    outputs = tf.keras.layers.Dense(
        n_outputs, activation=last_activation, name="output"
    )(outputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
