import tensorflow as tf
import layers.general


class DilatedSpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, dilation_rates=[6, 12, 18], spatial_dropout=0, inference_dropout=False
    ):
        super(DilatedSpatialPyramidPooling, self).__init__()
        self.n_filters = n_filters
        self.dilation_rates = dilation_rates

        self.conv1x1 = layers.general.ConvBatchNormAct(n_filters, kernel_size=1)
        self.proj_pooled = layers.general.ConvBatchNormAct(n_filters, kernel_size=1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.proj_output = layers.general.ConvBatchNormAct(
            n_filters, kernel_size=1, spatial_dropout=spatial_dropout, inference_dropout=inference_dropout
        )

    def build(self, input_shape):
        self.convs3x3 = []
        for dilation_rate in self.dilation_rates:
            self.convs3x3.append(
                layers.general.SeparableConvBatchNormAct(self.n_filters, kernel_size=3, dilation_rate=dilation_rate)
            )

        _, height, width, _ = input_shape
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(height, width))
        self.upsample_pooled = tf.keras.layers.UpSampling2D(size=(height, width), interpolation="bilinear")
        
    def call(self, x):
        pooled_features = self.avg_pool(x)
        pooled_features = self.proj_pooled(pooled_features)
        pooled_features = self.upsample_pooled(pooled_features)

        to_concatenate = [
            pooled_features, 
            self.conv1x1(x),
        ]

        for conv3x3 in self.convs3x3:
            to_concatenate.append(conv3x3(x))

        y = self.concat(to_concatenate)
        y = self.proj_output(y)
        return y


class DeepLabv3PlusHead(tf.keras.layers.Layer):
    def __init__(self, n_filters, upsampling_rates=[4, 4], spatial_dropout=0, inference_dropout=False):
        super(DeepLabv3PlusHead, self).__init__()
        self.proj_low_level = layers.general.ConvBatchNormAct(n_filters, kernel_size=1)
        self.upsample1 = tf.keras.layers.UpSampling2D(size=upsampling_rates[0], interpolation="bilinear")
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv1 = layers.general.SeparableConvBatchNormAct(n_filters, kernel_size=3)
        self.conv2 = layers.general.SeparableConvBatchNormAct(
            n_filters, kernel_size=3, spatial_dropout=spatial_dropout, inference_dropout=inference_dropout
        )
        self.upsample2 = tf.keras.layers.UpSampling2D(size=upsampling_rates[1], interpolation="bilinear")
        self.proj_output = layers.general.ConvBatchNormAct(n_filters, kernel_size=1)

    def call(self, xs):
        low_level_features, high_level_features = xs

        low_level_features = self.proj_low_level(low_level_features)
        high_level_features = self.upsample1(high_level_features)

        y = self.concat([low_level_features, high_level_features])
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.upsample2(y)
        y = self.proj_output(y)
        return y