import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="VALID"):
        super(Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.w_re = self.add_weight(
            name="w_re",
            shape=(self.kernel_size, self.kernel_size, input_shape[3], self.filters),
            initializer="random_normal",
            trainable=True
        )
        self.w_im = self.add_weight(
            name="w_im",
            shape=(self.kernel_size, self.kernel_size, input_shape[3], self.filters),
            initializer="random_normal",
            trainable=True
        )
        self.b_re = self.add_weight(
            name="b_re",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True
        )
        self.b_im = self.add_weight(
            name="b_im",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)

        y_re = tf.nn.conv2d(input=x_re, filters=self.w_re, strides=1, padding=self.padding)
        y_re -= tf.nn.conv2d(input=x_im, filters=self.w_im, strides=1, padding=self.padding)
        y_re += self.b_re

        y_im = tf.nn.conv2d(input=x_re, filters=self.w_im, strides=1, padding=self.padding)
        y_im += tf.nn.conv2d(input=x_im, filters=self.w_re, strides=1, padding=self.padding)
        y_im += self.b_im

        y = tf.stack([y_re, y_im], axis=-1)
        return y


class PartWiseActivation(tf.keras.layers.Layer):
    def __init__(self, activation_function):
        super(PartWiseActivation, self).__init__()
        self.activation_function = activation_function

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)

        y_re = self.activation_function(x_re)
        y_im = self.activation_function(x_im)

        y = tf.stack([y_re, y_im], axis=-1)
        return y


class PartWiseBatchNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(PartWiseBatchNormalization, self).__init__()
        self.batch_norm_re = tf.keras.layers.BatchNormalization()
        self.batch_norm_im = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)

        y_re = self.batch_norm_re(x_re)
        y_im = self.batch_norm_im(x_im)

        y = tf.stack([y_re, y_im], axis=-1)
        return y


class PartWiseAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, padding="VALID"):
        super(PartWiseAveragePooling2D, self).__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)

        y_re = tf.nn.avg_pool2d(x_re, self.ksize, self.strides, self.padding)
        y_im = tf.nn.avg_pool2d(x_im, self.ksize, self.strides, self.padding)

        y = tf.stack([y_re, y_im], axis=-1)
        return y

    
class UpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size, interpolation="bilinear"):
        super(UpSampling2D, self).__init__()
        self.upsampling_layer = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)

        y_re = self.upsampling_layer(x_re)
        y_im = self.upsampling_layer(x_im)

        y = tf.stack([y_re, y_im], axis=-1)
        return y


class Concatenate(tf.keras.layers.Layer):
    def __init__(self):
        super(Concatenate, self).__init__()

    def call(self, x):
        x_re, x_im = [], []

        for item in x:
            item_re, item_im = tf.unstack(item, axis=-1)
            x_re.append(item_re)
            x_im.append(item_im)

        y_re = tf.concat(x_re, axis=-1)
        y_im = tf.concat(x_im, axis=-1)

        y = tf.stack([y_re, y_im], axis=-1)
        return y


class Modulus(tf.keras.layers.Layer):
    def __init__(self):
        super(Modulus, self).__init__()

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)
        y = tf.math.sqrt(tf.math.square(x_re) + tf.math.square(x_im))
        return y

    
class ModulusSquared(tf.keras.layers.Layer):
    def __init__(self):
        super(ModulusSquared, self).__init__()

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)
        y = tf.math.square(x_re) + tf.math.square(x_im)
        return y


class ComplexToRealByConcatenation(tf.keras.layers.Layer):
    def __init__(self):
        super(ComplexToRealByConcatenation, self).__init__()

    def call(self, x):
        x_re, x_im = tf.unstack(x, axis=-1)
        y = tf.concat([x_re, x_im], axis=-1)
        return y
