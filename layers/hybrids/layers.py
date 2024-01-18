import tensorflow as tf


class PatchExtractionKeepingDims(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtractionKeepingDims, self).__init__()
        self.patch_size = patch_size
        
    def call(self, inputs):
        # make patches
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )
        # flatten tensor from two-dimensional grid of patches to one-dimensional
        batch_size, depth = tf.shape(inputs)[0], tf.shape(inputs)[-1]
        patches_count = tf.shape(patches)[1] * tf.shape(patches)[2]
        patches = tf.reshape(
            patches, 
            [batch_size, patches_count, self.patch_size, self.patch_size, depth]
        )
        return patches


class MicroFCN(tf.keras.layers.Layer):
    def __init__(self, embedding_size, activation=tf.nn.gelu):
        super(MicroFCN, self).__init__()
        self.convs = [tf.keras.layers.Conv3D(embedding_size, (1, 3, 3), padding="same") for _ in range(5)]
        self.bns = [tf.keras.layers.BatchNormalization() for _ in range(5)]
        self.acts = [activation for _ in range(5)]
        self.pools = [tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)) for _ in range(2)]
        self.ups = [tf.keras.layers.UpSampling3D(size=(1, 2, 2)) for _ in range(2)]
        self.adds = [tf.keras.layers.Add() for _ in range(2)]

    def call(self, inputs):
        conv0 = self.convs[0](inputs)
        bn0 = self.bns[0](conv0)
        act0 = self.acts[0](bn0)
        pool0 = self.pools[0](act0)
        conv1 = self.convs[1](pool0)
        bn1 = self.bns[1](conv1)
        act1 = self.acts[1](bn1)
        pool1 = self.pools[1](act1)
        conv2 = self.convs[2](pool1)
        bn2 = self.bns[2](conv2)
        act2 = self.acts[2](bn2)
        up0 = self.ups[0](act2)
        add0 = self.adds[0]([up0, act1])
        conv3 = self.convs[3](add0)
        bn3 = self.bns[3](conv3)
        act3 = self.acts[3](bn3)
        up1 = self.ups[1](act3)
        add1 = self.adds[1]([up1, act0])
        conv4 = self.convs[4](add1)
        bn4 = self.bns[4](conv4)
        act4 = self.acts[4](bn4)
        return act4


class AddPositionalEmbeddingKeepingDims(tf.keras.layers.Layer):
    def __init__(self):
        super(AddPositionalEmbeddingKeepingDims, self).__init__()
        
    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(1, input_shape[1], 1, 1, input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        
    def call(self, inputs):
        return inputs + self.positional_embedding


class GlobalPoolingPatchWise(tf.keras.layers.Layer):
    def __init__(self):
        super(GlobalPoolingPatchWise, self).__init__()
        
    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=[2, 3])


class RestorePatchDimensions(tf.keras.layers.Layer):
    def __init__(self):
        super(RestorePatchDimensions, self).__init__()
        
    def build(self, input_shape):
        self.n_patches = input_shape[1]
        self.depth = input_shape[2]

    def call(self, inputs):
        return tf.reshape(inputs, [-1, self.n_patches, 1, 1, self.depth])


class Hybrid4Block(tf.keras.layers.Layer):
    def __init__(self, embedding_size, n_heads, dropout, inference_dropout=False):
        super(Hybrid4Block, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.globalpool = GlobalPoolingPatchWise()
        self.mhsa = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=embedding_size // n_heads
        )
        self.restore = RestorePatchDimensions()
        self.add1 = tf.keras.layers.Add()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.microfcn1 = MicroFCN(embedding_size, activation=tf.nn.gelu)
        # self.microfcn2 = MicroFCN(embedding_size, activation=tf.nn.gelu)
        self.add2 = tf.keras.layers.Add()
        self.dropout = dropout
        self.inference_dropout = inference_dropout

    def build(self, input_shape):
        noise_shape = (input_shape[0], input_shape[1], 1, 1, input_shape[-1])
        self.dropout1 = tf.keras.layers.Dropout(self.dropout, noise_shape=noise_shape)
        # self.dropout2 = tf.keras.layers.Dropout(self.dropout, noise_shape=noise_shape)
        
    def call(self, inputs):
        # MHSA branch
        x1 = self.norm1(inputs)
        x1 = self.globalpool(x1)
        x1 = self.mhsa(x1, x1, training=self.inference_dropout)
        x1 = self.restore(x1)
        # residual connection 1
        x2 = self.add1([inputs, x1])
        # MLP branch
        x3 = self.norm2(x2)
        x3 = self.microfcn1(x3)
        x3 = self.dropout1(x3, training=self.inference_dropout)
        # x3 = self.microfcn2(x3)
        # x3 = self.dropout2(x3, training=self.inference_dropout)
        # residual connection 2
        outputs = self.add2([x2, x3])
        return outputs
   