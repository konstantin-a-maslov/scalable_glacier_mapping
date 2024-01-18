import tensorflow as tf


class PatchExtraction(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtraction, self).__init__()
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
        batch_size = tf.shape(inputs)[0]
        patches_count, depth = tf.shape(patches)[1] * tf.shape(patches)[2], tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, patches_count, depth])
        return patches


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_size):
        super(PatchEmbedding, self).__init__()
        self.projection = tf.keras.layers.Dense(embedding_size)
        
    def call(self, inputs):
        embedded_patches = self.projection(inputs)
        return embedded_patches


class AddClassToken(tf.keras.layers.Layer):
    def __init__(self):
        super(AddClassToken, self).__init__()
        
    # note the use of the `build` method
    # it is useful to define weights/sublayers that are dependent on the input shape
    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token",
            shape=(input_shape[-1], ),
            initializer="random_normal",
            trainable=True
        )
        
    def call(self, inputs):
        patches_with_class_token = tf.map_fn(
            lambda x: tf.concat((x, [self.class_token]), axis=0),
            inputs
        )
        return patches_with_class_token


class AddPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(AddPositionalEmbedding, self).__init__()
        
    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True
        )
        
    def call(self, inputs):
        return inputs + self.positional_embedding


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_size, mlp_size, n_heads, dropout, inference_dropout=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.mhsa = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=embedding_size // n_heads
        )
        self.add = tf.keras.layers.Add()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(mlp_size, activation=tf.nn.gelu)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.inference_dropout = inference_dropout
        self.dense2 = tf.keras.layers.Dense(embedding_size, activation=tf.nn.gelu)
        
    def call(self, inputs):
        # MHSA branch
        x1 = self.norm1(inputs)
        x1 = self.mhsa(x1, x1, training=self.inference_dropout)
        # residual connection 1
        x2 = self.add([inputs, x1])
        # MLP branch
        x3 = self.norm2(x2)
        x3 = self.dense1(x3)
        x3 = self.dropout(x3, training=self.inference_dropout)
        x3 = self.dense2(x3)
        x3 = self.dropout(x3, training=self.inference_dropout)
        # residual connection 2
        outputs = self.add([x2, x3])
        return outputs


class ExtractClassToken(tf.keras.layers.Layer):
    def __init__(self):
        super(ExtractClassToken, self).__init__()
        
    def call(self, inputs):
        return inputs[:, -1, :] # all images, last patch, all features
   