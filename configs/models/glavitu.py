import tensorflow as tf
import models.mapping


model_name = "GlaViTU"
dropout = 0.10
inference_dropout = False

use_deepsupervision = True

model_builder, model_args = models.mapping.GlaViTU, {
    "fusion_activation": tf.nn.leaky_relu,
    "resunet_activation": tf.nn.leaky_relu,
    "last_activation": "softmax", 
    "dropout": dropout, 
    "inference_dropout": inference_dropout, 
    "name": model_name,
    "use_deepsupervision": use_deepsupervision,
}

n_ds_branches = 2 if use_deepsupervision else 1

weights_path = None
