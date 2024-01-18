import configs.training
import tensorflow as tf
import utils.deeplearning
import os


lr_start = 5e-5
epochs = 80
optimizer = tf.keras.optimizers.Adam()


get_loss = configs.training.get_loss
get_metrics = configs.training.get_metrics


def get_callbacks(model_name, decay_steps):
    callbacks = [
        utils.deeplearning.LRCosineDecay(
            start=lr_start, 
            decay_steps=decay_steps,
            idle_steps=0,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("weights", f"{model_name}_weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join("logs", f"{model_name}_log.csv")
        )
    ]
    return callbacks
