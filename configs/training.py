import tensorflow as tf
import losses
import metrics
import utils.deeplearning
import os


lr_start = 5e-4
# lr_start = 1e-4
restart_epochs = [10, 30, 70, 150]
optimizer = tf.keras.optimizers.Adam()

gamma = 2.0


def get_loss(n_ds_branches=1):
    # loss = [
    #     losses.FocalLoss(gamma=gamma) for _ in range(n_ds_branches)
    # ]
    # loss = [
    #     losses.WeightedLoss(
    #         [losses.FocalLoss(gamma=gamma), losses.DiceLoss(class_idx=1)],
    #         weights=[0.5, 0.5],
    #     )
    #     for _ in range(n_ds_branches)
    # ]
    # loss = [
    #     losses.WeightedLoss(
    #         [losses.FocalLoss(gamma=gamma), losses.TverskyLoss(class_idx=1, alpha=1.0, beta=1.0)],
    #         weights=[0.5, 0.5],
    #     )
    #     for _ in range(n_ds_branches)
    # ]
    loss = [
        losses.WeightedLoss(
            [losses.FocalLoss(gamma=gamma)],
            weights=[0.5 if branch_idx + 1 != n_ds_branches else 1.0],
        )
        for branch_idx in range(n_ds_branches)
    ]
    if n_ds_branches == 1:
        loss = loss[-1]
    return loss


def get_metrics(n_outputs, n_ds_branches=1):
    metrics_list = [
        [
            tf.keras.metrics.CategoricalAccuracy(),
            *[
                tf.keras.metrics.Precision(class_id=class_idx, name=f"precision_{class_idx}") 
                for class_idx in range(n_outputs)
            ],
            *[
                tf.keras.metrics.Recall(class_id=class_idx, name=f"recall_{class_idx}") 
                for class_idx in range(n_outputs)
            ],
            *[
                metrics.IoU(
                    class_idx=class_idx, 
                    name=f"iou_{class_idx}" if n_ds_branches > 1 else f"output_iou_{class_idx}"
                ) 
                for class_idx in range(n_outputs)
            ],
        ] 
        for _ in range(n_ds_branches)
    ]
    if n_ds_branches == 1:
        metrics_list = metrics_list[0]
    return metrics_list


def get_callbacks(model_name, restart_steps, n_outputs):
    callbacks = [
        utils.deeplearning.LRRestartsWithCosineDecay(
            start=lr_start, 
            restart_steps=restart_steps,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("weights", f"{model_name}_minloss_weights.h5"),
            monitor="val_output_loss",
            model="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("weights", f"{model_name}_weights.h5"),
            monitor=f"val_output_iou_{n_outputs - 1}",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join("logs", f"{model_name}_log.csv"),
        )
    ]
    return callbacks
