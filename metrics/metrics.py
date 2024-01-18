import tensorflow as tf


class IoU(tf.keras.metrics.Metric):
    def __init__(self, class_idx=0, name="IoU", **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        self.class_idx = class_idx
        self.iou = self.add_weight(name="iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)

        y_pred = tf.cast(y_pred == self.class_idx, self.dtype)
        y_true = tf.cast(y_true == self.class_idx, self.dtype)

        axis = [_ for _ in range(1, len(y_pred.shape))]
        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=axis)
        epsilon = tf.keras.backend.epsilon()
        values = (tp + epsilon) / (tp + fp + fn + epsilon)

        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)

        self.iou.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.iou / self.count

    def reset_state(self):
        self.iou.assign(0.0)
        self.count.assign(0.0)
