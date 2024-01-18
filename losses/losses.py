import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(
        self, gamma=2.0, alphas=None, reduction=tf.keras.losses.Reduction.AUTO, name="focal_loss", **kwargs
    ):
        super(FocalLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.gamma = gamma
        self.alphas = alphas

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1 - epsilon)
        ce = y_true * tf.math.log(y_pred)
        loss = -tf.math.pow(1 - y_pred, self.gamma) * ce
        if self.alphas is not None:
            loss = self.alphas * loss
        focal_loss = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=-1))
        return focal_loss


class DiceLoss(tf.keras.losses.Loss):
    def __init__(
        self, class_idx, reduction=tf.keras.losses.Reduction.AUTO, name="dice_loss", **kwargs
    ):
        super(DiceLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.class_idx = class_idx

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true_class = y_true[..., self.class_idx]
        y_pred_class = y_pred[..., self.class_idx]
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        dice_coeff = (2.0 * intersection + epsilon) / (tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) + epsilon)
        dice_loss = tf.math.reduce_mean(1.0 - dice_coeff)
        return dice_loss


class TverskyLoss(tf.keras.losses.Loss):
    def __init__(
        self, class_idx, alpha=1.0, beta=1.0, reduction=tf.keras.losses.Reduction.AUTO, name="tversky_loss", **kwargs
    ):
        super(TverskyLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.class_idx = class_idx
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true_class = y_true[..., self.class_idx]
        y_pred_class = y_pred[..., self.class_idx]

        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        false_positives = tf.reduce_sum(y_pred_class * (1 - y_true_class))
        false_negatives = tf.reduce_sum(y_true_class * (1 - y_pred_class))

        tversky_coeff = (intersection + epsilon) / (intersection + self.alpha * false_positives + self.beta * false_negatives + epsilon)
        tversky_loss = tf.math.reduce_mean(1.0 - tversky_coeff)
        return tversky_loss


class WeightedLoss(tf.keras.losses.Loss):
    def __init__(
        self, losses, weights=None, reduction=tf.keras.losses.Reduction.AUTO, name="weighted_loss", **kwargs
    ):
        super(WeightedLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.losses = losses
        if weights is None:
            weights = [1 for _ in losses]
        if len(losses) != len(weights):
            raise ValueError()
        self.weights = weights

    def call(self, y_true, y_pred):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += (loss(y_true, y_pred) * weight)
        return total_loss
