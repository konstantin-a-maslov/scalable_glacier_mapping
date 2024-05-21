import tensorflow as tf
import numpy as np


class LRWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, target, start=0.0, verbose=0):
        super(LRWarmup, self).__init__()
        self.steps = 0
        self.warmup_steps = warmup_steps
        self.target = target
        self.start = start
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        self.steps = self.steps + 1

    def on_batch_begin(self, batch, logs=None):
        if self.steps <= self.warmup_steps:
            lr = (self.target - self.start) * (self.steps / self.warmup_steps) + self.start
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print(f"\nLRWarmup callback: set learning rate to {lr}")


class LRRestartsWithCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, start, restart_steps, verbose=0):
        super(LRRestartsWithCosineDecay, self).__init__()
        self.steps = 0
        self.start = start
        self.restart_steps = [0] + restart_steps
        self.stage = 0
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        self.steps = self.steps + 1

    def on_batch_begin(self, batch, logs=None):
        if self.stage >= len(self.restart_steps):
            return

        next_stage_steps = self.restart_steps[self.stage]
        if self.steps < next_stage_steps:
            current_stage_steps = self.restart_steps[self.stage - 1]
            lr = 0.5 * self.start * (1 + np.cos(np.pi * (self.steps - current_stage_steps) / (next_stage_steps - current_stage_steps)))
        else:
            self.stage += 1
            lr = self.start
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print(f"\nLRRestartsWithCosineDecay callback: set learning rate to {lr}")


class LRCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, start, decay_steps, idle_steps=0, verbose=0):
        super(LRCosineDecay, self).__init__()
        self.steps = 0
        self.start = start
        self.idle_steps = idle_steps
        self.decay_steps = decay_steps
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        self.steps = self.steps + 1

    def on_batch_begin(self, batch, logs=None):
        if self.steps < self.idle_steps:
            return
        if self.steps >= self.idle_steps + self.decay_steps:
            return

        lr = 0.5 * self.start * (1 + np.cos(np.pi * (self.steps - self.idle_steps) / self.decay_steps))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print(f"\nLRCosineDecay callback: set learning rate to {lr}")


def make_model_sharpness_aware(model, rho=0.05):
    import types

    def train_step_sam(self, data):
        """
        Heavily inspired by https://www.kaggle.com/code/ashaykatrojwar/sharpness-aware-minimization-with-tensorflow
        """
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
            
        trainable = self.trainable_variables
            
        # first forward and backward passes in initial point
        with tf.GradientTape() as tape:
            y_pred1 = self(x, training=True)
            loss = self.compiled_loss(y, y_pred1, sample_weight=sample_weight, regularization_losses=self.losses)
            
        grad = tape.gradient(loss, trainable)
        grad_norm = tf.linalg.global_norm(grad)
        
        # moving to adversarial point
        adversarial_deltas = []
        for i in range(len(trainable)):
            delta = tf.math.multiply(grad[i], rho / grad_norm)
            trainable[i].assign_add(delta)
            adversarial_deltas.append(delta)
            
        # second forward and backward passes in adversarial point
        with tf.GradientTape() as tape:
            y_pred2 = self(x, training=True) 
            loss = self.compiled_loss(y, y_pred2, sample_weight=sample_weight, regularization_losses=self.losses)

        grad = tape.gradient(loss, trainable)
        
        # moving back to initial point
        for i in range(len(trainable)):
            trainable[i].assign_sub(adversarial_deltas[i])
            
        # update weights
        self.optimizer.apply_gradients(zip(grad, trainable))
        # update metrics and return
        self.compiled_metrics.update_state(y, y_pred1, sample_weight=sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}
    
    model.train_step = types.MethodType(train_step_sam, model)
    return model
