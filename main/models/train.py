#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from main.metrics import calculate_accuracy


def make_training_cls_model(model, optimizer,
                            loss='sparse_categorical_crossentropy'):
    """Build training step function for classification.

    Args:
        model: tf.keras.models.Model
            base model to train
        optimizer: tf.keras.optimizers
        loss: str or tf.keras.losses
    Returns:
        function:
            run training step
    """
    if loss == 'sparse_categorical_crossentropy':
        loss_func = tf.keras.losses.sparse_categorical_crossentropy
    elif loss == 'categorical_crossentropy':
        loss_func = tf.keras.losses.categorical_crossentropy
    elif callable(loss) and \
            loss.__module__ == 'tensorflow.python.keras.losses':
        loss_func = loss
    else:
        ValueError('Invalid loss function is given. '
                   'Choose from `sparse_categorical_crossentropy` '
                   'or `categorical_crossentropy`')

    @tf.function
    def train_cls_step(inputs, labels):
        """Training step for simple classification.

        Args:
            inputs: list or tuple
                a list of tensors
                multiple input tensors can be passed to given model
            labels: tensor
        Returns:
            loss: batch loss
            acc: train accuracy
        """
        nonlocal model
        nonlocal optimizer
        nonlocal loss_func

        with tf.GradientTape() as tape:
            out = model(*inputs)
            if isinstance(out, tuple):
                out = out[0]
            cost = loss_func(labels, out, from_logits=True)
            loss = tf.reduce_mean(cost)

        trainables = model.trainable_variables
        gradients = tape.gradient(loss, trainables)
        optimizer.apply_gradients(zip(gradients, trainables))

        acc = calculate_accuracy(out, labels)

        return loss, acc
    return train_cls_step
