#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from main.metrics import calculate_accuracy

loss_func = None


@tf.function
def train_cls_step(model, inputs, labels, optimizer,
                   inputs_val=None, labels_val=None,
                   loss='sparse_categorical_crossentropy'):
    """Training step for simple classification.

    Args:
        inputs: list or tuple
            a list of tensors
            multiple input tensors can be passed to given model
        labels: tensor
        optimizer: Optimizer instance
        inputs_val: list or tuple
            if given, evaluate validation accuracy
            other than that, same as `inputs`
        labels_val: tensor
        loss: str or loss function
            loss
    Returns:
        loss: batch loss
        acc: train accuracy
        acc_val: validation accuracy
    """
    global loss_func

    if loss_func is None:
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

    with tf.GradientTape() as tape:
        out = model(*inputs)
        losses = loss_func(labels, out)
        loss = tf.reduce_mean(losses)

    trainables = model.trainable_variables
    gradients = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))

    acc = calculate_accuracy(out, labels)

    if inputs_val is not None and labels_val is not None:
        out_val = model(*inputs_val)
        acc_val = calculate_accuracy(out_val, labels_val)

    return loss, acc, acc_val
