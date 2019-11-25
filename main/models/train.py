#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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


def make_training_seq_model(model, sequence_length, optimizer,
                            encoder_model=None,
                            loss='sparse_categorical_crossentropy'):
    """Build training step function for sequence generator.

    Args:
        model: tf.keras.models.Model
            base model to train
        sequence_length: int
            sequence length to generate
            this is fixed value so that even if end of sentence
            token is generated, it will generate something until
            reach the length
        optimizer: tf.keras.optimizers
        encoder_model: tf.keras.models.Model
            if encode inside this function,
            provide the model
        loss: str or tf.keras.losses
    Return:
        function:
            run training step

            methods:
                get_encoded_outputs
                    Returns: tuple

                    if encoder_model is provided
                    it can return outputs by it,
                    which is, the outputs generate several data,
                    outputs[0] will be used for next step and
                    this method returns outputs[1:]

                    this method can be used only when
                    encoder_model is provided.
    """
    embedding_dim = model.embedding.output_dim

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
    def train_seq_step(x, inputs, labels):
        """Run training step.
        Args:
            x: initial word of generated sequence
            inputs, labels: training set, inputs must be a list or tuple
        Returns:
            loss: float
            predicts:
            attention_weights: list of attention weights of sequence
                each index in the list represents weights used to
                predict the word in output sequence
                attention_weights[0] is the first words and
                attention_weights[3] is the forth words
        """
        nonlocal model
        nonlocal embedding_dim
        nonlocal sequence_length
        nonlocal optimizer
        nonlocal encoder_model
        nonlocal loss_func

        with tf.GradientTape() as tape:
            features = encoder_model(*inputs)

            if isinstance(features, tuple):
                features, *_ = features

            hidden = np.zeros((len(labels), embedding_dim))

            loss = 0
            predicts = []
            attention_weights = []

            for i in range(1, sequence_length):
                pred, hidden, weights = model(x, inputs[0], features, hidden)
                cost = loss_func(labels[:, i], pred, from_logits=True, axis=-1)
                x = labels[:, i]
                loss += tf.reduce_mean(cost)
                predicts.append(pred)
                attention_weights.append(weights)

        trainables = model.trainable_variables
        if encoder_model:
            trainables += encoder_model.trainable_variables
        gradients = tape.gradient(loss, trainables)
        optimizer.apply_gradients(zip(gradients, trainables))

        return loss, predicts, attention_weights

    if encoder_model is None:
        # if encoder_model is not provided
        # identity function will be used
        encoder_model = lambda x: x

    return train_seq_step
