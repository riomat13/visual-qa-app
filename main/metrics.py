#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def calculate_accuracy(pred, labels):
    """Caluculate accuracy score for classification."""
    pred = tf.argmax(pred, 1, output_type=tf.dtypes.int32)

    # if labels are onehot encoded, take argmax to compare the results
    if len(labels.shape) > 1:
        labels = tf.argmax(labels, 1, output_type=tf.dtypes.int32)

    correct = tf.equal(pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
