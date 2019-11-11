#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def calculate_accuracy(pred, labels):
    """Caluculate accuracy score for classification."""
    if len(pred.shape) > 1:
        pred = tf.argmax(pred, 1, output_type=tf.dtypes.int32)

    # if labels are onehot encoded, take argmax to compare the results
    if len(labels.shape) > 1:
        labels = tf.argmax(labels, 1, output_type=tf.dtypes.int32)

    correct = tf.equal(pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def calculate_accuracy_np(pred, labels):
    """Caluculate accuracy score for classification by numpy.
    This will be used for simplifying calculation when
    tensorflow1.14 is used."""
    if len(pred.shape) > 1:
        pred = np.argmax(pred, 1)

    # if labels are onehot encoded, take argmax to compare the results
    if len(labels.shape) > 1:
        labels = np.argmax(labels, 1)

    correct = np.sum(pred == labels)
    accuracy = correct / len(labels)
    return accuracy
