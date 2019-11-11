#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

import numpy as np
import tensorflow as tf

from main.metrics import calculate_accuracy, calculate_accuracy_np


class AccuracyCalculationTest(tf.test.TestCase):

    def test_calculate_accuracy_works(self):
        batch_size = 10
        num_classes = 5
        # this should be calculated argmax first
        input1 = tf.Variable(
            np.random.rand(batch_size, 4)
        )

        _labels = np.zeros((batch_size, num_classes+1), dtype=np.int32)
        for i in range(batch_size):
            _labels[i, random.randint(0, num_classes)] = 1
        labels1 = tf.Variable(_labels)

        score = calculate_accuracy(input1, labels1)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)

        # this does not need to be calculated argmax
        labels2 = tf.Variable(
            np.arange(1, batch_size+1, dtype=np.int32)
        )

        score = calculate_accuracy(input1, labels2)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_calculate_accuracy_with_numpy(self):
        # check if numpy compatibility
        batch_size = 10
        num_classes = 5
        # this should be calculated argmax first
        input1 = \
            np.random.rand(batch_size, 4)

        labels1 = np.zeros((batch_size, num_classes+1))
        for i in range(batch_size):
            labels1[i, random.randint(0, num_classes)] = 1

        score = calculate_accuracy(input1, labels1)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)

        # this does not need to be calculated argmax
        labels2 = np.arange(1, batch_size+1)

        score = calculate_accuracy(input1, labels2)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_calculate_accuracy_np_works(self):
        # check function work only for numpy
        batch_size = 10
        num_classes = 5
        # this should be calculated argmax first
        input1 = \
            np.random.rand(batch_size, 4)

        labels1 = np.zeros((batch_size, num_classes+1))
        for i in range(batch_size):
            labels1[i, random.randint(0, num_classes)] = 1

        score = calculate_accuracy_np(input1, labels1)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)

        # this does not need to be calculated argmax
        labels2 = np.arange(1, batch_size+1)

        score = calculate_accuracy_np(input1, labels2)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)
