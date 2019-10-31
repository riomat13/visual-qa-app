#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

from main.metrics import calculate_accuracy

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AccuracyCalculationTest(tf.test.TestCase):

    def test_calculate_accuracy_works(self):
        batch_size = 10
        num_classes = 5
        # this should be calculated argmax first
        input1 = tf.Variable(
            np.random.rand(batch_size, 4)
        )
        labels1 = tf.Variable(
            np.random.randint(0, 1, (batch_size, num_classes+1), dtype=np.int32)
        )

        score = calculate_accuracy(input1, labels1)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)

        # this does not need to be calculated argmax
        labels2 = tf.Variable(
            np.random.randint(0, 1, (batch_size,), dtype=np.int32)
        )

        score = calculate_accuracy(input1, labels2)

        self.assertGreater(score, 0)
        self.assertLess(score, 1)
