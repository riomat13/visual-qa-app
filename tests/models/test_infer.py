#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.utils.loader import fetch_question_types
from main.utils.preprocess import text_processor
from main.models.infer import (
    _get_q_type_model, _get_y_n_model,
    predict_question_type,
    PredictionModel,
)


class PredictQuestionTypeTest(unittest.TestCase):

    def setUp(self):
        self.num_classes = len(fetch_question_types())
        self.processor = text_processor(from_config=True)

    def test_predicted_shape(self):
        batch_size = 4
        sentence = 'this is a test sentence'
        seq = self.processor(sentence)

        model = _get_q_type_model()
        pred = model.predict(seq)

        # result is score of softmax with 81 classes
        self.assertEqual(pred.shape, (1, self.num_classes))


class PredictYesNoTest(unittest.TestCase):

    def test_model_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            model = _get_y_n_model()
