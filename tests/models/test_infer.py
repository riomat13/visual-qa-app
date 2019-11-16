#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.settings import Config
from main.utils.loader import fetch_question_types
from main.utils.preprocess import text_processor
from main.models.infer import (
    _get_q_type_model, _get_y_n_model,
    awake_models,
    predict_question_type,
    PredictionModel,
)


class PredictQuestionTypeTest(unittest.TestCase):

    def test_predicted_shape(self):
        batch_size = 4
        size = 5
        num_classes = len(fetch_question_types())
        seq = np.arange(size).reshape(-1, size)

        model = _get_q_type_model()
        pred = model.predict(seq)

        # result is score of softmax with 81 classes
        self.assertEqual(pred.shape, (1, num_classes))


class PredictYesNoTest(unittest.TestCase):

    @unittest.skip
    def test_model_is_not_implemented(self):
        model = _get_y_n_model()


class SettingUpModelsTest(unittest.TestCase):

    @patch('main.models.infer._get_y_n_model')
    @patch('main.models.infer._get_q_type_model')
    def test_awaking_models(self, mock_qtype, mock_y_n):
        mock_qtype.__name__ = 'mock_qtype'
        mock_y_n.__name__ = 'mock_y_n'
        awake_models()
        mock_qtype.assert_called_once()
        mock_y_n.assert_called_once()
