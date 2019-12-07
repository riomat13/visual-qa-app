#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

# ignore tensorflow debug info and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.settings import Config
from main.utils.loader import fetch_question_types
from main.utils.preprocess import text_processor
from main.models.infer import (
    _get_q_type_model,
    _get_y_n_model,
    awake_models,
    predict_question_type,
    predict_yes_or_no,
    PredictionModel,
    convert_output_to_sentence,
)


class PredictionModelTest(unittest.TestCase):

    @patch('main.models.infer._set_weights_by_config')
    def test_get_model_instance(self, mock_setter):
        with self.assertRaises(RuntimeError):
            model = PredictionModel()
            model2 = PredictionModel()

        model = PredictionModel.get_model()
        model2 = PredictionModel.get_model()

        self.assertTrue(model is model2)

    def test_handle_not_understandable_question(self):
        target = 'Can not understand the question'
        model = PredictionModel.get_model()

        test_sents = [
            '',
            '? ? ? *&^@#',
            ';saldfnjvafdlgj',
            'dso njmgf aju'
        ]

        for sent in test_sents:
            res = model.predict(sent, '')
            self.assertEqual(res, target)

    @patch('main.models.infer.predict_yes_or_no')
    @patch('main.models.infer.predict_question_type')
    def test_run_prediction_from_yes_or_no(self,
                                           mock_qtype,
                                           mock_y_n):
        model = PredictionModel.get_model()
        model._processor = Mock()
        sequence = np.array([[1, 2, 3]])
        model._processor.return_value = sequence
        # this will be considered as yes/no type
        mock_qtype.return_value = 0
        target = 'test'
        mock_y_n.return_value = ([0], None)

        res, w, pred_id = model.predict('test', '')
        mock_qtype.assert_called_once_with(sequence)
        mock_y_n.assert_called_once_with(sequence, '')
        self.assertEqual(res, 'yes')

        # should be the same as given type
        self.assertEqual(pred_id, 0)

        # yes/no does not return weights
        self.assertIsNone(w)


class PredictQuestionTypeTest(unittest.TestCase):

    def test_predicted_shape(self):

        batch_size = 4
        size = 5
        seq = np.arange(size).reshape(-1, size)

        model = _get_q_type_model()
        pred = model.predict(seq)

        # result is score of softmax with 9 classes
        self.assertEqual(pred.shape, (1, 9))

    def test_predict_by_function(self):
        seq = np.random.randint(1, 100, (1, 10))
        pred = predict_question_type(seq)
        self.assertTrue(0 <= pred < 9)


class PredictYesNoTest(unittest.TestCase):

    @patch('main.models.infer.load_image')
    def test_run_prediction_and_check_shape(self, mock_loader):
        mock_loader.return_value = \
            np.random.rand(224, 224, 3)
        pred, _ = predict_yes_or_no(np.arange(5).reshape(1, -1), '')
        self.assertEqual(pred.shape, (1,))


class SetUpModelsTest(unittest.TestCase):

    @patch('main.models.infer._get_y_n_model')
    @patch('main.models.infer._get_q_type_model')
    def test_awaking_models(self, mock_qtype, mock_y_n):
        mock_qtype.__name__ = 'mock_qtype'
        mock_y_n.__name__ = 'mock_y_n'
        awake_models((mock_qtype, mock_y_n))
        mock_qtype.assert_called_once()
        mock_y_n.assert_called_once()


class ConvertResultTest(unittest.TestCase):

    def test_handle_invalid_inputs(self):
        test_cases = [
            np.zeros((1, 3, 3, 3)),
            np.zeros((10, 5, 3))
        ]

        for case in test_cases:
            with self.assertRaises(ValueError):
                convert_output_to_sentence(case)

    @patch('main.models.infer.processor.index_word')
    def test_convert_output_to_seq(self, mock_processor):
        mock_processor.__getitem__.side_effect = lambda x: 'test'

        seq_length = 10
        skip_words = 5
        vocab_size = 100

        # first few words should not be selected
        # to avoid being skipped
        left = np.zeros([seq_length, skip_words])
        right = np.random.rand(seq_length, vocab_size-skip_words)
        sample = np.hstack([left, right])

        res = convert_output_to_sentence(sample)
        self.assertEqual(len(res.split()), seq_length)
