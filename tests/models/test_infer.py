#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

# ignore tensorflow debug info and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.CRITICAL)

import unittest
from unittest.mock import patch, MagicMock

from main.models._models import (
    QuestionTypeClassification,
    ClassificationModel,
    QuestionAnswerModel,
)

mobilenet_encoder = patch('main.models.common.get_mobilenet_encoder')
mobilenet_encoder.start()

QTypeClsModel = patch('main.models._models.QuestionTypeClassification')
QTypeClsModel.start()

ClsModel = patch('main.models._models.ClassificationModel')
ClsModel.start()

QAModel = patch('main.models._models.QuestionAnswerModel')
QAModel.start()

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


def mock_processor(arr=None):
    if arr is None:
        arr = np.array([[1, 2, 3]], dtype=np.int32)

    def processor(*args, **kwargs):
        nonlocal arr
        return lambda x: arr
    return processor


@patch('main.models.infer.awake_models', new=MagicMock())
@patch('main.models.infer._get_what_model', new=MagicMock())
@patch('main.models.infer._get_y_n_model', new=MagicMock())
@patch('main.models.infer._get_q_type_model', new=MagicMock())
class PredictionModelTest(unittest.TestCase):

    @patch('main.models.infer._set_weights_by_config')
    @patch('main.models.infer.text_processor')
    def test_get_model_instance(self, processor, mock_setter):

        # should be called by get_model() for singleton
        with self.assertRaises(RuntimeError):
            model = PredictionModel()
            model2 = PredictionModel()

        model = PredictionModel.get_model()
        model2 = PredictionModel.get_model()

        self.assertTrue(model is model2)

    @patch('main.models.infer.predict_what', new=MagicMock())
    @patch('main.models.infer.text_processor')
    def test_handle_not_understandable_question(self, processor):
        target = 'Can not understand the question'
        model = PredictionModel.get_model()

        # if processor could not handle and return empty
        m = MagicMock()
        m.return_value = np.array([], dtype=np.int32)
        processor.side_effect = m

        # override processor
        PredictionModel._PredictionModel__instance._processor = processor

        res = model.predict('test', '')
        self.assertEqual(res, target)

    @patch('main.models.infer.predict_yes_or_no')
    @patch('main.models.infer.predict_question_type')
    @patch('main.models.infer._set_weights_by_config', new=MagicMock())
    @patch('main.models.infer.text_processor')
    def test_run_prediction_from_yes_or_no(self, processor,
                                           mock_pred, target_pred):
        seq = np.array([[1, 2, 3]], dtype=np.int32)
        m = MagicMock()
        m.return_value = seq
        processor.side_effect = m

        model = PredictionModel.get_model()

        # override processor
        PredictionModel._PredictionModel__instance._processor = processor

        mock_pred.return_value = 0  # classes (yes, no or others)
        target_pred.return_value = ([0], [])  # (result, weights)

        model.predict('test', 'path')

        mock_pred.assert_called_once_with(seq)
        target_pred.assert_called_once_with(seq, 'path')


class PredictQuestionTypeTest(unittest.TestCase):

    @patch('main.models.infer._set_weights_by_config')
    @patch('main.models.infer.QuestionTypeClassification')
    def test_predicted_shape(self, Model, set_weights):

        batch_size = 4
        size = 5
        seq = np.arange(size).reshape(-1, size)

        target = 'test'

        Model.return_value.predict.return_value = target

        model = _get_q_type_model()
        pred = model.predict(seq)

        self.assertEqual(model, Model.return_value)
        self.assertEqual(pred, target)

        cfg = Config.MODELS['QTYPE']
        Model.assert_called_once_with(
            embedding_dim=cfg.get('embedding_dim'),
            units=cfg.get('units'),
            vocab_size=cfg.get('vocab_size'),
            num_classes=9
        )

        set_weights.assert_called_once_with('QTYPE', Model.return_value)

    @patch('main.models.infer._get_q_type_model')
    def test_predict_by_function(self, q_model):
        m = MagicMock()
        q_model.return_value = m

        res = np.linspace(0.1, 0.5, num=5).reshape(1, -1)
        m.predict.return_value = res

        pred = predict_question_type('')
        q_model.assert_called_once()

        # should be take argmax as the result of word
        self.assertEqual(pred, np.argmax(res, axis=1))


@patch('main.models.infer.awake_models', new=MagicMock())
class PredictYesNoTest(unittest.TestCase):

    @patch('main.models.infer._set_weights_by_config')
    @patch('main.models.infer.load_image')
    @patch('main.models.infer.ClassificationModel')
    def test_run_prediction_and_check_shape(self, Model, mock_loader, set_weights):
        mock_loader.return_value = \
            np.random.rand(224, 224, 3)

        res = np.linspace(0.1, 0.5, num=5).reshape(1, -1)
        Model.return_value.return_value = res, ''

        pred, _ = predict_yes_or_no([], '')
        self.assertEqual(pred.shape, (1,))

        set_weights.assert_called_once_with('Y/N', Model.return_value)


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
    @patch('main.models._models.ClassificationModel')
    def test_convert_output_to_seq(self, Model, mock_processor):
        mock_processor.__getitem__.side_effect = lambda x: 'test'

        seq_length = 10
        skip_words = 5
        vocab_size = 100

        # first few words should not be selected
        # to avoid being skipped
        left = np.zeros([seq_length, skip_words])
        right = np.random.rand(seq_length, vocab_size-skip_words)
        sample = np.hstack([left, right])

        Model.return_value.return_value = '', ''

        res = convert_output_to_sentence(sample)
        self.assertEqual(len(res.split()), seq_length)
