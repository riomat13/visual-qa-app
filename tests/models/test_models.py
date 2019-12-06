#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.models import SequenceGeneratorModel, QuestionAnswerModel


class SequenceGeneratorModelTest(unittest.TestCase):

    def test_shape_of_sequence_generator_model(self):
        batch_size = 4
        units = 16
        vocab_size = 200
        embedding_dim = units
        seq_length = 5

        model = SequenceGeneratorModel(units,
                                       vocab_size,
                                       seq_length)

        x = np.array([1] * batch_size, dtype=np.int8)
        seq = np.random.rand(batch_size, seq_length, embedding_dim)
        features = np.random.rand(batch_size, 49, units)
        hidden = np.zeros((batch_size, units))

        pred, new_hidden, weights = model(x, seq, features, hidden)

        self.assertEqual(pred.shape, (batch_size, vocab_size))
        self.assertEqual(weights.shape, (batch_size, seq_length, 1))
        self.assertEqual(new_hidden.shape, hidden.shape)


class QuestionAnswerModelTest(unittest.TestCase):

    @patch('main.models._models.SequenceGeneratorModel.load_weights')
    @patch('main.models.common.QuestionImageEncoder.load_weights')
    def test_question_answer_model_shape(self, mock_qa_loader, mock_gen_loader):
        batch_size = 4
        in_seq_len = 15
        out_seq_len = 7

        model = QuestionAnswerModel(512, in_seq_len, out_seq_len, 20000, 'what')

        x = np.array([1] * batch_size, dtype=np.int8)
        qs = np.arange(15 * batch_size).reshape(batch_size, in_seq_len)
        imgs = np.random.rand(batch_size, 49, 1024)
        hidden = np.random.rand(batch_size, 512)

        out, w = model(x, qs, imgs, hidden)

        self.assertEqual(out.shape, (batch_size, out_seq_len-1))
        self.assertEqual(w.shape, (batch_size, out_seq_len-1, 15))
