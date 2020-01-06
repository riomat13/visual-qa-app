#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.disable(logging.CRITICAL)

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.models._models import SequenceGeneratorModel, SequenceGeneratorModel_v2
from main.models import QuestionAnswerModel


class SequenceGeneratorModelTest(unittest.TestCase):

    def test_shape_of_sequence_generator_model(self):
        batch_size = 4
        units = 512
        vocab_size = 200
        # units and embedding_dim must match
        embedding_dim = units
        seq_length = 5

        model = SequenceGeneratorModel(units,
                                       vocab_size,
                                       seq_length,
                                       embedding_dim)

        x = np.random.rand(batch_size, embedding_dim)
        seq = np.random.rand(batch_size, seq_length, embedding_dim)
        features = np.random.rand(batch_size, 49, units)
        hidden = np.zeros((batch_size, units))

        pred, new_hidden = model(x, seq, features, hidden)

        self.assertEqual(pred.shape, (batch_size, vocab_size))
        self.assertEqual(new_hidden.shape, hidden.shape)


class SequenceGeneratorV2ModelTest(unittest.TestCase):

    def test_shape_of_sequence_generator_model(self):
        batch_size = 4
        units = 512
        vocab_size = 200
        # units and embedding_dim must match
        embedding_dim = units
        seq_length = 5

        model = SequenceGeneratorModel_v2(units,
                                          vocab_size,
                                          seq_length,
                                          embedding_dim)

        x = np.random.rand(batch_size, embedding_dim)
        seq = np.random.rand(batch_size, seq_length, embedding_dim)
        features = np.random.rand(batch_size, 49, units)
        hidden = np.zeros((batch_size, units))

        pred, new_hidden = model(x, seq, features, hidden)

        self.assertEqual(pred.shape, (batch_size, vocab_size))
        self.assertEqual(new_hidden.shape, hidden.shape)

class QuestionAnswerModelTest(unittest.TestCase):

    @patch('main.models._models.tf.train.Checkpoint')
    def test_question_answer_model_shape(self, Checkpoint):
        batch_size = 4
        in_seq_len = 10
        out_seq_len = 5

        model = QuestionAnswerModel(512,
                                    in_seq_len,
                                    out_seq_len,
                                    20000,
                                    512,
                                    'what')

        x = np.array([1] * batch_size, dtype=np.int8)
        qs = np.arange(in_seq_len * batch_size).reshape(batch_size, in_seq_len)
        imgs = np.random.rand(batch_size, 49, 1024)
        hidden = np.random.rand(batch_size, 512)

        out, w = model(x, qs, imgs, hidden)

        self.assertEqual(out.shape, (batch_size, out_seq_len-1))

        # currently stop available weights
        #self.assertEqual(w.shape, (batch_size, out_seq_len-1, 15))
