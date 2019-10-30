#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.models import (
    Attention,
    QuestionTypeClassification,
    Encoder,
    Decoder
)

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelOutputShapeTest(unittest.TestCase):
    def test_attention_model(self):
        # feature shape is (128,)
        batch_size = 4
        seq_length = 10

        # encoded sequence shape (batch_size, time_steps, seq_len)
        encoded = np.random.randn(batch_size, seq_length,  128)
        hidden = np.random.randn(batch_size, 128)

        units = 32
        model = Attention(units)
        out, weights = model(encoded, hidden)
        self.assertEqual(out.shape, (batch_size, units))
        self.assertEqual(weights.shape, (batch_size, seq_length, 1))

    def test_questiontypeclassification_model(self):
        batch_size = 4
        input_shape = (batch_size, 10)  # (batch_size, feature)
        units = 64
        vocab_size = 100
        n_classes = 25

        # input sequence
        seqs = np.random.randint(0, vocab_size-1, input_shape)

        model = QuestionTypeClassification(
            128, units, vocab_size, n_classes)
        out = model(seqs)
        self.assertEqual(out.shape, (batch_size, n_classes))

    def test_encoder_model(self):
        batch_size = 4
        image_features = np.random.randn(batch_size, 1024)
        sent_features = np.random.randn(batch_size, 128)

        units = 64
        model = Encoder(units=units)
        out = model(image_features, sent_features)
        self.assertEqual(out.shape, (batch_size, units))

    @patch('main.models.Attention.call')
    def test_decoder_model(self, mock_attention):
        attention_units = 16
        units = 32
        batch_size = 4
        vocab_size = 100
        input_shape = (batch_size, 1)

        input_word = np.random.randint(0, vocab_size-1, input_shape)

        # mocking layers
        # attention return shape (batch_size, seq_length, units)
        mock_attention.return_value = \
            (np.random.randn(batch_size, attention_units),
             np.random.randn(batch_size, attention_units, 1))

        embedding_layer = Mock()
        embedding_layer.return_value = \
            np.random.randn(batch_size, 1, vocab_size)

        model = Decoder(units, vocab_size, embedding_layer)

        out, state, attention_weights = model(input_word, None, None)

        mock_attention.assert_called_once()

        # predicted next one word by softmax
        self.assertEqual(out.shape, (batch_size, vocab_size))
        self.assertEqual(state.shape, (batch_size, units))
