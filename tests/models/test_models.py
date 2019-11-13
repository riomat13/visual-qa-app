#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.models import (
    get_mobilenet_encoder,
    Attention,
    QuestionTypeClassification,
    Encoder,
    Decoder
)


class MobileNetEncoderTest(unittest.TestCase):

    def test_mobilenet_encoder(self):
        batch_size = 4
        model = get_mobilenet_encoder()

        sample_normalized_imgs = \
            np.random.randint(0, 255,
                              size=(batch_size, 224, 224, 3)) \

        sample_normalized_imgs = sample_normalized_imgs.astype(np.float32)

        out = model(sample_normalized_imgs)

        self.assertEqual(out.shape, (batch_size, 7, 7, 1024))


class AttentionModelTest(unittest.TestCase):

    def test_attention_model(self):
        # feature shape is (128,)
        batch_size = 4
        seq_length = 10
        hidden_units = 128

        # encoded sequence shape (batch_size, time_steps, seq_len)
        features = np.random.randn(batch_size, seq_length, hidden_units)
        states = np.random.randn(batch_size, hidden_units)

        # different unit size from input
        units = 32
        model = Attention(units)
        out, weights = model(features, states)
        self.assertEqual(out.shape, (batch_size, units))
        self.assertEqual(weights.shape, (batch_size, seq_length, 1))


class QuestionTypeClassificationTest(unittest.TestCase):

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


class EncoderTest(unittest.TestCase):

    def test_encoder_model(self):
        batch_size = 4
        image_features = np.random.randn(batch_size, 1024)
        sent_features = np.random.randn(batch_size, 128)

        units = 64
        model = Encoder(units=units)
        out = model(image_features, sent_features)
        self.assertEqual(out.shape, (batch_size, units))


class DecoderTest(unittest.TestCase):

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

    @patch('main.models.Attention.call')
    def test_decoder_model_without_reusing_embeddings(self, mock_attention):
        attention_units = 16
        units = 32
        batch_size = 4
        vocab_size = 100
        input_shape = (batch_size, 1)

        input_word = np.random.randint(0, vocab_size-1, input_shape)

        # mocking layers
        # attention return shape (batch_size, seq_length, units)
        mock_attention.return_value = \
            (np.random.randn(batch_size, attention_units)
                .astype(np.float32),
             np.random.randn(batch_size, attention_units, 1)
                .astype(np.float32))

        model = Decoder(units, vocab_size)

        out, state, attention_weights = model(input_word, None, None)

        mock_attention.assert_called_once()

        # predicted next one word by softmax
        self.assertEqual(out.shape, (batch_size, vocab_size))
        self.assertEqual(state.shape, (batch_size, units))
