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
    SimpleQuestionImageEncoder,
    QuestionImageEncoder,
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

        self.assertEqual(out.shape, (batch_size, 49, 1024))


class AttentionModelTest(unittest.TestCase):

    def test_dot_attention_model(self):
        # feature shape is (128,)
        batch_size = 4
        seq_length = 10
        hidden_units = 128

        # encoded sequence shape (batch_size, time_steps, seq_len)
        features = np.random.randn(batch_size, seq_length, hidden_units)
        states = np.random.randn(batch_size, hidden_units)

        # different unit size from input
        model = Attention(32, seq_length)
        out, weights = model(features, states)
        self.assertEqual(out.shape, (batch_size, hidden_units))
        self.assertEqual(weights.shape, (batch_size, seq_length, 1))

    def test_additive_attention_model(self):
        # feature shape is (128,)
        batch_size = 4
        seq_length = 10
        hidden_units = 128

        # encoded sequence shape (batch_size, time_steps, seq_len)
        features = np.random.randn(batch_size, seq_length, hidden_units)
        states = np.random.randn(batch_size, hidden_units)

        # different unit size from input
        units = 32
        model = Attention(units, seq_length, mode='additive')
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


class QuestionImageEncoderTest(unittest.TestCase):

    def test_question_image_encoder_model(self):
        batch_size = 4
        seq_length = 10
        embedding_dim = 64
        vocab_size = 1000
        image_features = np.random.randn(batch_size, 49, 1024)
        sequence = np.random.randint(0, vocab_size, (batch_size, seq_length))

        units = 64
        model = QuestionImageEncoder(units, 1000, embedding_dim)
        q_out, img_out = model(sequence, image_features)
        self.assertEqual(q_out.shape, (batch_size, seq_length, embedding_dim))
        self.assertEqual(img_out.shape, (batch_size, 49, units))

    def test_simple_question_image_encoder_model(self):
        batch_size = 4
        seq_length = 10
        embedding_dim = 64
        vocab_size = 1000
        image_features = np.random.randn(batch_size, 49, 1024)
        sequence = np.random.randint(0, vocab_size, (batch_size, seq_length))

        units = 64
        model = SimpleQuestionImageEncoder(units, 1000, embedding_dim)
        q_out, img_out = model(sequence, image_features)
        self.assertEqual(q_out.shape, (batch_size, seq_length, embedding_dim))
        self.assertEqual(img_out.shape, (batch_size, 49, units))
