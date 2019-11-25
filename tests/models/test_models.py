#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ignore tensorflow debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from unittest.mock import patch, Mock

import numpy as np

from main.models import QuestionAnswerModel


class QuestionAnswerModelTest(unittest.TestCase):

    @patch('main.models._models.SequenceGeneratorModel.load_weights')
    @patch('main.models._models.QuestionImageEncoder.load_weights')
    def test_mobilenet_encoder(self, mock_qa_loader, mock_gen_loader):
        batch_size = 4
        seq_len = 7

        model = QuestionAnswerModel(512, seq_len, 20000, 256, 'what')

        input_words = np.array([1] * batch_size)
        qs = np.arange(15 * batch_size).reshape(batch_size, 15)
        imgs = np.random.rand(batch_size, 49, 1024)
        hidden = np.random.rand(batch_size, 256)

        out, w = model(input_words, qs, imgs, hidden)

        self.assertEqual(out.shape, (batch_size, seq_len-1))
        self.assertEqual(w.shape, (batch_size, seq_len-1, 15))
