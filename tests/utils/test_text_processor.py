#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import random

import numpy as np

from main.utils.preprocess import text_processor, one_hot_converter


class TextProcessorTest(unittest.TestCase):
    def test_text_processor(self):
        sample = ['test sentence', 'new sentence']

        vocab = set(
            [word for sent in sample
                for word in sent.split()]
        )
        init_size = len(vocab)

        processor = text_processor(sample)
        processed = processor(sample)
        self.assertEqual(processor.vocab_size, init_size)

        # check if each word in sentences is processed
        for s, p in zip(sample, processed):
            self.assertEqual(len(s.split()), len(p))

        # new one sentence with unseen words
        new_sample = ['test with new words']
        processed = processor(new_sample)

        # new sample has words wihich are not in original one
        # so that the lengths after processed should be smaller
        self.assertLess(len(processed[0]), len(new_sample[0].split()))

        for sent in new_sample:
            for word in sent.split():
                vocab.add(word)
        self.assertEqual(processor.vocab_size, init_size)

        processor.update(new_sample)
        self.assertEqual(processor.vocab_size, len(vocab))

        # updated vocab so that it has to match the length
        processed = processor(new_sample)
        self.assertEqual(len(processed[0]), len(new_sample[0].split()))

    def test_one_hot_encoding(self):
        n = 5
        test_data = list(range(n))
        random.shuffle(test_data)

        matrix = one_hot_converter(test_data, C=n)

        self.assertEqual(matrix.shape, (n, n))
        self.assertEqual(np.sum(matrix), n)

        for vec, val in zip(matrix, test_data):
            self.assertEqual(vec[val], 1)


if __name__ == '__main__':
    unittest.main()
