#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import random

import numpy as np

from main.utils.preprocess import (
    text_processor,
    one_hot_converter,
    data_generator
)


class TextProcessorTest(unittest.TestCase):
    def test_text_processor(self):
        sample = ['test sentence', 'new sentence']

        vocab = set(
            [word for sent in sample
                for word in sent.split()] + ['<pad>', '<unk>']
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
        # and '<unk>' token should be stored in the index
        self.assertEqual(len(processed[0]), len(new_sample[0].split()))

        for sent in new_sample:
            for word in sent.split():
                vocab.add(word)
        self.assertEqual(processor.vocab_size, init_size)

        processor.update(new_sample)
        self.assertEqual(processor.vocab_size, len(vocab))

        # updated vocab so that it has to match the length
        processed = processor(new_sample)
        self.assertEqual(len(processed[0]), len(new_sample[0].split()))

    def test_text_processor_passing_str(self):
        sample = 'test sentence'

        vocab = set(sample.split() + ['<pad>', '<unk>'])
        init_size = len(vocab)

        processor = text_processor(sample)
        processed = processor(sample)
        self.assertEqual(processor.vocab_size, init_size)

    def test_processed_sentence_lengths_are_same(self):
        sample_text = [
            'sample text sentence',
            'This is another sentence',
            'This is not processed yet'
        ]
        processor = text_processor(sample_text)
        examples = [
            'this should be processed by tokenizer',
            'this is also be processed'
        ]
        processed = processor(examples)
        self.assertEqual(len(processed[0]), len(processed[1]))

    def test_text_processor_load_data_by_config(self):
        processor = text_processor(from_config=True)
        self.assertGreater(processor.vocab_size, 0)

    def test_reuse_text_processor_by_json(self):
        sample_text = ['sample text sentence']
        processor = text_processor(sample_text)
        target = processor(sample_text)
        json_cfg = processor.to_json()

        # rebuild with json config
        processor_from_json = text_processor(json_cfg, from_json=True)
        processed = processor_from_json(sample_text)

        self.assertTrue(np.all(processed == target))

    def test_one_hot_encoding(self):
        n = 5
        test_data = list(range(n))
        random.shuffle(test_data)

        matrix = one_hot_converter(test_data, C=n)

        self.assertEqual(matrix.shape, (n, n))
        self.assertEqual(np.sum(matrix), n)

        for vec, val in zip(matrix, test_data):
            self.assertEqual(vec[val], 1)


class DataGeneratorTest(unittest.TestCase):

    def test_data_generator_for_single_data_without_process(self):
        # test with sets with single data type
        steps = 5
        batch_size = 4
        data_size = batch_size * steps
        dataset = list(range(data_size))

        for i, d in enumerate(data_generator(dataset, batch_size), 1):
            self.assertEqual(len(d), batch_size)

        self.assertEqual(i, steps)

    def test_data_generator_for_dataset_without_process(self):
        # test with dataset with sets of two types of data
        steps = 5
        batch_size = 4
        data_size = batch_size * steps
        dataset = [(i, j) for i, j in enumerate(['test'] * data_size)]

        for i, batch in enumerate(data_generator(dataset, batch_size, expand=True), 1):
            self.assertEqual(len(batch), 2)
            self.assertEqual(len(batch[0]), batch_size)

        self.assertEqual(i, steps)

    def sample_processor1(self, dataset):
        # fake process function
        size = len(dataset)
        return list(range(size)), list(reversed(range(size)))

    def test_data_generator_from_process_func(self):
        # test with sets with single data type
        steps = 5
        batch_size = 4
        data_size = batch_size * steps
        dataset = list(range(data_size))

        for i, batch in enumerate(
                data_generator(dataset, batch_size,
                               process_func=self.sample_processor1), 1):
            self.assertEqual(len(batch), 2)
            self.assertEqual(len(batch[0]), batch_size)

        self.assertEqual(i, steps)

    def sample_processor2(self, dataset):
        # fake process function
        size = len(dataset)
        return [(u, v) for u, v in zip(range(size), reversed(range(size)))]

    def test_data_generator_with_expanding_data_from_process_func(self):
        # test with sets with single data type
        steps = 5
        batch_size = 4
        data_size = batch_size * steps
        dataset = list(range(data_size))

        for i, batch in enumerate(
                data_generator(dataset, batch_size,
                               expand=True,
                               process_func=self.sample_processor2), 1):
            self.assertEqual(len(batch), 2)
            self.assertEqual(len(batch[0]), batch_size)

        self.assertEqual(i, steps)


if __name__ == '__main__':
    unittest.main()
