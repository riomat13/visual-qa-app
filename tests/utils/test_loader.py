#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# ignore tensorflow debug info and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

import unittest
from unittest.mock import patch, mock_open

import numpy as np

from main.utils.loader import (
    load_image, load_image_simple,
    VQA,
    fetch_question_types
)

logging.disable(logging.CRITICAL)


JSON_DATA = {
    'questions': [
        {'question_id': i, 'question': f'test sentence{i}', 'image_id': i}
        for i in range(1, 11)
    ],
    'annotations': [
        {
            'question_id': i,
            'question_type': f'what this is {i}',
            'answers': [
                {'answer': word + f'{i}'}
                for word in 'test sample example'.split()
            ]
        }
        for i in range(1, 11)
    ],
}


class ImageLoaderTest(unittest.TestCase):

    def test_image_loader(self):
        path = 'tests/data/test_img1.jpg'
        img = load_image(path)
        self.assertEqual(img.shape, (224, 224, 3))

    def test_image_loader_simple(self):
        path = 'tests/data/test_img1.jpg'
        img = load_image_simple(path)
        self.assertEqual(img.shape, (224, 224, 3))
        
        img = load_image_simple(path, normalize=True)
        self.assertLessEqual(np.max(img), 1.0)
        self.assertGreaterEqual(np.min(img), 0.0)


class LoadDatasetTest(unittest.TestCase):
    def setUp(self):
        self.vqa = VQA()

    def create_mock_dataset(self, data_type, num_data):
        # make sure no error
        if data_type not in ('questions', 'annotations'):
            raise ValueError()

        dataset = []
        if data_type == 'questions':
            for i, data in enumerate(JSON_DATA[data_type], 1):
                dataset.append((i, (data['question'], 'img.jpg')))
        else:
            for i, data in enumerate(JSON_DATA[data_type], 1):
                answers = [d['answer'] for d in data['answers']]
                dataset.append((i, (data['question_type'], answers)))

        return dataset

    @patch('main.utils.loader.json.load')
    @patch('builtins.open', new_callabel=mock_open)
    def test_load_dataset_works(self, mock_open, mock_json_load):
        self.vqa._create_dataset = self.create_mock_dataset

        mock_json_load.return_value = JSON_DATA

        # load all data
        target_data_size = len(JSON_DATA['questions'])
        self.vqa.load_data(num_data=target_data_size)

        data_size = len(self.vqa)
        self.assertEqual(data_size, target_data_size)

        data = self.vqa.dataset[0]

        self.assertEqual(len(data), 4)
        # check question
        self.assertEqual(data[0], JSON_DATA['questions'][0]['question'])

        # check question type
        self.assertEqual(data[1], JSON_DATA['annotations'][0]['question_type'])

        # chekc answers
        target_answers = [
            d['answer'] for d in JSON_DATA['annotations'][0]['answers']
        ]
        self.assertEqual(data[2], target_answers)

        self.assertEqual(data[3], 'img.jpg')

    def test_generator_raise_error(self):
        with self.assertRaises(RuntimeError):
            next(self.vqa.data_generator())

    @patch('main.utils.loader.json.load')
    @patch('builtins.open', new_callabel=mock_open)
    def test_generating_batch_dataset(self, mock_open, mock_json_load):
        mock_json_load.return_value = JSON_DATA
        batch_size = 4

        target_data_size = len(JSON_DATA['questions'])
        self.vqa.load_data(num_data=target_data_size)

        target_steps = (target_data_size - 1) // batch_size + 1

        count = 0
        total_data = 0

        for batch in self.vqa.data_generator(batch_size=batch_size):
            count += 1
            self.assertGreater(len(batch[0]), 0)
            total_data += len(batch[0])
            # batch contains (quesions, question_types, answers, images)
            self.assertEqual(len(batch), 4)

        self.assertEqual(count, target_steps)
        self.assertEqual(total_data, target_data_size)

    @patch('main.utils.loader.json.load')
    @patch('builtins.open', new_callabel=mock_open)
    def test_generating_dataset_once(self, mock_open, mock_json_load):
        mock_json_load.return_value = JSON_DATA
        batch_size = 4

        target_data_size = len(JSON_DATA['questions'])
        self.vqa.load_data(num_data=target_data_size)

        dataset = next(self.vqa.data_generator())

        # batch contains (quesions, question_types, answers, images)
        self.assertEqual(len(dataset), 4)

        self.assertEqual(len(dataset[0]), target_data_size)

    def test_load_dataset_and_check_datasize(self):
        self.vqa._create_dataset = self.create_mock_dataset

        target_data_size = 10
        self.vqa.load_data(num_data=10)

        data_size = len(self.vqa)
        self.assertEqual(data_size, target_data_size)
        self.assertEqual(len(self.vqa._dataset[0]), 4)

    def test_fetch_class_list(self):
        # compare to abstract question types which has 81 types
        target_class_num = 81
        classes = fetch_question_types()
        self.assertEqual(len(classes), target_class_num)


if __name__ == '__main__':
    unittest.main()
