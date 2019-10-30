#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, mock_open

from main.utils.loader import VQA


JSON_DATA = {
    'questions': [
        {'question_id': i, 'question': f'test sentence{i}', 'image_id': i}
        for i in range(1, 11)
    ],
    'annotations': [
        {
            'question_id': i,
            'answers': [
                {'answer': word + f'{i}'}
                for word in 'test sample example'.split()
            ]
        }
        for i in range(1, 11)
    ],
}


class LoadDatasetTest(unittest.TestCase):
    def setUp(self):
        self.vqa = VQA()

    @patch('main.utils.loader.json.load')
    @patch('builtins.open', new_callabel=mock_open)
    def test_load_dataset_works(self, mock_open, mock_json_load):
        mock_json_load.return_value = JSON_DATA

        # load all data
        target_data_size = len(JSON_DATA['questions'])
        self.vqa.load_data(num_data=target_data_size)

        data_size = len(self.vqa)
        self.assertEqual(data_size, target_data_size)

        data = self.vqa.dataset[0]

        self.assertEqual(len(data), 3)
        # check question
        self.assertEqual(data[0], JSON_DATA['questions'][0]['question'])

        # chekc answers
        target_answers = [
            d['answer'] for d in JSON_DATA['annotations'][0]['answers']
        ]
        self.assertEqual(data[1], target_answers)

        # check image path
        target_image = 'COCO_train2014_{:012d}.jpg' \
            .format(JSON_DATA["questions"][0]["image_id"])
        self.assertTrue(data[2].endswith(target_image))

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

        self.assertEqual(count, target_steps)
        self.assertEqual(total_data, target_data_size)

    def test_load_dataset_and_check_datasize(self):
        target_data_size = 10
        self.vqa.load_data(num_data=10)

        data_size = len(self.vqa)
        self.assertEqual(data_size, target_data_size)


if __name__ == '__main__':
    unittest.main()
