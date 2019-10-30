#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Load data from VQA
#

import os.path
import time
import random
import json

from main.settings import ROOT_DIR


class VQA(object):
    """
    Load data from VQA and COCO dataset and generate data.

    Example usage:
        >>> vqa = VQA()
        >>> vqa.load_data(num_data=1000)
        >>> len(vqa)
        1000
        >>> for data in vqa.data_generator(batch_size=16):
        ...     print(len(data), len(data[0]))
        3 16
    """

    def __init__(self, data_dir=None):
        self._data_dir = data_dir or f'{ROOT_DIR}/data'
        self._dataset = []
        self._generatable = False

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return list(self._dataset)

    def _create_dataset(self, data_type, num_data):
        if data_type not in ('questions', 'annotations'):
            raise ValueError(
                "data_type must be chosen from 'questions' or 'annotations'"
            )

        if data_type == 'questions':
            data_extractor = self._get_question
        elif data_type == 'annotations':
            data_extractor = self._get_answers

        # file name has changed due to update data in VQA ver.2
        file_name = f'v2_mscoco_train2014_{data_type}.json'
        file_path = os.path.join(self._data_dir, data_type, file_name)

        with open(file_path, 'r') as f:
            data = json.load(f)

        dataset = []

        # always output the same dataset
        for img_data in data[data_type][:num_data]:
            # data is pivotted by question_id
            question_id = img_data['question_id']
            data = data_extractor(img_data)

            dataset.append((question_id, data))

        return dataset

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self._dataset, f)
        print(f'Saved data to {filepath}')

    def _get_answers(self, data):
        answers = [d['answer'] for d in data['answers']]
        return answers

    def _get_question(self, data):
        img_path = self._get_image(data['image_id'])
        return (data['question'], img_path)

    def _get_image(self, image_id, mode='train'):
        # use only training dataset for now
        filename = 'COCO_{}2014_{:012d}.jpg'.format(mode, image_id)
        img_path = os.path.join(self._data_dir, f'{mode}2014', filename)
        return img_path

    def load_data(self, num_data=-1):
        start = time.time()
        dataset = []
        questions = self._create_dataset(
                data_type='questions', num_data=num_data)
        answers = self._create_dataset(
                data_type='annotations', num_data=num_data)

        for q, a in zip(questions, answers):
            # check if the question_id is the same
            if q[0] == a[0]:
                dataset.append((q[1][0], a[1], q[1][1]))
            else:
                raise RuntimeWarning(
                    f'question_id does not match: {q[0]} != {a[0]}'
                )

        self._dataset = dataset

        if self._dataset:
            self._generatable = True

        end = time.time()
        print('Loaded {} dataset in {:.4f} sec.'.format(
            len(self._dataset), end - start))

    @classmethod
    def load_from_json(cls, filepath):
        data = json.load(filepath)

    def data_generator(self, batch_size=32, shuffle=True, repeat=False):
        """
        Genarate data.

        Args:
            batch_size: int
                data size to generate in each iteration
            shuffle: boolean
                shuffle data if True
            repeat: boolean
                repeat to generate data after cosumed all data
        Return:
            generator: questions, answers, images(paths)
        """
        if not self._generatable:
            raise RuntimeError('Must run load_data() first')

        # flag to check if repeat generating dataset
        keep = True

        num_steps = (len(self) - 1) // batch_size + 1

        while keep:
            if shuffle:
                random.shuffle(self._dataset)
            for step in range(num_steps):
                start_idx = step * batch_size
                data = self._dataset[start_idx:start_idx+batch_size]
                # return in (questions, answers, image_paths) order
                yield list(zip(*data))

            keep &= repeat
