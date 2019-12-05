#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Load data from VQA
#

import os.path
import warnings
import logging
import time
import random
import json

import tensorflow as tf
from tensorflow.keras.applications import mobilenet

from keras_preprocessing import image

from main.settings import ROOT_DIR

log = logging.getLogger(__name__)


def load_image(img_path):
    """Load image and store value into numpy.ndarray.
    This will resize image into required size (224, 2224, 3) for
    MobileNet.

    Args:
        img_path: str
            path to an image file
    Returns:
        Tensor object with shape=(224, 224, 3), dtype=float32
    """
    # https://www.tensorflow.org/tutorials/text/image_captioning
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # input shape of MobileNet is (224, 224, 3)
    img = tf.image.resize(img, (224, 224))
    img = mobilenet.preprocess_input(img)
    return img


def load_image_simple(img_path, normalize=True):
    """Load image and store value into numpy.ndarray.
    This is for skipping image processing steps and speeding up
    training steps, therefore all image shapes must be (224, 224, 3).

    Args:
        img_path: str
            path to an image file
        normalize: bool
            return array with range [0.0, 1.0]
    Returns:
        numpy.ndarray with shape=(224, 224, 3), dtype=float32
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    if normalize:
        img /= 255.
    return img


def fetch_question_types(data_dir=None):
    """Fetch class types from a text in VQA dataset.
    Return a list of str
    """
    data_dir = data_dir or f'{ROOT_DIR}/data/QuestionTypes'
    filename = 'abstract_v002_question_types.txt'
    file_path = os.path.join(data_dir, filename)

    classes = []

    with open(file_path, 'r') as f:
        for line in f:
            classes.append(line.strip())

    log.info('Total Number of Classes:', len(classes))
    return classes


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
        for elem in data[data_type][:num_data]:
            # data is pivotted by question_id
            question_id = elem['question_id']
            data = data_extractor(elem)

            dataset.append((question_id, data))

        return dataset

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self._dataset, f)
        log.info(f'Saved data to {filepath}')

    def _get_answers(self, data):
        question_type = data['question_type']
        answers = [d['answer'] for d in data['answers']]
        return question_type, answers

    def _get_question(self, data):
        img_path = self._get_image(data['image_id'])
        return (data['question'], img_path)

    def _get_image(self, image_id, mode='train'):
        # use only training dataset for now
        filename = 'COCO_{}2014_{:012d}.jpg'.format(mode, image_id)
        img_path = os.path.join(self._data_dir, f'{mode}2014', filename)
        return img_path

    def load_data(self, num_data=-1):
        # TODO: this has overhead since this loads entire data once
        start = time.time()
        dataset = []
        questions = self._create_dataset(
                data_type='questions', num_data=num_data)
        answers = self._create_dataset(
                data_type='annotations', num_data=num_data)

        for q, a in zip(questions, answers):
            # check if the question_id is the same
            if q[0] == a[0]:
                dataset.append((q[1][0], a[1][0], a[1][1], q[1][1]))
            else:
                warnings.warn(
                    f'question_id does not match: {q[0]} != {a[0]}',
                    RuntimeWarning
                )

        self._dataset = dataset

        if self._dataset:
            self._generatable = True

        end = time.time()
        log.info('Loaded {} dataset in {:.4f} sec.'.format(
            len(self._dataset), end - start))

    @classmethod
    def load_from_json(cls, filepath):
        # TODO
        pass

    def data_generator(self, batch_size=None, shuffle=True, repeat=False):
        """
        Genarate data.

        Args:
            batch_size: int
                data size to generate in each iteration
                if None, generate all data at once
            shuffle: boolean
                shuffle data if True
            repeat: boolean
                repeat to generate data after cosumed all data
        Return:
            generator: questions, question_types, answers, images(paths)
        """
        if not self._generatable:
            raise RuntimeError('Must run load_data() first')

        # flag to check if repeat generating dataset
        keep = True

        if batch_size is not None:
            num_steps = (len(self._dataset) - 1) // batch_size + 1

        while keep:
            if shuffle:
                random.shuffle(self._dataset)
            if batch_size is not None:
                for step in range(num_steps):
                    start_idx = step * batch_size
                    data = self._dataset[start_idx:start_idx+batch_size]
                    # return in (questions, answers, image_paths) order
                    yield list(zip(*data))
            else:
                # if not specified data size, generate all data
                yield list(zip(*self._dataset))

            keep &= repeat
