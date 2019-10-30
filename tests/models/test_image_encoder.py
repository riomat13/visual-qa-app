#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy as np

from main.models.images import (
    load_image,
    create_mobilenet_encoder,
    ImageEncoder
)

INPUT_SHAPE = (224, 224, 3)  # batch size = 1
TEST_IMG = np.random.randint(0, 255, INPUT_SHAPE) \
        .astype(np.float32)


class ImageLoaderTest(unittest.TestCase):

    def test_image_loader(self):
        path = 'data/train2014/COCO_train2014_000000000009.jpg'
        img = load_image(path)
        self.assertEqual(img.shape, (224, 224, 3))


class MobileNetEncoderTest(unittest.TestCase):

    def test_process_and_output_with_correct_shape(self):
        mobilenet = create_mobilenet_encoder()
        out = mobilenet(np.expand_dims(TEST_IMG, axis=0))
        self.assertEqual(out.shape, (1, 128))


class ImageEncoderTest(unittest.TestCase):

    @patch('main.models.images.load_image')
    def test_encoding_image_and_check_output_shape(self, mock_load_image):
        mock_load_image.return_value = TEST_IMG
        encoder = ImageEncoder()
        feature = encoder.get_image_features('path/to/img')
        self.assertEqual(feature.shape, (128,))

    @patch('main.models.images.load_image')
    def test_encoding_multiple_images(self, mock_load_image):
        test_data_size = 10
        mock_load_image.return_value = TEST_IMG
        encoder = ImageEncoder()
        data_paths = ['path/to/img'] * test_data_size
        feature = encoder.get_imgs_features(data_paths)
        self.assertEqual(feature.shape, (test_data_size, 128,))


if __name__ == '__main__':
    unittest.main()
