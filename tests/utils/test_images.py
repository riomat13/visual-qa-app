#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock
import logging

import numpy as np

from main.orm.models.data import Image
from main.utils.images import update_row_image

logging.disable(logging.CRITICAL)


class ImageDataTest(unittest.TestCase):

    @patch('main.utils.images.load_image_simple')
    @patch('main.utils.images.delete_image')
    @patch('main.utils.images.os.path.join')
    def test_save_image(self, mock_path, mock_delete, mock_load):
        mock_load.return_value = np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)
        path = '/tmp/test1.jpg'
        mock_path.return_value = path

        img_obj = Mock()
        img_obj.filename = 'test1.jpg'

        update_row_image(img_obj, True)

        import os
        self.assertTrue(os.path.isfile(path))
        mock_delete.assert_called_once_with(path)
        try:
            os.remove(path)
        except FileNotFoundError:
            self.fail('should not be raised')

    @patch('main.utils.images.load_image_simple')
    @patch('main.utils.images.os.path.join')
    def test_save_and_delete_image(self, mock_path, mock_load):
        mock_load.return_value = np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)

        path = '/tmp/test1.jpg'
        mock_path.return_value = path

        img_obj = Mock()
        img_obj.filename = 'test1.jpg'

        update_row_image(img_obj, True)

        # check if generated file removed successfully
        # this should work since both new and old path are the same
        import os.path
        self.assertFalse(os.path.isfile(path))
