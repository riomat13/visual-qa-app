#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import os
from functools import wraps
import tempfile
import logging
logging.disable(logging.CRITICAL)

from main.settings import set_config
set_config('test')

from main.settings import Config


def mock_logger(*args, **kwargs):
    def logger(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            res = func(*args, **kwargs)
            return res
        return decorator
    return logger

patch('main.tasks.sender.AppLog.save').start()
mock_logger = patch('main.tasks.sender.save_log', mock_logger).start()

from main.tasks.sender import send_data, send_dataset


class SendDataTest(unittest.TestCase):

    def setUp(self):
        self.files = [
            tempfile.NamedTemporaryFile('r') for _ in range(5)
        ]

    def tearDown(self):
        for f in self.files:
            f.close()
        self.files = []

    def create_tmp_file(self, n=1):
        for _ in range(n):
            f = tempfile.NamedTemporaryFile('r')
            self.files.append(f)

    @patch('main.tasks.sender.os.path.join')
    @patch('main.tasks.sender.make_sftp_connection')
    def test_send_single_data(self, mock_sftp, mock_path):
        #self.create_tmp_file()
        mock_path.return_value = 'test_path'

        send_data('', data_type='image')
        mock_sftp.return_value.__enter__.assert_called()
        mock_sftp.return_value.__enter__.return_value.put.assert_called_once_with(
            'test_path', 'test_path'
        )

        # run with no exception
        send_data('', data_type='log')
        send_data('', data_type='question')
        send_data('', data_type='answer')

        with self.assertRaises(ValueError):
            send_data('', data_type='invalid')

    def test_send_multiple_data(self):
        size = 10
        self.create_tmp_file(n=size)

        try:
            send_dataset([f.name for f in self.files], data_type='image')
        except Exception:
            self.fail()
