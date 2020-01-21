#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import warnings

from main.models.ml import RequestLog
from main.mixins.models import BaseMixin, ModelLogMixin
from main.utils.logger import save_log


class AddLogToModelTest(unittest.TestCase):

    def test_fail_to_decorate_when_no_adapter(self):
        class Log(object):
            pass

        # must inherit mixins to apply save_log
        with self.assertRaises(TypeError):
            @save_log(Log)
            def test_func():
                pass

    def test_on_request_log(self):

        error_text = 'should be raised'
        test_outputs = []

        # mock log
        class Log(ModelLogMixin, BaseMixin):
            def __init__(self, log_type, log_text, log_class=None):
                self.log_type = log_type
                self.log_text = log_text
                self.log_class = log_class

            def save(self):
                # store log data to temp list
                test_outputs.append([self.log_type, self.log_text, self.log_class])

        # handle error
        @save_log(Log)
        def test_func_error():
            raise ValueError(error_text)

        # should be saved the error text by ValueError
        with self.assertRaises(ValueError):
            test_func_error()

        ltype, ltext, lcls = test_outputs.pop()
        self.assertEqual(ltype, 'error')
        self.assertEqual(ltext, error_text)
        self.assertEqual(lcls, 'ValueError')

        warning_text = 'should be warned'

        # handle warning
        @save_log(Log)
        def test_func_warning():
            warnings.warn(warning_text, RuntimeWarning)

        test_func_warning()
        ltype, ltext, lcls = test_outputs.pop()
        self.assertEqual(ltype, 'warning')
        self.assertIn(warning_text, ltext)
        self.assertEqual(lcls, 'RuntimeWarning')


        # without error nor warning
        @save_log(Log)
        def test_func_success():
            return 1

        test_func_success()
        ltype, ltext, lcls = test_outputs.pop()
        self.assertEqual(ltype, 'success')
        self.assertEqual(ltext, 'success')
        self.assertIsNone(lcls)
