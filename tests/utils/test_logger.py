#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from main.orm.models.ml import RequestLog
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

        # mock log
        class Log(ModelLogMixin, BaseMixin):
            def __init__(self, log_type, log_text):
                self.log_type = log_type
                self.log_text = log_text

            def save(self):
                # check if saving log_text
                raise NotImplementedError(self.log_text)

        @save_log(Log)
        def test_func_error():
            raise ValueError(error_text)

        with self.assertRaises(NotImplementedError, msg=error_text):
            test_func_error()

        @save_log(Log)
        def test_func_success():
            return 1

        try:
            test_func_success()
        except:
            self.fail('Should not fail the test')
