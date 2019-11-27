#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.web.forms import QuestionForm, CitationForm


class QuestionFormTest(unittest.TestCase):

    def setUp(self):
        self.app = create_app('test')

    def tearDown(self):
        pass

    def test_question_form_is_valid(self):
        sample = 'test sentence'
        with self.app.test_request_context('/'):
            form = QuestionForm(data=dict(question=sample))

        self.assertEqual(form.data.get('question'), sample)
