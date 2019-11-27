#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.web.forms import UserForm, QuestionForm, UpdateForm, CitationForm


class _Base(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        self.app_context = app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()


class UserFormTest(_Base):

    def test_user_form_validation(self):
        form = UserForm(data=dict(username='test',
                                  email='test@example.com'))
        self.assertFalse(form.validate())

        form.password.data = 'test-password'
        self.assertTrue(form.validate())


class QuestionFormTest(_Base):

    def test_question_form_validation(self):
        form = QuestionForm()
        self.assertFalse(form.validate())

        form.question.data = 'test sentence'
        self.assertTrue(form.validate())


class UpdateFormTest(_Base):

    def test_update_form_validation(self):
        form = UpdateForm()
        self.assertFalse(form.validate())

        form.content.data = 'test content'
        self.assertTrue(form.validate())



class CitationFormTest(_Base):

    def test_citation_form_validation(self):
        form = CitationForm()
        self.assertFalse(form.validate())

        form.author.data = 'tester'
        form.year.data = 1990
        self.assertFalse(form.validate())

        form.title.data = 'test title'
        self.assertTrue(form.validate())


if __name__ == '__main__':
    unittest.main()
