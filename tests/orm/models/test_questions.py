#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import logging

import sqlalchemy

from main.settings import set_config
set_config('test')

from .base import _Base
from main.web.app import create_app
from main.orm.models.questions import QuestionType

logging.disable(logging.CRITICAL)


class QuestionModelTest(_Base):

    def test_question_model_save_and_query(self):
        test = 'test type'
        model = QuestionType(type=test)
        model.save(session=self.session)

        self.session.commit()

        data = QuestionType.query(session=self.session).first()
        self.assertEqual(data.id, model.id)
        self.assertEqual(data.type, test)

        # not saved model which violated unique constraints
        error_model = QuestionType(type=test)
        error_model.save(session=self.session)
        self.assertEqual(
            QuestionType.query(session=self.session).count(),
            1
        )

    def test_register_question_type(self):
        test = 'test type'
        QuestionType.register(test)

        data = QuestionType.query(session=self.session).first()
        self.assertEqual(data.type, test)


if __name__ == '__main__':
    unittest.main()
