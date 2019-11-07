#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.orm.db import Base, session_builder
from main.web.models.ml import MLModel, ModelLog, ModelRequestLog

Session = session_builder()
session = None

SAMPLE_TEXT = 'sample text'

class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global session
        session = Session()

    @classmethod
    def tearDownClass(cls):
        session.close()
        Session.remove()

    def setUp(self):
        app = create_app('test')
        app_context = app.app_context()
        app_context.push()

        self.app_context = app_context

        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

    def tearDown(self):
        session.rollback()
        self.app_context.pop()
        Base.metadata.drop_all(self.engine)


class MLModelTest(_Base):

    def test_model_type_save_and_query(self):
        model = MLModel(type='question_type',
                           category='classification',
                           module='main.models.questions.types')
        session.add(model)
        session.flush()

        data = MLModel.query(session).first()

        self.assertEqual(model.id, data.id)


class ModelLogTest(_Base):

    def test_model_log_saved(self):
        log = ModelLog(log_type='success', log_text=SAMPLE_TEXT)

        session.add(log)
        session.flush()

        data = ModelLog.query(session).first()

        self.assertEqual(log.id, data.id)
        self.assertEqual(log.log_text, data.log_text)


class ModelRequestLogTest(_Base):

    def test_model_request_log_saved(self):
        log = ModelRequestLog(log_type='success', log_text=SAMPLE_TEXT)

        session.add(log)
        session.flush()

        data = ModelRequestLog.query(session).first()

        self.assertEqual(log.id, data.id)
        self.assertEqual(log.log_text, data.log_text)


if __name__ == '__main__':
    unittest.main()
