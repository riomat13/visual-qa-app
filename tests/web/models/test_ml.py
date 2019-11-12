#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from .base import _Base
from main.web.models.ml import (
    MLModel, ModelLog, ModelRequestLog, PredictionScore
)

SAMPLE_TEXT = 'sample text'


class MLModelTest(_Base):

    def test_model_type_save_and_query(self):
        model = MLModel(name='test_model',
                        type='question_type',
                        category='classification',
                        module='main.models.questions.types')

        model.save(session=self.session)

        data = MLModel.query(session=self.session).first()

        self.assertEqual(model.id, data.id)


class ModelLogTest(_Base):

    def test_model_log_saved(self):
        log = ModelLog(log_type='success', log_text=SAMPLE_TEXT)

        log.save(session=self.session)

        data = ModelLog.query(session=self.session).first()

        self.assertEqual(log.id, data.id)
        self.assertEqual(log.log_text, data.log_text)


class ModelRequestLogTest(_Base):

    def test_model_request_log_saved(self):
        log = ModelRequestLog(log_type='success', log_text=SAMPLE_TEXT)

        log.save(session=self.session)

        data = ModelRequestLog.query(session=self.session).first()

        self.assertEqual(log.id, data.id)
        self.assertEqual(log.log_text, data.log_text)


class PredictionScoreTest(_Base):

    def test_model_saved_properly(self):
        pred = PredictionScore('sample.jpg', 'some question', 'some result', rate=1)
        pred.save(session=self.session)

        data = PredictionScore.query(session=self.session).first()
        self.assertEqual(pred.id, data.id)
        self.assertEqual(pred.filename, data.filename)


if __name__ == '__main__':
    unittest.main()
