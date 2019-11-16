#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from .base import _Base
from main.orm.models.ml import (
    MLModel, ModelLog, RequestLog, PredictionScore
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


class RequestLogTest(_Base):

    def test_model_request_log_saved(self):
        log = RequestLog(
            filename='test.txt',
            question_type='test',
            question='is this test',
            log_type='success',
            log_text=SAMPLE_TEXT)

        log.save(session=self.session)

        data = RequestLog.query(session=self.session).first()

        self.assertEqual(log.id, data.id)
        self.assertEqual(log.log_text, data.log_text)


class PredictionScoreTest(_Base):

    def test_model_saved_properly(self):
        log = RequestLog(
            filename='test',
            question_type='test',
            question='test',
            log_type='success',
            log_text='none')
        log.save()

        pred = PredictionScore('some result',
                               log=log,
                               rate=1)
        pred.save(session=self.session)

        data = PredictionScore.query(session=self.session).first()
        # check saved properly
        self.assertEqual(pred.id, data.id)
        # check make relationship with log
        self.assertEqual(pred.log_id, log.id)


if __name__ == '__main__':
    unittest.main()
