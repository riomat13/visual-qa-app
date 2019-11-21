#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import logging

from main.settings import set_config
set_config('test')

from .base import _Base
from main.orm.models.ml import (
    MLModel, ModelLog, RequestLog, PredictionScore
)

logging.disable(logging.CRITICAL)

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

    def test_check_unique_name_with_handling_error(self):
        model_name = 'test_model'
        type_ = 'classification'
        model = MLModel(name=model_name,
                        type=type_,
                        category='question_type',
                        module='main.models.questions.types')
        model.save(session=self.session)
        self.session.commit()

        model2 = MLModel(name=model_name,
                         type='seq2seq',
                         category='what',
                         module='main.models.seq')

        # should be handled by mixin
        model2.save(session=self.session)

        data = MLModel.query(session=self.session) \
            .filter_by(name='test_model') \
            .first()

        self.assertEqual(data.type, type_)

    def test_update_score_data(self):
        new_score = 0.68
        model = MLModel(name='test_model',
                        type='classification',
                        category='question_type',
                        module='main.models.questions.types',
                        metrics='validation_accuracy',
                        score=0.65)

        model.update_score(score=new_score)

        data = MLModel.query(session=self.session).first()
        self.assertEqual(data.score, new_score)

    def test_type_must_be_chosen_from_registered_items(self):
        model = MLModel(name='test_model',
                        type='cls',
                        category='question_type',
                        module='main.models.questions.types')


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
