#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import logging

import sqlalchemy

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
        model_name = 'test_model'
        model = MLModel(name=model_name,
                        type='cls',
                        category='question_type',
                        module='main.models',
                        object='Class')

        model.save()

        data = MLModel.query().first()

        self.assertEqual(data.name, model_name)

    def test_model_type_choice_field_work(self):
        model_name = 'test_model'
        model = MLModel(name=model_name,
                        type='cls',
                        category='question_type',
                        module='main.models',
                        object='Class')
        model.save()

        data = MLModel.query().filter_by(name=model_name).first()
        self.assertEqual(data.type, 'classification')

    def test_check_unique_name_with_handling_error(self):
        model_name = 'test_model'
        cat = 'question_type'
        model = MLModel(name=model_name,
                        type='cls',
                        category=cat,
                        module='main.models',
                        object='Class')
        model.save()

        model2 = MLModel(name=model_name,
                         type='seq',
                         category='what',
                         module='main.models',
                         object='Class')

        model2.save()

        data = MLModel.query().filter_by(name=model_name).first()
        self.assertEqual(data.category, cat)

    def test_update_score_data(self):
        new_score = 0.68
        model = MLModel(name='test_model',
                        type='cls',
                        category='question_type',
                        module='main.models',
                        object='Class',
                        metrics='validation_accuracy',
                        score=0.65)

        model.update_score(score=new_score)

        data = MLModel.query().first()
        self.assertEqual(data.score, new_score)

    def test_type_must_be_chosen_from_registered_items(self):
        model = MLModel(name='test_model',
                        type='cls',
                        category='question_type',
                        module='main.models',
                        object='Class')


class ModelLogTest(_Base):

    def test_model_log_saved(self):
        log = ModelLog(log_type='success', log_text=SAMPLE_TEXT)

        log.save()

        data = ModelLog.query() \
            .filter_by(log_text=SAMPLE_TEXT) \
            .first()

        self.assertEqual(data.log_text, SAMPLE_TEXT)


class RequestLogTest(_Base):

    def test_model_request_log_saved(self):
        log = RequestLog(
            filename='test.txt',
            question_type='test',
            question='is this test',
            log_type='success',
            log_text=SAMPLE_TEXT)

        log.save()

        data = RequestLog.query().first()

        self.assertEqual(data.log_text, SAMPLE_TEXT)


class PredictionScoreTest(_Base):

    def test_model_saved_properly(self):
        prediction='some result'
        log = RequestLog(
            filename='test',
            question_type='test',
            question='test',
            log_type='success',
            log_text=SAMPLE_TEXT)
        log.save()

        pred = PredictionScore(prediction=prediction,
                               log=log,
                               rate=1)
        pred.save()

        data = PredictionScore.query().first()

        # check saved properly
        self.assertEqual(data.prediction, prediction)
        # check make relationship with log
        self.assertEqual(data.log.log_text, SAMPLE_TEXT)


if __name__ == '__main__':
    unittest.main()
