#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import logging

import sqlalchemy

from main.settings import set_config
set_config('test')

from .base import _Base
from main.orm.models.ml import (
    MLModel, ModelLog, RequestLog, PredictionScore,
    QuestionType
)
from main.orm.models.data import Image, Question

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

    def setUp(self):
        super(RequestLogTest, self).setUp()

        QuestionType(type='testcase').save()
        Question(question='is this test').save()

        Image(filename='test.jpg').save()

        # take actual stored data from DB
        self.qtype = QuestionType.query().filter_by(type='testcase').first()
        self.question = Question.query().filter_by(question='is this test').first()
        self.img = Image.query().filter_by(filename='test.jpg').first()

    def test_model_request_log_saved(self):
        log = RequestLog(
            question_type=self.qtype,
            question=self.question,
            image=self.img,
            log_type='success',
            log_text=SAMPLE_TEXT)

        log.save()

        data = RequestLog.query().first()

        self.assertEqual(data.log_text, SAMPLE_TEXT)


class PredictionScoreTest(_Base):

    def setUp(self):
        super(PredictionScoreTest, self).setUp()

        QuestionType(type='testcase').save()
        Question(question='is this test').save()

        Image(filename='test.jpg').save()

        # take actual stored data from DB
        self.qtype = QuestionType.query().filter_by(type='testcase').first()
        self.question = Question.query().filter_by(question='is this test').first()
        self.img = Image.query().filter_by(filename='test.jpg').first()
    def test_model_saved_properly(self):
        predict = 'some result'
        log = RequestLog(
            question_type=self.qtype,
            question=self.question,
            image=self.img,
            log_type='success',
            log_text=SAMPLE_TEXT)
        log.save()

        PredictionScore(prediction=predict, log=log, rate=1).save()

        data = PredictionScore.query().first()

        # check saved properly
        self.assertEqual(data.prediction, predict)
        # check make relationship with log
        self.assertEqual(data.log.log_text, SAMPLE_TEXT)


class QuestionTypeModelTest(_Base):

    def test_question_model_save_and_query(self):
        test = 'test type'
        QuestionType(type=test).save()

        data = QuestionType.query().filter_by(type=test).first()
        self.assertEqual(data.type, test)
        init_size = QuestionType.query().count()

        # not saved model which violated unique constraints
        error_model = QuestionType(type=test)
        error_model.save()
        self.assertEqual(
            QuestionType.query().count(),
            init_size
        )

    def test_register_question_type(self):
        test = 'test type'
        QuestionType.register(test)

        data = QuestionType.query().first()
        self.assertEqual(data.type, test)


if __name__ == '__main__':
    unittest.main()
