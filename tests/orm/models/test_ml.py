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

#logging.disable(logging.CRITICAL)

SAMPLE_TEXT = 'sample text'


class MLModelTest(_Base):

    def test_model_type_save_and_query(self):
        model_name = 'test_model'
        with self.assertRaises(ModuleNotFoundError):
            model = MLModel(name=model_name,
                            type='cls',
                            category='test',
                            module='invalid_module',
                            object='Class')
            model.save()

        model = MLModel(name=model_name,
                        type='cls',
                        category='question_type',
                        module='main.models',
                        object='Class')
        model.save()

        data = MLModel.query().first()

        self.assertEqual(data.id, model.id)
        self.assertEqual(data.name, model_name)

    def test_model_type_choice_field_work(self):
        model_name = 'test_model'
        model = MLModel(name=model_name,
                        type='cls',
                        category='question_type',
                        module='main.models',
                        object='Class')
        model.save()

        data = MLModel.get(model.id)
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

        with self.assertRaises(Exception):
            with self.assertLogs(level=logging.ERROR):
                model2.save()

        data = MLModel.query().filter_by(name=model_name).first()
        self.assertEqual(data.category, cat)

        with self.assertRaises(ValueError):
            MLModel(name='invalid_type',
                    type='incalid',
                    category=cat,
                    module='main.models',
                    object='Class')


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

        self.qtype = QuestionType(type='testcase')
        self.qtype.save()

        q = Question(question='is this test')
        q.save()
        self.question_id = q.id

        img = Image(filename='test.jpg')
        img.save()
        self.img_id = img.id


    def test_model_request_log_saved(self):
        log = RequestLog(
            question_type=self.qtype,
            question_id=self.question_id,
            image_id=self.img_id,
            log_type='success',
            log_text=SAMPLE_TEXT)

        log.save()

        data = RequestLog.query().first()

        self.assertEqual(data.log_text, SAMPLE_TEXT)

    def test_serialize_data_as_dict(self):
        log = RequestLog(
            question_type_id=self.qtype.id,
            question_id=self.question_id,
            image_id=self.img_id,
            log_type='success',
            log_text=SAMPLE_TEXT)

        log.save()

        data = log.to_dict()

        self.assertEqual(data['id'], log.id)
        self.assertEqual(data['question_type_id'], self.qtype.id)
        self.assertEqual(data['log_type'], 'success')
        self.assertEqual(data['log_text'], SAMPLE_TEXT)
        self.assertEqual(data['image_id'], self.img_id)

        # not stored and check if it can handle empty data
        self.assertIsNone(data['model_id'])


class PredictionScoreTest(_Base):

    def setUp(self):
        super(PredictionScoreTest, self).setUp()

        self.qtype = QuestionType(type='testcase')
        self.qtype.save()

        q = Question(question='is this test')
        q.save()
        self.question_id = q.id

        img = Image(filename='test.jpg')
        img.save()
        self.img_id = img.id

        self.test_predict = 'some result'
        log = RequestLog(
            question_type=self.qtype,
            question_id=self.question_id,
            image_id=self.img_id,
            log_type='success',
            log_text=SAMPLE_TEXT)
        log.save()

        pred = PredictionScore(prediction=self.test_predict,
                               log_id=log.id,
                               rate=1)
        pred.save()
        self.id = pred.id

    def test_model_saved_properly(self):
        data = PredictionScore.get(self.id)

        # check saved properly
        self.assertEqual(data.prediction, self.test_predict)

        # check make relationship with log
        self.assertEqual(data.log.log_text, SAMPLE_TEXT)

    def test_update_score_information(self):
        data = PredictionScore.get(self.id)

        with self.assertRaises(ValueError):
            data.update()

        target = 'validated'
        data.update(question_type='updated', answer=target)

        new_data = PredictionScore.get(self.id)
        self.assertEqual(new_data.answer, target)

    def test_handle_out_of_range_rate(self):
        with self.assertRaises(ValueError):
            PredictionScore(prediction='invalid',
                            log_id=1,
                            rate=10)

        with self.assertRaises(ValueError):
            PredictionScore(prediction='invalid',
                            log_id=1,
                            rate=0)


class QuestionTypeModelTest(_Base):

    def test_question_model_save_and_query(self):
        test = 'test type'
        qt = QuestionType(type=test)
        qt.save()

        data = QuestionType.query().filter_by(type=test).first()
        self.assertEqual(data.type, test)
        self.assertEqual(data.type, qt.type)
        init_size = QuestionType.query().count()

        # not saved model which violated unique constraints
        error_model = QuestionType(type=test)
        with self.assertRaises(Exception):
            with self.assertLogs(level=logging.ERROR):
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
