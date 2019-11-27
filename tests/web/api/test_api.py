#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock

import logging
from collections import namedtuple
from functools import partial

from requests.auth import _basic_auth_str

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.orm.db import Base, engine
from main.orm.models.base import User
from main.orm.models.ml import RequestLog, MLModel, PredictionScore, QuestionType
from main.orm.models.data import Image, Question

logging.disable(logging.CRITICAL)


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app = create_app('test')
        cls.app_context = app.app_context()
        cls.app_context.push()
        client = app.test_client()

        # add authentication
        headers = {
            'Authorization': _basic_auth_str('test', 'pwd')
        }
        client.get = partial(client.get, headers=headers)
        client.post = partial(client.post, headers=headers)
        cls.client = client

        from main.orm.db import engine
        cls.engine = engine
        Base.metadata.create_all(engine)

        user = User(username='test',
                    email='test@example.com',
                    password='pwd')
        user.save()

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(cls.engine)
        cls.app_context.pop()


class ModelListTest(_Base):

    @patch('main.web.api._api.MLModel')
    def test_extracting_model_list(self, mock_model):
        data_size = 4
        target = [
            {k: v for k, v in zip('test', range(4))}
        ]
        mock_model.to_dict.return_value = target
        
        mock_query = Mock()
        mock_query.all.return_value = [mock_model for _ in range(data_size)]
        mock_model.query.return_value = mock_query

        response = self.client.get('/api/models/all')

        self.assertEqual(response.status_code, 200)

        json_data = response.json

        self.assertEqual(len(json_data), 4)
        
        for data in json_data:
            self.assertEqual(data, target)

    def test_register_model(self):
        model_name = 'test_model'
        res = self.client.post(
            '/api/register/model',
            data=dict(name=model_name,
                      type='cls',
                      category='test',
                      module='tests.web.api',
                      object='TestCase')
        )
        data = MLModel.query().filter_by(name=model_name).first()
        self.assertTrue(data)

    def test_register_model_handle_invalid_input(self):
        res = self.client.post(
            '/api/register/model',
            data=dict(name='test',
                      type='invalid_type',
                      category='test',
                      module='tests.web.api',
                      object='TestCase')
        )

        self.assertEqual(res.status_code, 400)
        self.assertTrue('error' in res.json)

        res = self.client.post(
            '/api/register/model',
            data=dict(name='test',
                      type='cls',
                      category='test',
                      module='tests.web.invalid.module',
                      object='TestCase')
        )

        self.assertEqual(res.status_code, 400)
        self.assertTrue('error' in res.json)

    def test_extract_model_info_by_id(self):
        model_name = 'test'
        model = MLModel(name=model_name,
                        type='cls',
                        category='test',
                        module='tests.web.api',
                        object='TestCase')
        model.save()

        data = MLModel.query().filter_by(name=model_name).first()
        res = self.client.get(f'/api/model/{data.id}')
        self.assertEqual(res.json.get('name'), model_name)

        # test with non-exist id
        res = self.client.get(f'/api/model/1000')
        # this must be empty
        self.assertEqual(len(res.json), 0)


class QuestionTypeTest(_Base):

    @patch('main.web.api._api.run_model')
    @patch('main.web.api._api.asyncio.run')
    def test_predict_question_type_request(self, mock_async_run, mock_run_model):
        test_return = 'model'
        mock_run_model.return_value = test_return
        mock_async_run.return_value = 'test'

        q = 'is this test?'
        response = self.client.post('/api/predict/question_type',
                                    data=dict(question=q))

        # check data is properly passed
        data = response.json
        self.assertEqual(data.get('question'), q)

        # model is called with empty string and the given string
        mock_run_model.assert_called_once_with('', q)

        # execute server by run_model function
        mock_async_run.assert_called_once_with(test_return)

    @patch('main.web.api._api.run_model')
    @patch('main.web.api._api.asyncio.run')
    def test_error_handling_question_type_prediction(self,
                                                     mock_async_run,
                                                     mock_run_model):
        # error code
        mock_async_run.return_value = '<e>'

        response = self.client.post('/api/predict/question_type',
                                    data=dict(question='invalid'))

        self.assertEqual(response.status_code, 200)

        data = response.json
        self.assertTrue('error' in data)


class ExtractRequestLogsTest(_Base):

    @patch('main.web.api._api.RequestLog.query')
    def test_extract_all_logs(self, mock_query):
        model = Mock(RequestLog())
        model.to_dict.return_value = {'key': 'test'}
        size = 4

        mock_all = Mock()
        mock_all.return_value = [model] * size

        mock_query.return_value.all = mock_all

        # TODO: filter by question type
        response = self.client.get('/api/logs/requests')
        self.assertEqual(response.status_code, 200)
        data = response.json

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'test')

    @patch('main.web.api._api.RequestLog.query')
    def test_extract_question_type_logs(self, mock_query):
        model = Mock(RequestLog)
        model.to_dict.return_value = {'key': 'value'}
        size = 4

        # mocking query
        mock_all = Mock()
        mock_all.return_value = [model] * size
        mock_filter = Mock()
        mock_filter.return_value.all = mock_all
        mock_query.return_value.filter = mock_filter

        # TODO: filter by question type
        response = self.client.post('/api/logs/requests',
                                    data=dict(question_type='test'))
        self.assertEqual(response.status_code, 200)
        data = response.json

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'value')

    def test_extract_non_registered_logs(self):
        res = self.client.post('/api/logs/requests',
                               data=dict(question_type='test'))
        self.assertEqual(res.status_code, 200)
        # should return empty
        self.assertEqual(len(res.json), 0)

    def test_extract_actual_logs(self):
        size = 5
        target_size = 3

        target_type = 'question_type'
        base_filename = 'img{}.jpg'
        QuestionType(type=target_type).save()
        QuestionType(type='dummy').save()

        target_qtype = QuestionType.query() \
            .filter_by(type=target_type) \
            .first()
        dummy_qtype = QuestionType.query() \
            .filter_by(type='dummy') \
            .first()

        for i in range(size):
            Image(filename=base_filename.format(i)).save()
            Question(question='test question').save()

        img = Image.query().first()
        question = Question.query().first()

        for i in range(target_size):
            RequestLog(question_type=target_qtype,
                       question=question,
                       image=img,
                       log_type='test',
                       log_text=f'this is test {i}').save()

        # append type which should be filtered out
        for i in range(size - target_size):
            RequestLog(question_type=dummy_qtype,
                       question=question,
                       image=img,
                       log_type='test',
                       log_text=f'this is dummy {i}').save()

        response = self.client.post(
            '/api/logs/requests',
            data=dict(question_type=target_type)
        )
        data = response.json
        self.assertEqual(len(data), target_size)


class ExtractPredictionScoreLogsTest(_Base):

    def setUp(self):
        super(ExtractPredictionScoreLogsTest, self).setUp()


    @patch('main.web.api._api.PredictionScore.query')
    def test_extract_all_scores(self, mock_query):
        size = 4

        Log = namedtuple(
            'TestLog',
            'rate, prediction, probability, answer, predicted_time, log_id, log'
        )
        log = Log(**{'rate': 1,
                     'prediction': 'test',
                     'probability': 0.4,
                     'answer': 'test',
                     'predicted_time': '',
                     'log_id': 1,
                     'log': None})

        mock_all = Mock(PredictionScore)
        mock_all.return_value = [log] * size
        mock_query.return_value.all = mock_all

        # send request and extract data
        response = self.client.get('/api/logs/predictions')
        self.assertEqual(response.status_code, 200)
        data = response.json

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('prediction'), 'test')

    def test_extract_scores_by_question_type(self):
        size = 5
        target_size = 3

        target_type = 'question_type'
        base_filename = 'img{}.jpg'
        QuestionType(type=target_type).save()
        QuestionType(type='dummy').save()

        target_qtype = QuestionType.query() \
            .filter_by(type=target_type) \
            .first()
        dummy_qtype = QuestionType.query() \
            .filter_by(type='dummy') \
            .first()

        for i in range(size):
            Image(filename=base_filename.format(i)).save()
            Question(question='test question').save()

        img = Image.query().first()
        question = Question.query().first()

        for i in range(target_size):
            RequestLog(question_type=target_qtype,
                       question=question,
                       image=img,
                       log_type='test',
                       log_text=f'this is test {i}').save()

        # append type which should be filtered out
        for i in range(size - target_size):
            RequestLog(question_type=dummy_qtype,
                       question=question,
                       image=img,
                       log_type='test',
                       log_text=f'this is dummy {i}').save()

        for log in RequestLog.query().all():
            PredictionScore(log=log, prediction='pred').save()

        # send request and extract data
        response = self.client.post(
            '/api/logs/predictions',
            data=dict(question_type=target_type)
        )

        self.assertEqual(response.status_code, 200)
        data = response.json

        self.assertEqual(len(data), target_size)

        for log in data:
            self.assertEqual(log.get('prediction'), 'pred')
