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
from main.orm.db import Base, reset_db
from main.orm.models.base import User
from main.orm.models.ml import RequestLog, MLModel, PredictionScore, QuestionType
from main.orm.models.data import Image, Question
from main.orm.models.web import Update, Citation

logging.disable(logging.CRITICAL)


class _Base(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        self.app_context = app.app_context()
        self.app_context.push()

        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

        user = User(username='testcase',
                    email='test@example.com',
                    password='pwd')
        user.save()

        # add authentication
        headers = {
            'Authorization': _basic_auth_str('testcase', 'pwd')
        }

        client = app.test_client()
        client.get = partial(client.get, headers=headers)
        client.post = partial(client.post, headers=headers)
        self.client = client

    def tearDown(self):
        self.app_context.pop()
        Base.metadata.drop_all(self.engine)


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

        json_data = response.json.get('data')

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
        # not successed process
        self.assertFalse(res.json.get('done'))
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
        data = res.json.get('data')
        self.assertEqual(data.get('name'), model_name)

        # test with non-exist id
        res = self.client.get(f'/api/model/1000')
        data = res.json.get('data')
        # this must be empty
        self.assertEqual(len(data), 0)


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

        self.assertEqual(response.status_code, 400)

        data = response.json
        self.assertTrue('message' in data)


class ExtractRequestLogsTest(_Base):

    @patch('main.web.api._api.RequestLog.query')
    def test_extract_all_logs(self, mock_query):
        model = Mock(RequestLog())
        model.to_dict.return_value = {'key': 'test'}
        size = 4

        mock_query.return_value.all.return_value = [model] * size

        # TODO: filter by question type
        response = self.client.get('/api/logs/requests')
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'test')

    @patch('main.web.api._api.RequestLog.query')
    def test_extract_question_type_logs(self, mock_query):
        model = Mock(RequestLog)
        model.to_dict.return_value = {'key': 'value'}
        size = 4

        # mocking query
        mock_query.return_value \
                .filter.return_value \
                .all.return_value = [model] * size

        # TODO: filter by question type
        response = self.client.post('/api/logs/requests',
                                    data=dict(question_type='test'))
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'value')

    def test_extract_non_registered_logs(self):
        res = self.client.post('/api/logs/requests',
                               data=dict(question_type='test'))
        self.assertEqual(res.status_code, 200)
        # should return empty
        self.assertEqual(len(res.json.get('data')), 0)

    def test_extract_actual_logs(self):
        size = 5
        target_size = 3

        target_type = 'question_type'
        target_qtype = QuestionType(type=target_type)
        target_qtype.save()
        dummy_qtype = QuestionType(type='dummy')
        dummy_qtype.save()

        img = Image(filename='img.jpg')
        img.save()
        question = Question(question='test question')
        question.save()

        for i in range(target_size):
            RequestLog(question_type=target_qtype,
                       question_id=question.id,
                       image_id=img.id,
                       log_type='test',
                       log_text=f'this is test {i}').save()

        # append type which should be filtered out
        for i in range(size - target_size):
            RequestLog(question_type=dummy_qtype,
                       question_id=question.id,
                       image_id=img.id,
                       log_type='test',
                       log_text=f'this is dummy {i}').save()

        response = self.client.post(
            '/api/logs/requests',
            data=dict(question_type=target_type)
        )
        data = response.json.get('data')
        self.assertEqual(len(data), target_size)


class ExtractSingleRequestLogTest(_Base):

    @patch('main.web.api._api.RequestLog')
    def test_extract_log_by_id(self, mock_log):
        mock_log.get.return_value.to_dict.return_value = 'test'
        target_id = 10
        response = self.client.get(f'/api/logs/request/{target_id}')
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        mock_log.get.assert_called_once_with(target_id)
        self.assertEqual('test', data)

    @patch('main.web.api._api.WeightFigure')
    @patch('main.web.api._api.Image')
    @patch('main.web.api._api.Question')
    @patch('main.web.api._api.RequestLog')
    def test_extract_log_data_by_id(self, mock_log, mock_q, mock_img, mock_fig):
        Log = namedtuple('RequestLog', 'question_id,image_id,fig_id,score')
        mock_score = Mock()
        mock_score.prediction = 'test_prediction'
        log = Log(1, 1, 1, mock_score)
        mock_log.get.return_value = log
        mock_q.get.return_value.question = 'test_question'
        mock_img.get.return_value.filename = 'test_image'
        mock_fig.get.return_value.filename = 'test_fig'

        target_id = 10
        response = self.client.get(f'/api/logs/qa/{target_id}')
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        mock_log.get.assert_called_once_with(target_id)
        self.assertEqual(target_id, data.get('request_id'))
        self.assertEqual('test_question', data.get('question'))
        self.assertEqual('test_prediction', data.get('prediction'))
        self.assertEqual('test_image', data.get('image'))
        self.assertEqual('test_fig', data.get('figure'))


class ExtractPredictionScoreLogsTest(_Base):

    @patch('main.web.api._api.RequestLog.query')
    def test_extract_all_scores(self, mock_query):
        size = 4

        Score = namedtuple(
            'PredictionScore',
            'rate,prediction,probability,answer,predicted_time,log'
        )
        Request = namedtuple('RequestLog', 'id,score')

        mock_all = Mock(RequestLog)
        mock_all.return_value = \
            [Request(id=i+1,
                     score=Score(**{'rate': 1,
                                    'prediction': 'test',
                                    'probability': 0.4,
                                    'answer': 'test',
                                    'predicted_time': '',
                                    'log': None}))
                for i in range(size)]
        mock_query.return_value.all = mock_all

        # send request and extract data
        response = self.client.get('/api/logs/predictions')
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        self.assertEqual(len(data), size)

        for id_, log in enumerate(data, 1):
            self.assertEqual(log.get('request_id'), id_)
            self.assertEqual(log.get('prediction'), 'test')

    def test_extract_scores_by_question_type(self):
        size = 5
        target_size = 3

        target_type = 'question_type'
        target_qtype = QuestionType(type=target_type)
        target_qtype.save()
        dummy_qtype = QuestionType(type='dummy')
        dummy_qtype.save()

        img = Image(filename='img.jpg')
        img.save()
        question = Question(question='test question')
        question.save()

        for i in range(target_size):
            score = PredictionScore(prediction='pred')
            score.save()
            RequestLog(question_type=target_qtype,
                       question_id=question.id,
                       image_id=img.id,
                       score_id=score.id,
                       log_type='test',
                       log_text=f'this is test {i}').save()

        # append type which should be filtered out
        for i in range(size - target_size):
            RequestLog(question_type=dummy_qtype,
                       question_id=question.id,
                       image_id=img.id,
                       score_id=score.id,
                       log_type='test',
                       log_text=f'this is dummy {i}').save()

        # send request and extract data
        response = self.client.post(
            '/api/logs/predictions',
            data=dict(question_type=target_type)
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json.get('done'))
        data = response.json.get('data')

        self.assertEqual(len(data), target_size)

        for log in data:
            self.assertEqual(log.get('prediction'), 'pred')


class UpdateItemTest(_Base):

    def test_extract_update_list(self):
        size = 4
        for i in range(size):
            Update(content=f'test {i}').save()

        res = self.client.get('/api/updates/all')
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json.get('done'))
        data = res.json.get('data')
        self.assertEqual(len(data), size)

    def test_register_new_update(self):
        size = 4
        init_size = Update.query().count()
        for i in range(4):
            with self.subTest(i=i):
                res = self.client.post(
                    '/api/update/register',
                    data=dict(content=f'test {i}')
                )
                self.assertEqual(res.status_code, 200)
                self.assertTrue(res.json.get('done'))

        self.assertEqual(Update.query().count(), size+init_size)

    def test_fail_to_register_update(self):
        res = self.client.post(
            '/api/update/register',
            data={},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn('error', res.json)


class CitationItemTest(_Base):

    def test_extract_references(self):
        size = 4
        init_size = Citation.query().count()
        for i in range(size):
            Citation(author='tester',
                     title=f'test {i}',
                     year=1990+i).save()

        res = self.client.get('/api/references/all')
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json.get('done'))
        data = res.json.get('data')
        self.assertEqual(len(data), size+init_size)

    def test_register_new_citation(self):
        size = 4
        init_size = Citation.query().count()
        for i in range(4):
            with self.subTest(i=i):
                res = self.client.post(
                    '/api/reference/register',
                    data=dict(author='tester',
                              title=f'test {i}')
                )
                self.assertEqual(res.status_code, 200)
                self.assertTrue(res.json.get('done'))

        self.assertEqual(Citation.query().count(), size+init_size)

    def test_fail_to_register_citation(self):
        res = self.client.post(
            '/api/reference/register',
            data=dict(author='tester'),
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn('error', res.json)


if __name__ == '__main__':
    unittest.main()
