#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock

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

        self.client = app.test_client()

    def tearDown(self):
        self.app_context.pop()


class AuthorizationTest(_Base):
    def setUp(self):
        super(AuthorizationTest, self).setUp()

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

        self.client.get = partial(client.get, headers=headers)
        self.client.post = partial(client.post, headers=headers)

    def tearDown(self):
        self.app_context.pop()
        Base.metadata.drop_all(self.engine)


@patch('main.web.api._api._is_authorized', new=MagicMock())
@patch('main.web.api._api.MLModel')
class ModelListTest(_Base):

    def test_extract_model_list(self, Model):
        data_size = 4
        target = {'model': 'test'}
        m = MagicMock()
        m.to_dict.return_value = target
        
        Model.query.return_value.all.return_value = \
            [m for _ in range(data_size)]

        response = self.client.get('/api/models/all')

        self.assertEqual(response.status_code, 200)

        json_data = response.json.get('data')

        self.assertEqual(len(json_data), 4)
        
        for data in json_data:
            self.assertEqual(data, target)

    def test_register_model(self, Model):
        model_name = 'test_model'
        type = 'test_type'
        category = 'test_cat'
        res = self.client.post(
            '/api/register/model',
            data=(dict(name=model_name,
                       type=type,
                       category=category)),
        )

        self.assertEqual(res.status_code, 201)

        Model.assert_called_once_with(name=model_name,
                                      type=type,
                                      category=category,
                                      module=None,
                                      object=None,
                                      path=None,
                                      metrics=None,
                                      score=None)
        Model.return_value.save.assert_called_once()

    def test_register_model_handle_invalid_input(self, Model):
        Model.return_value.save.side_effect = ValueError()
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

    def test_extract_model_info_by_id(self, Model):
        id_ = 11
        model_name = 'test_model'

        m = MagicMock()
        m.to_dict.return_value = {'name': model_name}
        Model.get.return_value = m

        res = self.client.get(f'/api/model/{id_}')

        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        self.assertEqual(data.get('name'), model_name)
        Model.get.assert_called_once_with(id_)
        m.to_dict.assert_called_once()

        # test with non-exist id
        Model.get.return_value = None

        res = self.client.get(f'/api/model/1000')
        data = res.json.get('data')
        # this must be empty
        self.assertEqual(data, {})


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


@patch('main.web.api._api._is_authorized', new=MagicMock())
@patch('main.web.api._api.RequestLog')
class ExtractRequestLogsTest(_Base):

    def test_extract_all_logs(self, Log):
        size = 4

        m = MagicMock()
        m.to_dict.return_value = {'key': 'test'}

        Log.query.return_value.all.return_value = [m] * size

        response = self.client.get('/api/logs/requests')
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'test')

    def test_extract_question_type_logs(self, Log):
        size = 4

        m = MagicMock()
        m.to_dict.return_value = {'key': 'value'}

        m1 = Log.query.return_value
        m2 = m1.filter.return_value
        m3 = m2.filter.return_value
        m3.all.return_value = [m] * size

        response = self.client.post('/api/logs/requests',
                                    data=dict(image_id=1,
                                              question_type='test'))
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'value')

    def test_extract_non_registered_logs(self, Log):
        res = self.client.post('/api/logs/requests',
                               data=dict(image_id=1,
                                         question_type='test'))
        self.assertEqual(res.status_code, 200)

        # should return empty
        self.assertEqual(len(res.json.get('data')), 0)


@patch('main.web.api._api._is_authorized', new=MagicMock())
@patch('main.web.api._api.RequestLog')
class ExtractSingleRequestLogTest(_Base):

    def test_extract_log_by_id(self, Log):
        Log.get.return_value.to_dict.return_value = 'test'
        id_ = 10

        res = self.client.get(f'/api/logs/request/{id_}')
        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        Log.get.assert_called_once_with(id_)
        self.assertEqual('test', data)

    @patch('main.web.api._api.WeightFigure')
    @patch('main.web.api._api.Image')
    @patch('main.web.api._api.Question')
    def test_extract_log_data_by_id(self, Q, Img, Fig, Log):
        img_id = 1
        q_id = 2
        fig_id = 3

        # set up mock log data
        score = MagicMock(prediction='test_prediction')
        log = MagicMock(image_id=img_id,
                        question_id=q_id,
                        fig_id=fig_id,
                        score=score)

        Log.get.return_value = log
        Q.get.return_value.question = 'test_question'
        Img.get.return_value.filename = 'test_image'
        Fig.get.return_value.filename = 'test_fig'

        target_id = 10

        res = self.client.get(f'/api/logs/qa/{target_id}')
        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        Log.get.assert_called_once_with(target_id)
        Img.get.assert_called_once_with(img_id)
        Q.get.assert_called_once_with(q_id)
        Fig.get.assert_called_once_with(fig_id)

        self.assertEqual(target_id, data.get('request_id'))
        self.assertEqual('test_question', data.get('question'))
        self.assertEqual('test_prediction', data.get('prediction'))
        self.assertEqual('test_image', data.get('image'))
        self.assertEqual('test_fig', data.get('figure'))


@patch('main.web.api._api._is_authorized', new=MagicMock())
@patch('main.web.api._api.RequestLog')
class ExtractPredictionScoreLogsTest(_Base):

    def test_extract_all_scores(self, Log):
        size = 4

        req_log = namedtuple('ReqLog', 'id, score')
        score = namedtuple('Score', 'rate, prediction, probability, answer, predicted_time')

        logs = []
        for i in range(1, size << 1):
            if i & 1:
                score_ = score(rate=3,
                               prediction='test',
                               probability=None,
                               answer='tested',
                               predicted_time=None)
            else:
                # dummy data, should not be captured
                score_ = None

            req = req_log(id=i, score=score_)
            logs.append(req)

        assert len(logs) == (size << 1) - 1

        Log.query.return_value.all.return_value = logs

        # send request and extract data
        response = self.client.get('/api/logs/predictions')
        self.assertEqual(response.status_code, 200)
        data = response.json.get('data')

        self.assertEqual(len(data), size)

    def test_extract_scores_by_question_type(self, Log):
        size = 5

        score = MagicMock(rate=3,
                          prediction='pred',
                          probability=None,
                          answer='ans',
                          predicted_time=None)

        m1 = Log.query.return_value
        m2 = m1.filter.return_value
        m2.all.return_value = \
            [MagicMock(id=i, score=score) for i in range(1, size + 1)]

        # send request and extract data
        res = self.client.post(
            '/api/logs/predictions',
            data=dict(question_type='test_q')
        )

        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('prediction'), 'pred')

        m1.filter.assert_called_once_with(Log.question_type.has.return_value)


@patch('main.web.api._api._is_authorized', new=MagicMock())
class UpdateItemTest(_Base):

    @patch('main.web.api._api.Update')
    def test_extract_update_list(self, Update):
        size = 4
        m = MagicMock()
        m.to_dict.return_value = 'test'
        Update.query.return_value.all.return_value = [m] * size

        res = self.client.get('/api/updates/all')

        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')
        self.assertEqual(len(data), size)

    @patch('main.web.api._api.Update')
    def test_register_new_update(self, Update):
        test_content = 'test_content'

        res = self.client.post(
            '/api/update/register',
            data=dict(content=test_content)
        )

        self.assertEqual(res.status_code, 200)

        Update.assert_called_once_with(content=test_content)

        Update.return_value.save.assert_called()

    def test_fail_to_register_update(self):
        res = self.client.post(
            '/api/update/register',
            data={},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn('error', res.json)


@patch('main.web.api._api._is_authorized', new=MagicMock())
@patch('main.web.api._api.Citation')
class ReferenceItemTest(_Base):

    def test_register_reference_item(self, Citation):
        author = 'tester'
        title = 'test title'
        year = '1990'

        res = self.client.post(
            '/api/reference/register',
            data=dict(author=author,
                      title=title,
                      year=year)
        )
        self.assertEqual(res.status_code, 201)

        Citation.assert_called_once_with(author=author,
                                         title=title,
                                         year=year,
                                         url=None)

        Citation.return_value.save.side_effect = ValueError()

        # check if fail to register
        res = self.client.post(
            '/api/reference/register',
            data=dict(author='tester'),
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn('error', res.json)

    def test_reference_list_all(self, Citation):
        size = 4
        m = MagicMock()
        m.to_dict.return_value = {'title': 'test'}

        Citation.query.return_value.all.return_value = [m] * size

        res = self.client.get('/api/references/all')

        self.assertEqual(res.status_code, 200)

        data = res.json.get('data')
        self.assertEqual(len(data), size)


if __name__ == '__main__':
    unittest.main()
