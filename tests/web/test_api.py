#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock

import logging
from datetime import datetime
from collections import namedtuple
from functools import partial, wraps
import json
import io
import random

# disable login_required
def mock_login_required(f=None, *, admin=False):
    if f is None:
        return partial(mock_login_required, admin=admin)

    # identity function
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

patch('main.web.auth.login_required', mock_login_required).start()


from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.web.auth import generate_token, get_user_by_token
from main.orm.db import Base, reset_db

logging.disable(logging.CRITICAL)

USERNAME = 'testcase'
EMAIL = 'test@test.com'
PASSWORD = 'pwd'


class _Base(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        self.app_context = app.app_context()
        self.app_context.push()

        client = app.test_client()
        self.client = client

    def tearDown(self):
        self.app_context.pop()


class PredictionTest(_Base):

    @patch('main.web.api.os.path.splitext')
    @patch('main.web.api.Image')
    def test_upload_image(self, _Image, mock_splitext):
        test_data = io.BytesIO(b'test_data')
        mock_splitext.return_value = 'test', '.jpg'

        _Image.return_value.id = 100
        _Image.query.return_value.filter_by.return_value.count.return_value = 0

        # store image data into buffer temporarily
        res = self.client.post(
            '/api/upload/image',
            data=dict(file=(test_data, 'test.jpg')),
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 201)
        _Image.assert_called_once_with(filename='test.jpg')

        data = res.json.get('data')
        self.assertEqual(data.get('img_id'), 100)

        # use the same file name as before
        test_data = io.BytesIO(b'test_data')
        _Image.query.return_value.filter_by.return_value.count.return_value = 1

        res = self.client.post(
            '/api/upload/image',
            data=dict(file=(test_data, 'test.jpg')),
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 201)

        data = res.json.get('data')
        self.assertEqual(data.get('img_id'), 100)
        # used in third-party lib as well, thus not only once
        mock_splitext.assert_called_with('test.jpg')


    @patch('main.web.api.os.path.splitext')
    @patch('main.web.api.Image')
    def test_upload_image_with_long_filename(self, _Image, mock_splitext):
        test_data = io.BytesIO(b'test_data')
        # set long filename
        filename = ''.join([chr(random.randint(ord('a'), ord('z'))) for _ in range(128)]) + '.jpg'
        mock_splitext.return_value = filename[:-4], '.jpg'

        _Image.return_value.id = 100
        _Image.query.return_value.filter_by.return_value.count.return_value = 0

        # store image data into buffer temporarily
        res = self.client.post(
            '/api/upload/image',
            data=dict(file=(test_data, filename)),
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 201)

        target_filename = filename[:64-4] + '.jpg'
        _Image.assert_called_once_with(filename=target_filename)

        data = res.json.get('data')
        self.assertEqual(data.get('img_id'), 100)

        # use the same file name as before
        test_data = io.BytesIO(b'test_data')
        _Image.query.return_value.filter_by.return_value.count.return_value = 1

    @patch('main.web.api.Image')
    @patch('main.web.api.run_model')
    @patch('main.web.api.asyncio.run')
    def test_return_with_prediction(self, mock_run, mock_run_model, Image):
        test_sent = 'this is test sentence'
        # mock prediction result
        mock_run.return_value = test_sent, 1000

        # unregistered image -> return error
        Image.get.return_value = None

        res = self.client.post(
            '/api/prediction',
            data=json.dumps(dict(img_id=1,
                                 question='some question')),
            content_type='application/json',
        )

        self.assertEqual(res.status_code, 400)
        self.assertEqual('error', res.json.get('status'))

        Image.get.return_value = MagicMock()
        Image.get.return_value.filename = 'test.jpg'
        res = self.client.post(
            '/api/prediction',
            data=json.dumps(dict(img_id=1,
                                 question='some question')),
            content_type='application/json',
        )

        data = res.json.get('data')

        # check page contents
        self.assertEqual(res.status_code, 200)
        self.assertIn('success', res.json.get('status'))
        self.assertIn(test_sent, data.get('prediction'))

    @unittest.skip
    @patch('main.web.api.run_model')
    @patch('main.web.api.asyncio.run')
    def test_figure_is_passed_to_template(self, mock_run, mock_run_model):
        # TODO: attention weights data
        test_sent = 'this is test sentence'
        w = WeightFigure()
        w.filename = 'testfile'
        w.save()

        # mock prediction result
        mock_run.return_value = test_sent, w.id

        self.test_upload_image()
        res = self.client.post(
            '/api/prediction',
            data=json.dumps(dict(image_id=1,
                                 question='some question')),
            content_type='application/json',
        )

        data = res.json


class NoteTest(_Base):

    @patch('main.web.api.Update.query')
    @patch('main.web.api.Citation.query')
    def test_note_return_data(self, mock_cits, mock_updates):
        mock_update = mock_updates.return_value
        mock_cit = mock_cits.return_value

        mock_updates.return_value.order_by.return_value.all.return_value = [mock_update]
        mock_cits.return_value.all.return_value = [mock_cit]

        mock_update.to_dict.return_value = dict(
            id=1,
            content='test update',
            update_at=datetime.utcnow()
        )
        mock_cit.to_dict.return_value = dict(
            id=1,
            author='tester',
            title='test title',
            year=1990,
        )

        res = self.client.get('/api/note')
        self.assertEqual(res.status_code, 200)

        data = res.json.get('data')

        self.assertIn('updates', data)
        self.assertIn('references', data)

        self.assertEqual(1, len(data.get('updates')))
        self.assertEqual(1, len(data.get('references')))


class UpdateItemTest(_Base):

    @patch('main.web.api.Update')
    def test_extract_update_list(self, Update):
        size = 4
        m = MagicMock()
        m.to_dict.return_value = 'test'
        Update.query.return_value.all.return_value = [m] * size
        
        res = self.client.post(
            '/api/update/items/all',
            data=json.dumps(dict(token='')),
            content_type='application/json',
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual('success', res.json.get('status'))
        data = res.json.get('data')
        self.assertEqual(len(data), size)

    @patch('main.web.api.Update')
    def test_register_new_update(self, Update):
        test_content = 'test_content'
        res = self.client.post(
            '/api/update/register',
            data=json.dumps(dict(token='',
                                 content=test_content)),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual('success', res.json.get('status'))

        Update.assert_called_once_with(content=test_content,
                                       summary=None)
        Update.return_value.save.assert_called()

    def test_fail_to_register_update(self):
        res = self.client.post(
            '/api/update/register',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn('error', res.json)

    @patch('main.web.api.Update')
    def test_edit_update_item(self, Update):
        id_ = 10
        update = MagicMock()
        Update.get.return_value = update
        target_content = 'test content'

        res = self.client.put(
            f'/api/update/edit/{id_}',
            data=json.dumps(dict(token='',
                                 content=target_content)),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 201)

        # check if data is updated
        update.save.assert_called_once_with()


@patch('main.web.api.Citation')
class ReferenceItemTest(_Base):

    def test_register_reference_item(self, Citation):
        author = 'tester'
        title = 'test title'
        year = 1990

        res = self.client.post(
            '/api/reference/register',
            data=json.dumps(dict(token='',
                                 author=author,
                                 title=title,
                                 year=year)),
            content_type='application/json',
        )

        self.assertEqual(res.status_code, 201)

        Citation.assert_called_once_with(author=author,
                                         title=title,
                                         year=year,
                                         url=None,
                                         summary=None)

    def test_reference_list_all(self, Citation):
        size = 4
        m = MagicMock()
        m.to_dict.return_value = {'title': 'test'}

        Citation.query.return_value.all.return_value = [m] * size

        res = self.client.post(
            '/api/reference/items/all',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)

        data = res.json.get('data')
        self.assertEqual(len(data), size)

    def test_reference_item(self, Citation):
        id_ = 10
        m = MagicMock()
        m.to_dict.return_value = {'title': 'test'}
        Citation.get.return_value = m

        res = self.client.post(
            f'/api/reference/item/{id_}',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )

        data = res.json.get('data')
        self.assertIn('title', data)
        Citation.get.assert_called_once_with(id_)

        # if item could not be found, return empty
        Citation.get.return_value = None
        res = self.client.post(
            f'/api/reference/item/{id_}',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        data = res.json.get('data')
        self.assertEqual(len(data), 0)

    def test_reference_edit(self, Citation):
        m = MagicMock()
        Citation.get.return_value = m
        id_ = 10

        target_title = 'test title'

        res = self.client.put(
            f'/api/reference/edit/{id_}',
            data=json.dumps(dict(token='',
                                 title=target_title)),
            content_type='application/json'
        )

        Citation.get.assert_called_once_with(id_)

        # check if model is saved
        m.save.assert_called_once()


@patch('main.web.api.MLModel')
class ModelListTest(_Base):

    def test_extract_model_list(self, Model):
        data_size = 4
        target = {'model': 'test'}
        
        m = MagicMock()
        m.to_dict.return_value = target

        Model.query.return_value.all.return_value = \
            [m for _ in range(data_size)]

        res = self.client.post(
            '/api/models/all',
            data=json.dumps(dict(token='')),
            content_type='application/json',
        )

        self.assertEqual(res.status_code, 200)

        json_data = res.json.get('data')

        self.assertEqual(len(json_data), 4)
        
        for data in json_data:
            self.assertEqual(data, target)

    def test_register_model(self, Model):
        model_name = 'test_model'
        type = 'test_type'
        category = 'test_cat'

        res = self.client.post(
            '/api/register/model',
            data=json.dumps(dict(token='',
                                 name=model_name,
                                 type=type,
                                 category=category)),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 201)

        self.assertEqual(res.json.get('status'), 'success')

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
            data=json.dumps(dict(token='',
                                 name='test')),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 400)

        # not successed process
        self.assertEqual('error', res.json.get('status'))
        self.assertTrue('error' in res.json)

    def test_extract_model_info_by_id(self, Model):
        id_ = 11
        model_name = 'test_model'

        m = MagicMock()
        m.to_dict.return_value = {'name': model_name}
        Model.get.return_value = m

        res = self.client.post(
            f'/api/model/{id_}',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )

        data = res.json.get('data')
        self.assertEqual(data.get('name'), model_name)
        Model.get.assert_called_once_with(id_)
        m.to_dict.assert_called_once()

        # test with non-exist id
        Model.get.return_value = None

        res = self.client.post(
            '/api/model/1000',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        data = res.json.get('data')
        # this must be empty
        self.assertEqual(data, {})


@patch('main.web.api.run_model')
@patch('main.web.api.asyncio.run')
class QuestionTypeTest(_Base):

    def test_predict_question_type_request(self, mock_async_run, mock_run_model):
        test_return = 'model'
        mock_run_model.return_value = test_return
        mock_async_run.return_value = 'test'

        q = 'is this test?'
        res = self.client.post(
            '/api/predict/question_type',
            data=json.dumps(dict(question=q)),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 201)

        # check data is properly passed
        data = res.json.get('data')
        self.assertEqual(data.get('question'), q)

        # model is called with empty string and the given string
        mock_run_model.assert_called_once_with('', q)

        # execute server by run_model function
        mock_async_run.assert_called_once_with(test_return)

    def test_error_handling_question_type_prediction(self,
                                                     mock_async_run,
                                                     mock_run_model):
        # error code
        mock_async_run.return_value = '<e>'

        res = self.client.post(
            '/api/predict/question_type',
            data=json.dumps(dict(question='invalid')),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 400)

        data = res.json
        self.assertIn('message', data)
        self.assertEqual('error', data.get('status'))


@patch('main.web.api.RequestLog')
class ExtractRequestLogsTest(_Base):

    def test_extract_all_logs(self, Log):
        size = 4

        m = MagicMock()
        m.to_dict.return_value = {'key': 'test'}

        Log.query.return_value.all.return_value = [m] * size

        res = self.client.post(
            '/api/logs/requests',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

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

        res = self.client.post(
            '/api/logs/requests',
            data=json.dumps(dict(token='',
                                 image_id=1,
                                 question_type='test')),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('key'), 'value')


@patch('main.web.api.RequestLog')
class ExtractSingleRequestLogTest(_Base):

    def test_extract_log_by_id(self, Log):
        Log.get.return_value.to_dict.return_value = 'test'
        id_ = 10

        res = self.client.post(
            f'/api/logs/request/{id_}',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        Log.get.assert_called_once_with(id_)
        self.assertEqual('test', data)

    @patch('main.web.api.WeightFigure')
    @patch('main.web.api.Image')
    @patch('main.web.api.Question')
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

        res = self.client.post(
            f'/api/logs/qa/{target_id}',
            data=json.dumps(dict(token='')),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        Log.get.assert_called_once_with(target_id)
        Img.get.assert_called_once_with(img_id)
        Q.get.assert_called_once_with(q_id)
        Fig.get.assert_called_once_with(fig_id)

        self.assertEqual(data.get('request_id'), target_id)
        self.assertEqual(data.get('question'), 'test_question')
        self.assertEqual(data.get('prediction'), 'test_prediction')
        self.assertEqual(data.get('image'), 'test_image')
        self.assertEqual(data.get('figure'), 'test_fig')


@patch('main.web.api.RequestLog')
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
        res = self.client.post(
            '/api/logs/predictions',
            data=json.dumps(dict()),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 200)
        data = res.json.get('data')

        self.assertEqual(len(data), size)

    def test_extract_scores_by_question_type(self, Log):
        size = 5

        score = MagicMock(rate=3,
                          prediction='test',
                          probability=None,
                          answer='ans',
                          predicted_time=None)

        m1 = Log.query.return_value
        m2 = m1.filter_by.return_value
        m2.all.return_value = \
            [MagicMock(id=i, score=score) for i in range(1, size + 1)]

        # send request and extract data
        res = self.client.post(
            '/api/logs/predictions',
            data=json.dumps(dict(question_type='test')),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 200)

        self.assertEqual(res.json.get('status'), 'success')
        data = res.json.get('data')

        self.assertEqual(len(data), size)

        for log in data:
            self.assertEqual(log.get('answer'), 'ans')

        m1.filter_by.assert_called_once_with(question_type='test')

        # if error occured during processing logs
        m2.all.side_effect = ValueError()

        res = self.client.post(
            '/api/logs/predictions',
            data=json.dumps(dict(question_type='test')),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 400)


if __name__ == '__main__':
    unittest.main()
