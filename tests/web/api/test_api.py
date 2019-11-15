#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.orm.db import Base, engine


class _Base(unittest.TestCase):

    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

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

        response = self.client.get('/api/model_list')
        
        self.assertEqual(response.status_code, 200)

        json_data = response.json

        self.assertEqual(len(json_data), 4)
        
        for data in json_data:
            self.assertEqual(data, target)


class QuestionTypeTest(_Base):

    @patch('main.web.api._api.run_model')
    @patch('main.web.api._api.asyncio.run')
    def test_send_question_request(self, mock_async_run, mock_run_model):
        test_return = 'model'
        mock_run_model.return_value = test_return
        mock_async_run.return_value = 'test'

        q = 'is this test?'
        response = self.client.post('/api/question_type',
                                    data={'question':q})

        # check data is properly passed
        data = response.json
        self.assertEqual(data.get('question'), q)

        # model is called with empty string and the given string
        mock_run_model.assert_called_once_with('', q)

        # execute server by run_model function
        mock_async_run.assert_called_once_with(test_return)
