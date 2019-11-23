#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from main.settings import set_config
set_config('test')

import io

from main.web.app import create_app
from main.orm.db import Base
from main.orm.models.base import User


class GeneralBaseViewResponseTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.client = self.app.test_client()
        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.app_context.pop()

    def test_index_view(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction_view(self):
        response = self.client.get('/prediction')
        self.assertEqual(response.status_code, 200)

    def test_log_in_and_out(self):
        uname = 'test'
        email = 'test@example.com'
        password = 'pwd'

        user = User(username=uname,
                    email=email,
                    password=password)
        user.save()

        res = self.client.post(
            '/login',
            data=dict(username='test',
                      email='invalid@example.com',
                      password='pwd')
        )
        # if incorrect, redirect to login page again
        self.assertEqual(res.status_code, 302)
        self.assertTrue(res.location.endswith('login'))

        res = self.client.post(
            '/login',
            data=dict(username=uname,
                      email=email,
                      password=password)
        )
        self.assertEqual(res.status_code, 302)
        # has to be jump to the index page
        self.assertEqual(res.location, 'http://localhost/')


class PredictionTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

    def tearDown(self):
        self.app_context.pop()

    def test_upload_image(self):
        # store image data into buffer temporarily
        test_data = io.BytesIO(b'test data')
        response = self.client.post(
            '/prediction',
            content_type='multipart/form-data',
            data=dict(
                action='upload',
                file=(test_data, 'test.jpg')
            )
        )
        self.assertEqual(response.status_code, 200)

    @patch('main.web.views.run_model')
    @patch('main.web.views.asyncio.run')
    def test_return_with_prediction(self, mock_run, mock_run_model):
        test_sent = 'this is test sentence'
        # mock prediction result
        mock_run.return_value = test_sent

        response = self.client.post(
            '/prediction',
            data=dict(
                action='Submit',
                question='some question',
            ),
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        # since no image is provided, pop up alert
        self.assertIn('Image is not provided', response.data.decode())

        # upload image to make it work
        self.test_upload_image()

        response = self.client.post(
            '/prediction',
            data=dict(
                action='Submit',
                question='some question',
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(test_sent, response.data.decode())
