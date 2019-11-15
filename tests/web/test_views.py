#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import io

from main.web.app import create_app


class GeneralBaseViewResponseTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.client = self.app.test_client()

    def tearDown(self):
        self.app_context.pop()

    def test_index_view(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction_view(self):
        response = self.client.get('/prediction')
        self.assertEqual(response.status_code, 200)


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
