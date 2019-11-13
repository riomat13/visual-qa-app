#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import io

from main.web.app import create_app


class GeneralBaseViewResponseTest(unittest.TestCase):
    def setUp(self):
        app = create_app('test')
        app_context = app.app_context()
        app_context.push()

        self.app = app
        self.app_context = app_context
        self.client = app.test_client()

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
        app = create_app('test')
        app_context = app.app_context()
        app_context.push()

        self.app = app
        self.app_context = app_context
        self.client = app.test_client()

    def tearDown(self):
        self.app_context.pop()

    def test_upload_image(self):
        # store image data into buffer temporarily
        with open('./main/web/static/media/tests/test_img1.jpg', 'rb') as f:
            test_img = io.BytesIO(f.read())
        response = self.client.post(
            '/prediction',
            content_type='multipart/form-data',
            data=dict(
                action='upload',
                file=(test_img, 'test_img.jpg')
            )
        )
        self.assertEqual(response.status_code, 200)

    def test_return_with_prediction(self):
        # TODO: check return result

        with self.app.test_request_context(
                '/prediction', data={'image': 'test_img.jpg'}):

            response = self.client.post(
                '/prediction',
                data=dict(
                    action='Submit',
                )
            )
            self.assertEqual(response.status_code, 200)
