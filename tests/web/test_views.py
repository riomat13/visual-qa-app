#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest

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
