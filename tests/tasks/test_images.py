#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.orm.db import Base
from main.tasks.images import image_process_task
from tests.factory import ImageFactory


class ImageProcessTaskTest(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        self.context = app.app_context()
        self.context.push()
        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.context.pop()

    @patch('main.tasks.images.update_row_image')
    def test_process_image_by_scheduled(self, mock_update):
        factory = ImageFactory()
        data_size = 3
        for _ in range(data_size):
            img = factory()
            img.save()

        res = image_process_task()
        self.assertEqual(len(res), data_size)

    @patch('main.tasks.images.update_row_image')
    def test_process_image_by_ids(self, mock_update):
        factory = ImageFactory()
        data_size = 3
        ids = []

        for _ in range(data_size):
            img = factory()
            img.save()
            ids.append(img.id)

        res = image_process_task(ids)
        self.assertEqual(len(res), data_size)
        self.assertEqual(res, ids)
