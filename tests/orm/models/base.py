#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.orm.db import Base, session_builder


class _Base(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        app_context = app.app_context()
        app_context.push()

        self.app_context = app_context

        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

    def tearDown(self):
        self.app_context.pop()
        Base.metadata.drop_all(self.engine)
