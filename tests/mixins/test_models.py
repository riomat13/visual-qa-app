#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from flask import jsonify

from main.web.app import create_app
from main.orm.db import Base, session_builder
from main.orm.models.ml import MLModel
from main.mixins.models import BaseMixin

Session = session_builder()
session = None


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


class BaseMixinTest(_Base):
    
    def test_save_and_get_data(self):
        # model count before save data
        init_size = MLModel.query().count()

        m = MLModel(name='test',
                    type='cls',
                    category='test',
                    module='main.mixins.models',
                    object='TestCase')

        # save model and added new one
        m.save()
        size = MLModel.query().count()
        self.assertEqual(size - init_size, 1)

        # take data under the current session
        m = MLModel.query().first()

        try:
            jsonify(m.to_dict())
        except TypeError:
            self.fail('Could not apply jsonify()')

        m.delete()
        size = MLModel.query().count()
        self.assertEqual(size, init_size)


if __name__ == '__main__':
    unittest.main()
