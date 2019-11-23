#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
import logging

logging.disable(logging.CRITICAL)


from flask import g, session

from main.settings import set_config
set_config('test')

from main.settings import Config
from main.web.app import create_app
from main.orm.db import Base
from main.orm.models.base import User
from main.web.auth import verify_user


class _BaseWeb(unittest.TestCase):

    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

        u = User(username='test',
                 email='test@example.com',
                 password='pwd')
        u.save()

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.app_context.pop()

    def login(self):
        return self.client.post(
            '/login',
            data=dict(username='test',
                      email='test@example.com',
                      password='pwd'),
            follow_redirects=True
        )

    def logout(self):
        return self.client.get('/logout',
                               follow_redirects=True)


class UserVerificationTest(_BaseWeb):

    def test_user_is_registered(self):
        self.assertIsNone(
            verify_user('invalid', 'invalid', 'invalid')
        )

        user = verify_user(username='test',
                           email='test@example.com',
                           password='pwd')
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'test')

    def test_set_user_after_login(self):
        self.login()
        self.client.get('/')
        user = g.user
        self.assertEqual(user.username, 'test')

        self.logout()
        user = g.user
        self.assertIsNone(user)

    def test_verify_without_email(self):
        user = verify_user(username='test',
                          password='pwd')
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'test')
