#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
from unittest.mock import patch

import random
from functools import namedtuple
import logging

logging.disable(logging.CRITICAL)


from flask import g, session

from main.settings import set_config
set_config('test')

from main.settings import Config
from main.web.app import create_app
from main.orm.db import Base
from main.orm.models.base import User as DBUser
from main.web.auth import generate_token, get_user_by_token, verify_user


USERNAME = 'testcase'
EMAIL = 'test@test.com'
PASSWORD = 'pwd'

_User = namedtuple('User', 'id, username, email, password, is_admin, verify_password')


def User(username=USERNAME, email=EMAIL, password=PASSWORD, is_admin=False, verify_password=None):
    return _User(id=random.randint(1, 10),
                 username=username,
                 email=email,
                 password=password,
                 is_admin=is_admin,
                 verify_password=verify_password)


class _BaseWeb(unittest.TestCase):

    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(engine)

        u = DBUser(username='test',
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


class AuthTokenTest(unittest.TestCase):

    def setUp(self):
        self.user = User(username=USERNAME,
                         email=EMAIL,
                         password=PASSWORD)

    def generate_token(self):
        token = generate_token(self.user)
        self.assertIsNotNone(token)
        self.assertTrue(isinstance(token, bytes))
        return token

    @patch('main.web.auth.User')
    def test_token_can_be_validated(self, mock_User):
        mock_User.get.return_value = User()
        token = self.generate_token()
        user = get_user_by_token(token)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, USERNAME)


class UserVerificationTest(_BaseWeb):

    def test_user_is_registered(self):
        self.assertFalse(
            verify_user('invalid', 'invalid', 'invalid')
        )

        self.assertTrue(
            verify_user(username='test',
                        email='test@example.com',
                        password='pwd')
        )

    def test_set_user_after_login(self):
        self.login()
        self.client.get('/')
        user = g.user
        self.assertEqual(user.username, 'test')

        self.logout()
        user = g.user
        self.assertIsNone(user)

    def test_verify_without_email(self):
        self.assertTrue(
            verify_user(username='test',
                        password='pwd')
        )
