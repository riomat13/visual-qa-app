#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from .base import _Base

from main.models.base import User, AppLog


class UserModelTest(_Base):

    def test_password_setter(self):
        u = User(username='testcase',
                 email='some_email@sample.com',
                 password='pwd')
        u.save()

        user = User.query().filter_by(username='testcase').first()

        with self.assertRaises(AttributeError):
            user.password

        self.assertTrue(user.verify_password('pwd'))

        user.password = 'new_pwd'
        self.assertTrue(user.verify_password('new_pwd'))


class AppLogTest(_Base):

    def test_save_and_fetch_log(self):
        size = 10
        target_size = 5
        for i in range(target_size):
            log = AppLog(log_type='success',
                         log_class='TargetException',
                         log_text=f'this is test log {i}')
            log.save()

        id_ = log.id

        for i in range(target_size, size):
            # set earlier time
            log = AppLog(log_type='fail',
                         log_class='SomeException',
                         log_text=f'this is test log {i}',
                         logged_time=datetime(2000, 1, 1))
            log.save()

        log = AppLog.get(id_)
        self.assertEqual(log.log_class, 'TargetException')

        logs = AppLog.fetch_logs(log_type='success')
        self.assertEqual(len(logs), target_size)

        logs = AppLog.fetch_logs(log_class='TargetException')
        self.assertEqual(len(logs), target_size)

        logs = AppLog.fetch_logs(log_time=datetime(2010, 1, 1))
        self.assertEqual(len(logs), target_size)

