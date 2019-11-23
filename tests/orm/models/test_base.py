#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .base import _Base

from main.orm.models.base import User


class UserModelTest(_Base):

    def test_password_setter(self):
        u = User(username='test',
                 email='some_email@sample.com',
                 password='pwd')
        u.save()

        user = User.query().filter_by(username='test').first()

        with self.assertRaises(AttributeError):
            user.password

        self.assertTrue(user.verify_password('pwd'))

        user.password = 'new_pwd'
        self.assertTrue(user.verify_password('new_pwd'))

