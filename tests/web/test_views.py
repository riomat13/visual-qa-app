#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock
import logging
from datetime import datetime

from functools import partial
import tempfile

from flask import g

from main.settings import set_config
set_config('test')

from main.settings import Config
from main.web.app import create_app
from main.orm.db import Base
from main.models.base import User
# from main.models.data import WeightFigure
from main.models.web import Update, Citation

logging.disable(logging.CRITICAL)


admin = {
    'username': 'testcase',
    'email': 'test@example.com',
    'password': 'pwd',
}


class _Base(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        self.app_context = app.app_context()
        self.app_context.push()

        self.client = app.test_client()
        from main.orm.db import engine
        self.engine = engine
        Base.metadata.create_all(self.engine)

        # set up user for authentication without admin
        user = User(username=admin['username'],
                    email=admin['email'],
                    password=admin['password'])
        user.save()

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.app_context.pop()

    def auth_login(self):
        user = User.query().filter_by(username=admin['username']).first()
        user.is_admin = True
        user.save()
        self.login()

    def login(self):
        self.client.post(
            '/login',
            data=dict(username=admin['username'],
                      password=admin['password'],
                      email=admin['email'])
        )

    def logout(self):
        user = User.query().filter_by(username=admin['username']).first()
        if user.is_admin:
            user.is_admin = False
            user.save()
        self.client.get('/logout')

    def check_status_code_with_admin(self, path, method='GET', **kwargs):
        user = User.query() \
            .filter_by(username=admin['username']) \
            .first()

        if method == 'GET':
            req = self.client.get
            target_code = 200
        elif method == 'POST':
            req = partial(self.client.post, data=kwargs)
            target_code = 302
        elif method == 'PUT':
            req = partial(self.client.put, data=kwargs)
            target_code = 302

        # redirect without login
        response = req(path)
        self.assertEqual(response.status_code, 302)

        # permission denied if not admin
        self.login()
        response = req(path)
        self.assertEqual(response.status_code, 403)
        self.logout()

        # can access with admin
        user.is_admin = True
        user.save()
        self.login()

        response = req(path)
        self.assertEqual(response.status_code, target_code)

        self.logout()
        user.is_admin = False
        user.save()

        return response


class GeneralBaseViewResponseTest(_Base):

    def test_index_view(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction_view(self):
        response = self.client.get('/prediction')
        self.assertEqual(response.status_code, 200)

    def test_log_in_and_out(self):
        res = self.client.post(
            '/login',
            data=dict(username='test',
                      email='invalid@example.com',
                      password='pwd')
        )
        # if incorrect, redirect to login page again
        self.assertEqual(res.status_code, 302)
        self.assertTrue(res.location.endswith('login'))

        res = self.client.post(
            '/login',
            data=dict(username=admin['username'],
                      email=admin['email'],
                      password=admin['password'])
        )
        self.assertEqual(res.status_code, 302)
        # has to be jump to the index page
        self.assertEqual(res.location, 'http://localhost/')

        self.assertEqual(g.user.username, admin['username'])

        res = self.client.get('/logout')
        # user information is removed
        self.assertIsNone(g.get('user'))


class PredictionTest(_Base):

    def test_upload_image(self):
        # store image data into buffer temporarily
        with tempfile.NamedTemporaryFile() as tmp:
            #test_data = io.BytesIO(b'test data')
            response = self.client.post(
                '/prediction',
                content_type='multipart/form-data',
                data=dict(
                    action='upload',
                    file=(tmp, tmp.name)
                )
            )
        self.assertEqual(response.status_code, 200)

    @patch('main.web.views.run_model')
    @patch('main.web.views.asyncio.run')
    def test_return_with_prediction(self, mock_run, mock_run_model):
        test_sent = 'this is test sentence'
        # mock prediction result
        mock_run.return_value = test_sent, 1000

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

        data = response.data.decode()

        # check page contents
        self.assertEqual(response.status_code, 200)
        self.assertIn('some question', data)
        self.assertIn(test_sent, data)

    # TODO: add attention weight figure
    # @patch('main.web.WeightFigure')
    # @patch('main.web.views.run_model')
    # @patch('main.web.views.asyncio.run')
    # def test_figure_is_passed_to_template(self, mock_run, mock_run_model, WeightFigure):
    #     test_sent = 'this is test sentence'

    #     # mock prediction result
    #     mock_run.return_value = test_sent, w.id

    #     self.test_upload_image()
    #     response = self.client.post(
    #         '/prediction',
    #         data=dict(
    #             action='Submit',
    #             question='some question',
    #         ),
    #         follow_redirects=True,
    #     )

    #     data = response.data.decode()
    #     self.assertRegex(data, r'<img src="[\w/_]+testfile">')
    #     WeightFigure.assert_called_once()


class NoteViewTest(_Base):

    @patch('main.web.views.Update.query')
    @patch('main.web.views.Citation.query')
    def test_note_view(self, mock_cits, mock_updates):
        mock_update = mock_updates.return_value
        mock_cit = mock_cits.return_value

        mock_updates.return_value.order_by.return_value.all.return_value = [mock_update]
        mock_cits.return_value.all.return_value = [mock_cit]

        mock_update.to_dict.return_value = dict(
            id=1,
            content='test update',
            update_at=datetime.utcnow()
        )
        mock_cit.to_dict.return_value = dict(
            id=1,
            author='tester',
            title='test title',
            year=1990,
        )

        response = self.client.get('/note')
        self.assertEqual(response.status_code, 200)

        self.assertIn(b'test update', response.data)
        self.assertIn(b'test title', response.data)


class UpdateFormViewTest(_Base):

    def test_update_register_view(self):
        self.check_status_code_with_admin('/update/register')

        self.auth_login()
        res = self.client.post(
            '/update/register',
            data=dict(
                content='test content',
                summary='test summary',
            )
        )
        self.assertEqual(res.status_code, 302)

        update = Update.query().filter_by(content='test content').first()
        self.assertEqual(update.summary, 'test summary')

    def test_update_list_all(self):
        content = 'this is a test update'
        summary = 'this is a test update for unittest'

        update = Update(content=content, summary=summary)
        update.save()

        res = self.check_status_code_with_admin('/update/items/all')
        data = res.data.decode()

        self.assertIn(update.content, data)

    def test_update_list_view(self):
        content = 'this is a test update'
        summary = 'this is a test update for unittest'

        update = Update(content=content, summary=summary)
        update.save()

        id_ = update.id

        path = f'/update/item/{id_}'

        res = self.check_status_code_with_admin(path)
        data = res.data.decode()

        self.assertIn(content, data)
        self.assertIn(summary, data)


    def test_update_edit_view(self):
        update = Update(content='test')
        update.save()
        id_ = update.id

        path = f'/update/edit/{id_}'

        self.check_status_code_with_admin(path)

        target_content = 'test content'
        self.check_status_code_with_admin(
            path,
            method='PUT',
            content=target_content,
        )

        # check if data is updated
        update_after = Update.get(id_)
        self.assertEqual(update_after.content, target_content)


class ReferenceFormViewTest(_Base):

    def test_reference_register_view(self):
        self.check_status_code_with_admin('/reference/register')

        self.auth_login()
        res = self.client.post(
            '/reference/register',
            data=dict(
                author='tester',
                title='test title',
                year=1990
            )
        )
        self.assertEqual(res.status_code, 302)

        cit = Citation.query().filter_by(author='tester').first()
        self.assertEqual(cit.title, 'test title')
        self.assertEqual(cit.year, 1990)

    def test_reference_list_all(self):
        author = 'test_author'
        title = 'test_title'
        summary = 'this is a test summary'
        cite = Citation(author=author, title=title, summary=summary)
        cite.save()

        res = self.check_status_code_with_admin('/reference/items/all')
        data = res.data.decode()

        self.assertIn(cite.author, data)
        self.assertIn(cite.title, data)

    def test_reference_item_view(self):
        author = 'test_author'
        title = 'test_title'
        summary = 'this is a test summary'
        cite = Citation(author=author, title=title, summary=summary)
        cite.save()
        id_ = cite.id
        path = f'/reference/item/{id_}'

        # check if accessible only if user is admin
        res = self.check_status_code_with_admin(path)
        data = res.data.decode()

        self.assertIn(author, data)
        self.assertIn(title, data)
        self.assertIn(summary, data)

    def test_reference_edit_view(self):
        cite = Citation(author='tester', title='test')
        cite.save()
        id_ = cite.id

        path = f'/reference/edit/{id_}'

        # check if accessible only if user is admin
        self.check_status_code_with_admin(path)

        target_title = 'test title'
        self.check_status_code_with_admin(
            path,
            method='PUT',
            author='tester',
            title=target_title,
        )

        # check if data is updated
        cite_after = Citation.get(id_)
        self.assertEqual(cite_after.title, target_title)


if __name__ == '__main__':
    unittest.main()
