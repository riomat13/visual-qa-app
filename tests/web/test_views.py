#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock

import io

from flask import g

from main.settings import set_config
set_config('test')

from main.web.app import create_app
from main.orm.db import Base
from main.orm.models.base import User
from main.orm.models.web import Note, Citation


admin = {
    'username': 'test',
    'email': 'test@example.com',
    'password': 'pwd',
}


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app = create_app('test')
        cls.app_context = app.app_context()
        cls.app_context.push()

        cls.client = app.test_client()
        from main.orm.db import engine
        cls.engine = engine
        Base.metadata.create_all(cls.engine)

        # set up user for authentication
        user = User(username=admin['username'],
                    email=admin['email'],
                    password=admin['password'])
        user.save()

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(cls.engine)
        cls.app_context.pop()


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

    @patch('main.web.views.Image.save')
    def test_upload_image(self, mock_img_save):
        # store image data into buffer temporarily
        test_data = io.BytesIO(b'test data')
        response = self.client.post(
            '/prediction',
            content_type='multipart/form-data',
            data=dict(
                action='upload',
                file=(test_data, 'test.jpg')
            )
        )
        self.assertEqual(response.status_code, 200)
        mock_img_save.assert_called_once()

    @patch('main.web.views.Question.save')
    @patch('main.web.views.run_model')
    @patch('main.web.views.asyncio.run')
    def test_return_with_prediction(self, mock_run, mock_run_model, mock_q_save):
        test_sent = 'this is test sentence'
        # mock prediction result
        mock_run.return_value = test_sent

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

        self.assertEqual(response.status_code, 200)
        self.assertIn(test_sent, response.data.decode())
        mock_q_save.assert_called_once()


class NoteViewTest(_Base):

    @patch('main.web.views.Note.query')
    @patch('main.web.views.Citation.query')
    def test_note_view(self, mock_cits, mock_notes):
        mock_note = mock_notes.return_value
        mock_ref = mock_cits.return_value

        mock_notes.return_value.all.return_value = [mock_note]
        mock_cits.return_value.all.return_value = [mock_ref]

        mock_note.to_dict.return_value = 'test_notes'
        mock_ref.to_dict.return_value = 'test_refs'

        response = self.client.get('/note')
        self.assertEqual(response.status_code, 200)

        self.assertIn(b'test_notes', response.data)
        self.assertIn(b'test_refs', response.data)

if __name__ == '__main__':
    unittest.main()
