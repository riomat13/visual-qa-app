#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from .base import _Base
from main.models.web import Update, Citation


class UpdateModelTest(_Base):

    def test_save_and_get_data(self):
        content = 'this is the first content.'
        update = Update(content=content)
        update.save()

        data = Update.query().first()

        self.assertEqual(data.id, update.id)
        self.assertEqual(data.content, content)

    def test_update_item_to_dict(self):
        content = 'this is the first content.'
        update = Update(content=content)
        update.save()

        data = update.to_dict(date=True)
        self.assertIn('content', data)
        self.assertEqual(content, data.get('content'))

        self.assertIn('update_at', data)
        self.assertTrue(isinstance(data.get('update_at'), str))
        dates = data.get('update_at').split('/')
        self.assertEqual(3, len(dates))

        for d in dates:
            self.assertTrue(d.isdigit())

        self.assertIn('summary', data)


class CitationModelTest(_Base):

    def test_save_and_get_data(self):
        author = 'tester'
        title = 'test content'
        invalid_year = 1980

        with self.assertRaises(ValueError):
            cite = Citation(author='invalid', title='invalid', year=invalid_year)

        year = 1994
        cite = Citation(author=author, title=title, year=year)
        cite.save()

        data = Citation.query().first()

        self.assertEqual(data.id, cite.id)
        self.assertEqual(data.author, author)


if __name__ == '__main__':
    unittest.main()
