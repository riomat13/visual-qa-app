#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from .base import _Base
from main.orm.models.web import Note, Citation


class NoteModelTest(_Base):

    def test_save_and_get_data(self):
        content = 'this is the first content.'
        note = Note(content=content)
        note.save()

        data = Note.query().first()

        self.assertEqual(data.content, content)

class CitationModelTest(_Base):

    def test_save_and_get_data(self):
        author = 'tester'
        name = 'test content'
        invalid_year = 1980

        with self.assertRaises(ValueError):
            cit = Citation(author=author, name=name, year=invalid_year)

        year = 1994
        cit = Citation(author=author, name=name, year=year)
        cit.save()

        data = Citation.query().first()

        self.assertEqual(data.author, author)


if __name__ == '__main__':
    unittest.main()
