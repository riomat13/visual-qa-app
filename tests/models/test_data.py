#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from main.models.data import Image, Question, WeightFigure

from .base import _Base


class ImageModelTest(_Base):

    def test_image_model_save_and_extract(self):
        fname = 'test.jpg'
        img = Image(filename=fname)
        img.save()

        data = Image.query().first()

        self.assertEqual(data.id, img.id)
        self.assertEqual(data.filename, fname)

    def test_update_image_state(self):
        fname = 'test.jpg'
        img = Image(filename=fname)
        img.save()

        # before update, no data is stored
        self.assertIsNotNone(img.original)

        img.update()

        target = f'{img.id:05d}.jpg'

        self.assertNotEqual(img.filename, fname)
        self.assertEqual(img.filename, target)
        # save original filename
        self.assertEqual(img.original, fname)

    def test_upload_invalid_file(self):
        with self.assertRaises(ValueError):
            img = Image(filename='invalid.txt')
            img.save()


class QuestionModelTest(_Base):

    def test_question_model_save_and_extract(self):
        question = 'is this question'
        q = Question(question=question)
        q.save()

        data = Question.get(q.id)
        self.assertEqual(data.question, question)

    def test_update_model_state(self):
        question = 'is this question'
        q = Question(question=question)
        q.save()

        data = Question.get(q.id)
        self.assertFalse(data.updated)
        data.update()
        self.assertTrue(data.updated)


class WeightFigureTest(_Base):

    def test_weight_model_automatically_set_filename(self):
        w = WeightFigure()
        w.save()

        self.assertEqual(w.filename, f'{w.id:05d}.jpg')


if __name__ == '__main__':
    unittest.main()
