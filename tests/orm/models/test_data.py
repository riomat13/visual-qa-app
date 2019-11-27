#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from main.orm.models.data import Image, Question

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

        data = Image.query().filter_by(filename=fname).first()
        data.update()

        target = f'{id_:05d}.jpg'

        self.assertNotEqual(data.filename, fname)
        self.assertEqual(data.filename, target)


class QuestionModelTest(_Base):

    def test_question_model_save_and_extract(self):
        question = 'is this question'
        Question(question=question).save()

        data = Question.query().filter_by(question=question).first()

        self.assertEqual(data.question, question)


if __name__ == '__main__':
    unittest.main()
