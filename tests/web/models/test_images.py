#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from main.settings import set_config
set_config('test')

from main.web.models.images import Image

from .base import _Base


class ImageModelTest(_Base):

    def test_image_model_save_and_extract(self):
        img = Image(filename='123')
        img.save(session=self.session)

        data = Image.query(session=self.session).first()

        self.assertEqual(data.id, img.id)
        self.assertEqual(data.filename, img.filename)


if __name__ == '__main__':
    unittest.main()
