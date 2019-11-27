#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
from datetime import datetime

from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from main.orm.db import Base
from main.mixins.models import BaseMixin


class Image(BaseMixin, Base):
    __tablename__ = 'image'

    filename = Column(String(64), nullable=False)
    # image is processed and stored as numpy array
    processed = Column(Boolean, nullable=False, default=False)
    original = Column(String(64), nullable=True)
    saved_at = Column(DateTime, default=datetime.utcnow())

    def __init__(self, filename):
        # add other file formats
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            raise ValueError('Invalid file. '
                             'Make sure extention is included in file name '
                             'and file is either `.jpg` or `.png`')
        # accept only filename as input
        super(Image, self).__init__(filename=filename)

    def update(self):
        """Execute once processed original data."""
        # replace name with id
        _, ext = os.path.splitext(self.filename)
        self.original = self.filename
        self.filename = f'{self.id:05d}{ext}'
        self.processed = True
        self.save()


class Question(BaseMixin, Base):
    __tablename__ = 'question'

    question = Column(String(256), nullable=False)

    type_id = Column(Integer, ForeignKey('question_type.id'))
    type = relationship('QuestionType')

    # this is a flag to check if set type is correct
    # initially type is set as predicted one
    updated = Column(Boolean, default=False)

    def update(self):
        """Set as True if type is checked."""
        self.updated = True
        self.save()
