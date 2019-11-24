#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime


from sqlalchemy import Column, String, Integer, DateTime

from main.orm.db import Base
from main.mixins.models import BaseMixin


class Note(BaseMixin, Base):
    """All update regarding this app to display on web app."""

    __tablename__ = 'note'

    content = Column(String(256), nullable=False)
    update = Column(DateTime, default=datetime.utcnow())


class Citation(BaseMixin, Base):
    """Refereces to be used for this app to display on web app.
    
    This will be:
        {author}, {name}, {year} [{link}]
    """

    __tablename__ = 'citation'

    author = Column(String(32), nullable=False)
    name = Column(String(128), nullable=False)
    year = Column(Integer, nullable=True)
    link = Column(String(129), nullable=True)

    def __init__(self, author, name, year=None, link=None):
        if year is not None and year < 1990:
            raise ValueError('year must be more than 1990')
        # TODO: add url validator

        super(Citation, self).__init__(author=author,
                                       name=name,
                                       year=year,
                                       link=link)

