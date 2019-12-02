#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime


from sqlalchemy import Column, String, Integer, DateTime

from main.orm.db import Base
from main.mixins.models import BaseMixin


class Update(BaseMixin, Base):
    """All update regarding this app to display on web app."""

    __tablename__ = 'update'

    content = Column(String(256), nullable=False)
    update_at = Column(DateTime, default=datetime.utcnow())
    summary = Column(String(512), nullable=True)


class Citation(BaseMixin, Base):
    """Refereces to be used for this app to display on web app.
    
    This will be:
        {author}, {name}, {year} [{link}]
    """

    __tablename__ = 'citation'

    author = Column(String(32), nullable=False)
    title = Column(String(128), nullable=False)
    year = Column(Integer, nullable=True)
    url = Column(String(129), nullable=True)
    summary = Column(String(512), nullable=True)

    def __init__(self, author, title, year=None, url=None, summary=None):
        if year is not None and year < 1990:
            raise ValueError('year must be more than 1990')
        # TODO: add url validator

        super(Citation, self).__init__(author=author,
                                       title=title,
                                       year=year,
                                       url=url,
                                       summary=summary)
