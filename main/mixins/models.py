#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from datetime import datetime

import sqlalchemy
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.inspection import inspect

from main.orm.db import provide_session

log = logging.getLogger(__name__)


def _serialize(obj):
    # TODO: check if all attrs and future attrs work with this function
    return {key: getattr(obj, key) for key in inspect(obj).attrs.keys()}


class BaseMixin(object):

    id = Column(Integer, primary_key=True)

    @classmethod
    @provide_session
    def query(cls, *, session=None):
        return session.query(cls)

    @classmethod
    def get(cls, id):
        """Pass id and return corresponding model instance."""
        return cls.query().get(id)

    @provide_session
    def save(self, *, autoflush=True, session=None):
        """Save current state to session."""
        try:
            session.add(self)
            if autoflush:
                session.flush()
        except sqlalchemy.exc.IntegrityError as e:
            log.error(e)
            session.rollback()
            raise
        except Exception as e:
            log.error(e)
            session.rollback()
            raise

    @provide_session
    def delete(self, *, session=None):
        """Delete model."""
        session.delete(self)

    def to_dict(self):
        return _serialize(self)


class BaseLogMixin(object):

    logged_time = Column(DateTime, default=datetime.utcnow())
    # Log Type: e.g. Info, Warning, Error
    log_type = Column(String(32), nullable=False)
    # class name if error. e.g. ValueError
    log_class = Column(String(32), nullable=True)
    # Log Detail
    log_text = Column(Text, nullable=False)


class ModelLogMixin(BaseLogMixin):

    @declared_attr
    def model_id(cls):
        return Column('model_id', ForeignKey('ml_model.id'))

    @declared_attr
    def model(cls):
        return relationship('MLModel')
