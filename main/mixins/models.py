#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.inspection import inspect

from main.orm.db import provide_session


def _serialize(obj):
    # TODO: check if all attrs and future attrs work with this function
    return {key: getattr(obj, key) for key in inspect(obj).attrs.keys()}


class BaseMixin(object):

    @classmethod
    @provide_session
    def query(cls, session=None):
        return session.query(cls)

    @classmethod
    def get(cls, id):
        """Pass id and return corresponding model instance."""
        return cls.query.get(id)

    @provide_session
    def save(self, session=None):
        """Save current state to session."""
        session.add(self)

    @provide_session
    def delete(self, session=None):
        """Delete model."""
        session.delete(self)

    def to_dict(self):
        return _serialize(self)


class ModelLogMixin(object):

    id = Column(Integer, primary_key=True)
    logged_time = Column(DateTime, default=datetime.utcnow())
    # Log Type: e.g. Info, Warning, Error
    log_type = Column(String(32), nullable=False)
    # Log Detail
    log_text = Column(Text, nullable=False)

    @declared_attr
    def model_id(cls):
        return Column('model_id', ForeignKey('ml_model.id'))

    @declared_attr
    def model(cls):
        return relationship('MLModel')
