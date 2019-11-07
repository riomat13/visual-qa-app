#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime

from main.orm.db import Base
from main.mixins.models import BaseMixin, ModelLogMixin


class MLModel(BaseMixin, Base):
    __tablename__ = 'ml_model'

    id = Column(Integer, primary_key=True)
    # model type: classification, encoder/decoder etc.
    type = Column(String(32), unique=True, nullable=False)
    # type of problem to solve: question_type etc.
    category = Column(String(32), nullable=False)
    # module path
    module = Column(String(64), nullable=False)
    # path to model data (e.g. weights)
    path = Column(String(128), nullable=True)
    # created/updated data
    created = Column(DateTime, default=datetime.utcnow())
    updated = Column(DateTime, default=datetime.utcnow())


class ModelLog(ModelLogMixin, BaseMixin, Base):
    # logs for running model
    __tablename__ = 'model_log'


class ModelRequestLog(ModelLogMixin, BaseMixin, Base):
    # logs when request to run model
    __tablename__ = 'model_request_log'
