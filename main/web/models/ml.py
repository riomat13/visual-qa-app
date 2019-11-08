#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Models to record created models
#   MLModel: store all models
#   PredictionModel: models actually used in predictions
#   ModelLog: store logs running models in MLModel
#   ModelRequestLog: store logs when requested from api
#   PredictionScore: store results of predictions for later update

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey
)
from sqlalchemy.orm import relationship

from main.orm.db import Base
from main.mixins.models import BaseMixin, ModelLogMixin


class MLModel(BaseMixin, Base):
    """Created models."""
    __tablename__ = 'ml_model'

    name = Column(String(64), nullable=False)
    # model type: classification, encoder/decoder etc.
    type = Column(String(32), unique=True, nullable=False)
    # type of problem to solve: question_type etc.
    category = Column(String(32), nullable=False)
    # class path (e.g. main.models.some.Model)
    module = Column(String(64), nullable=False)
    # path to model data (e.g. weights)
    path = Column(String(128), nullable=True)
    # created/updated data
    created = Column(DateTime, default=datetime.utcnow())
    updated = Column(DateTime, default=datetime.utcnow())

    metrics = Column(String(32), nullable=True)
    score = Column(Float, nullable=True)


class PredictionModel(BaseMixin, Base):
    """Models used for prediction."""
    __tablename__ = 'prediction_model'

    model_id = Column(Integer, ForeignKey('ml_model.id'))
    model = relationship('MLModel')


class ModelLog(ModelLogMixin, BaseMixin, Base):
    """Logs for running model."""
    __tablename__ = 'model_log'

    # TODO


class ModelRequestLog(ModelLogMixin, BaseMixin, Base):
    """Logs when request to run model."""
    __tablename__ = 'model_request_log'

    # TODO


class PredictionScore(BaseMixin, Base):
    """Store results of predictions."""
    __tablename__ = 'prediction_score'

    # just in case if not rated
    rate = Column(Integer, nullable=True)
    # file name to be used for prediction
    filename = Column(String(64), nullable=False)
    # predicted answers
    prediction = Column(String(128), nullable=False)
    # ideal answer (optional)
    answer = Column(String(128), nullable=True)
    predicted_time = Column(DateTime, default=datetime.utcnow())

    model_id = Column(Integer, ForeignKey('prediction_model.id'))
    model = relationship('PredictionModel')

    def __init__(self, filename, prediction, rate=None, answer=None):
        # validator
        if rate is not None:
            if not 0 < rate < 6:
                raise ValueError('Rate must be chosen from 1 to 5')
        super(PredictionScore, self).__init__(rate=rate,
                                              filename=filename,
                                              prediction=prediction,
                                              answer=answer)
