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

    # TODO: make relationship with scores as one-to-many
    #predictions = relationship('RequestLog', backref='ml_model')


class PredictionModel(BaseMixin, Base):
    """Models used for prediction."""
    __tablename__ = 'prediction_model'

    model_id = Column(Integer, ForeignKey('ml_model.id'))
    model = relationship('MLModel')


class ModelLog(ModelLogMixin, BaseMixin, Base):
    """Logs for running model."""
    __tablename__ = 'model_log'

    # TODO


class RequestLog(ModelLogMixin, BaseMixin, Base):
    """Logs when got request to predict."""
    __tablename__ = 'request_log'

    # file name to be used for prediction
    filename = Column(String(64), nullable=False)
    # question about image
    question_type = Column(String(32), nullable=False)
    question = Column(String(128), nullable=False)

    # TODO: uncomment after store model data to db
    #model_id = Column(Integer, ForeignKey('prediction_model.id'))
    #model = relationship('PredictionModel')


class PredictionScore(BaseMixin, Base):
    """Store results of predictions."""
    __tablename__ = 'prediction_score'

    # just in case if not rated
    rate = Column(Integer, nullable=True)
    # predicted answers
    prediction = Column(String(128), nullable=False)
    # likelihood probability for the answer
    # TODO: should not be nullable
    probability = Column(Float)
    # ideal answer (optional)
    answer = Column(String(128), nullable=True)

    predicted_time = Column(DateTime, default=datetime.utcnow())

    log_id = Column(Integer, ForeignKey('request_log.id'))
    log = relationship('RequestLog')

    def __init__(self, prediction, log, rate=None, **kwargs):
        """Predicted score.

        Args:
            prediction: str
                predicted answer
            log: RequestLog model
            rate(optional): int
                rate the result, 1 - 5
            question_type(optional): str
                category of question type (e.g. `what is`)
            answer(optional): str
                ideal answer
        """
        # validator
        if rate is not None:
            if not 0 < rate < 6:
                raise ValueError('Rate must be chosen from 1 to 5')
        super(PredictionScore, self).__init__(prediction=prediction,
                                              log=log,
                                              rate=rate,
                                              **kwargs)

    def update(self, question_type, answer):
        """Update information for later label."""
        self.question_type = question_type
        self.answer = answer
        self.save()
