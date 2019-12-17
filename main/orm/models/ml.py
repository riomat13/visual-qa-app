#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Models to record created models
#   MLModel: store all models
#   PredictionModel: models actually used in predictions
#   ModelLog: store logs running models in MLModel
#   ModelRequestLog: store logs when requested from api
#   PredictionScore: store results of predictions for later update

import logging
from datetime import datetime
from importlib.util import find_spec

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey
)
from sqlalchemy.orm import relationship

from main.orm.db import Base
from main.orm.types import ChoiceType
from main.mixins.models import BaseMixin, ModelLogMixin
from main.orm.models.data import Question, Image

log = logging.getLogger(__name__)


class MLModel(BaseMixin, Base):
    """Created models."""
    __tablename__ = 'ml_model'

    name = Column(String(64), unique=True, nullable=False)
    # model type: classification, encoder/decoder etc.
    type = Column(ChoiceType(
        {
            'cls': 'classification',
            'seq': 'seq2seq',
            'enc': 'encoder',
            'dec': 'decoder'
        }), nullable=False)
    # type of problem to solve: question_type etc.
    category = Column(String(32), nullable=False)
    # module path (e.g. main.models)
    module = Column(String(64), nullable=False)
    object = Column(String(32), nullable=False)
    # path to model data (e.g. weights)
    path = Column(String(128), nullable=True)
    # created/updated data
    created = Column(DateTime, default=datetime.utcnow())
    updated = Column(DateTime, default=datetime.utcnow())

    metrics = Column(String(32), nullable=True)
    score = Column(Float, nullable=True)

    prediction_model = relationship('PredictionModel', back_populates='model')

    def __init__(self, name, type, category, module, object,
                 path=None, metrics=None, score=None):
        if find_spec(module) is None:
            log.error(f'Invalid module name: {module}')
            raise ModuleNotFoundError(f'Could not find module: {module}')

        if type not in ('cls', 'seq', 'enc', 'dec'):
            raise ValueError('Invalid type name')

        super(MLModel, self).__init__(name=name,
                                      type=type,
                                      category=category,
                                      module=module,
                                      object=object,
                                      path=path,
                                      metrics=metrics,
                                      score=score)

    def update_score(self, score, metrics=None):
        """Update score data.

        Args:
            score: float
            metrics(optional): str
        """
        self.metrics = metrics
        self.score = score
        self.save()


class PredictionModel(BaseMixin, Base):
    """Models used for prediction."""
    __tablename__ = 'prediction_model'

    model_id = Column(Integer, ForeignKey('ml_model.id'))
    model = relationship('MLModel', back_populates='prediction_model')

    log = relationship('RequestLog', back_populates='model')


class ModelLog(ModelLogMixin, BaseMixin, Base):
    """Logs for running model."""
    __tablename__ = 'model_log'

    # TODO


class RequestLog(ModelLogMixin, BaseMixin, Base):
    """Logs when got request to predict."""
    __tablename__ = 'request_log'

    question_type_id = Column(Integer, ForeignKey('question_type.id'))
    question_type = relationship('QuestionType')

    question_id = Column(Integer, ForeignKey('question.id'))
    image_id = Column(Integer, ForeignKey('image.id'))

    fig_id = Column(Integer, ForeignKey('weight_figure.id'))
    fig = relationship('WeightFigure', back_populates='log')

    # predicted score and model
    score_id = Column(Integer, ForeignKey('prediction_score.id'))
    score = relationship('PredictionScore', back_populates='log')
    model_id = Column(Integer, ForeignKey('prediction_model.id'))
    model = relationship('PredictionModel', back_populates='log')

    def to_dict(self):
        return {
            'id': self.id,
            'log_type': self.log_type,
            'log_text': self.log_text,
            'question_type_id': self.question_type_id,
            'image_id': self.image_id,
            'question_id': self.question_id,
            'fig_id': self.fig_id,
            'model_id': self.model_id,
        }


class PredictionScore(BaseMixin, Base):
    """Store results of predictions."""
    __tablename__ = 'prediction_score'

    # just in case if not rated
    rate = Column(Integer, nullable=True)
    # predicted answers
    prediction = Column(String(128), nullable=False)
    # likelihood probability for the answer
    # TODO: should not be nullable
    probability = Column(Float, nullable=True)
    # ideal answer (optional)
    answer = Column(String(128), nullable=True)

    predicted_time = Column(DateTime, default=datetime.utcnow())

    log = relationship('RequestLog', back_populates='score', uselist=False)

    def __init__(self, prediction, rate=None, **kwargs):
        """Predicted score.

        Args:
            prediction: str
                predicted answer
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
                                              rate=rate,
                                              **kwargs)

    def update(self, *, question_type=None, answer=None):
        """Update information for later label."""
        if question_type is None and answer is None:
            raise ValueError('provide question_type and/or answer')
        self.question_type = question_type
        self.answer = answer
        self.save()


class QuestionType(BaseMixin, Base):
    __tablename__ = 'question_type'

    type = Column(String(64), unique=True, nullable=False)
    question = relationship('Question', back_populates='type')

    @classmethod
    def register(cls, type):
        """Shortcut to register new question type."""
        cls(type=type).save()
