#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
from functools import partial

import numpy as np

from main.settings import Config
from main.utils.loader import fetch_question_types
from main.utils.preprocess import text_processor
from ._base import BaseModel
from ._models import QuestionTypeClassification

log = logging.getLogger(__name__)


processor = text_processor(from_config=True)
# TODO: tmp
classes = fetch_question_types()
id2q = [q for q in classes]
num_classes = len(classes)


class PredictionModel(BaseModel):
    __instance = None

    def __init__(self):
        if PredictionModel.__instance is None:
            self.build_model()
            PredictionModel.__instance = self
        else:
            raise RuntimeError(f'This object can not be instantiated. Use {self.__class__.__name__}.get_model() instead')

    @property
    def type(self):
        return self.__type

    def predict(self, x):
        # TODO: build pipeline
        pred = predict_question_type(x)
        return pred

    def get_model(self):
        if PredictionModel.__instance is None:
            PredictionModel()
        return PredictionModel.__instance

    def _build_model(self):
        # count class types from file
        num_classes = num_classes
        self._models = {
            'QTYPE': _get_q_type_model(),
            #'Y/N': _get_y_n_model(),
        }


def _set_weights_by_config(type, model):
    path = Config.MODELS.get(type)
    if not path or not os.path.isdir(path):
        raise FileNotFoundError('Could not find weights. Path is not set or Model is not implemented yet')

    model.load_weights(os.path.join(path, 'weights'))


def _get_q_type_model():
    """Categorize the question type to make subproblems."""
    model = None

    def build_model():
        nonlocal model

        if model is None:
            # TODO: remove hard coded parameters
            model = QuestionTypeClassification(
                embedding_dim=256,
                units=64,
                vocab_size=processor.vocab_size,
                num_classes=num_classes,
            )
            _set_weights_by_config('QTYPE', model)

        return model

    return build_model()


def _get_y_n_model():
    """Predict to answer `yes` or `no`."""
    raise NotImplementedError('Not implemented model yet')

    model = None

    def build_model():
        nonlocal model

        if model is None:
            model = None
            _set_weights_by_config('Y/N', model)

        return model

    return build_model()


# TODO: temporary prediction function
def predict_question_type(sentence):
    model = _get_q_type_model()

    processor = text_processor(from_config=True)

    # processor handles list of sentences
    sequence = processor(sentence)
    # TODO: check edge cases such as all padded
    if sequence.shape[1] > 0:
        log.info('running prediction')
        pred = model.predict(sequence)
        pred = np.argmax(pred, axis=1)
        pred = int(pred[0])
    else:
        # last id in category(none of the above)
        log.warning('not found any word from vocabulary')
        pred = 80
    return id2q[pred]
