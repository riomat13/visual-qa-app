#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
from functools import partial

import numpy as np
import tensorflow as tf

from main.settings import Config
from main.utils.loader import fetch_question_types, load_image
from main.utils.preprocess import text_processor
from .common import get_mobilenet_encoder
from ._base import BaseModel
from ._models import (
    QuestionTypeClassification,
    ClassificationModel,
    QuestionAnswerModel,
)

log = logging.getLogger(__name__)


processor = text_processor(num_words=20000, from_config=True)
img_encoder = get_mobilenet_encoder()
# TODO: tmp
classes = fetch_question_types()
id2q = [q for q in classes]

# tokens storing IDs for specific tokens
_tokens = {
    'bos': processor.word_index['<bos>'],
    'eos': processor.word_index['<eos>'],
    'unk': processor.word_index['<unk>'],
    'pad': processor.word_index['<pad>'],
}


class PredictionModel(BaseModel):
    __instance = None

    _model_class = {
        0: 'Y/N',
        1: 'WHAT',
        2: 'WHY',
        3: 'WHICH',
        4: 'WHO',
        5: 'WHERE',
        6: 'HOW',
        7: 'COUNT',
        8: 'NONE',
    }

    def __init__(self):
        if PredictionModel.__instance is None:
            self._build_model()
            self._processor = text_processor(
                num_words=20000,
                maxlen=Config.MODELS.get('QTYPE').get('seq_length'),
                from_config=True
            )

            PredictionModel.__instance = self
        else:
            raise RuntimeError(f'This object can not be instantiated. Use {self.__class__.__name__}.get_model() instead')

    @property
    def type(self):
        return self.__type

    def predict(self, sentence, img_path):
        pred = ''

        # processor handles list of sentences
        sequence = self._processor(sentence)

        # if no word is detected including oov,
        # skip prediction
        if len(sequence.shape) == 1 or \
                len([s for s in sequence[0] if s > 1]) == 0:
            # last id in category(none of the above)
            log.warning('not found any word from vocabulary')
            return 'Can not understand the question'

        pred_id = predict_question_type(sequence)
        w = None

        if pred_id == 0:
            pred, w = predict_yes_or_no(sequence, img_path)
            if pred[0] == 0:
                pred = 'yes'
            elif pred[0] == 1:
                pred = 'no'
            else:
                pred = 'Sorry, could not understand the question'
        elif pred_id == 1:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        elif pred_id == 2:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        elif pred_id == 3:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        elif pred_id == 4:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        elif pred_id == 5:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        elif pred_id == 6:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        elif pred_id == 8:
            pred, w = predict_what(sequence, img_path)
            pred = convert_output_to_sentence(pred)
        # TODO: add couting model
        if not pred:
            pred = 'Sorry, could not understand the question'
        return pred, w

    @classmethod
    def get_model(cls):
        if PredictionModel.__instance is None:
            PredictionModel()
        return PredictionModel.__instance

    def _build_model(self):
        # TODO: replace models with type specifig ones
        models = (
            _get_q_type_model,
            _get_y_n_model,
            _get_what_model,
            #_get_where_model,
            #_get_which_model,
            #_get_who_model,
            #_get_why_model,
            #_get_how_model,
        )

        # load all data
        awake_models(models)


def _set_weights_by_config(type, model):
    model_type = Config.MODELS.get(type)
    if model_type is None:
        raise ValueError('Model type is not registered')
    path = model_type.get('path')
    if not path or not os.path.isdir(path):
        raise FileNotFoundError('Could not find weights. Path is not set or Model is not implemented yet')

    model.load_weights(os.path.join(path, 'weights'))
    log.info('Loaded weights: {}'.format(type))


def _get_q_type_model():
    """Categorize the question type to make subproblems."""
    model = None

    def build_model():
        nonlocal model

        if model is None:
            cfg = Config.MODELS['QTYPE']
            # TODO: remove hard coded parameters
            model = QuestionTypeClassification(
                embedding_dim=cfg.get('embedding_dim'),
                units=cfg.get('units'),
                vocab_size=cfg.get('vocab_size'),
                num_classes=9,
            )
            _set_weights_by_config('QTYPE', model)

        return model

    return build_model()


def _get_y_n_model():
    """Predict to answer `yes` or `no`."""
    model = None

    def build_model():
        nonlocal model

        if model is None:
            cfg = Config.MODELS['Y/N']

            model = ClassificationModel(
                units=cfg.get('units'),
                vocab_size=cfg.get('vocab_size'),
                embedding_dim=cfg.get('embedding_dim'),
                num_classes=3,
            )
            _set_weights_by_config('Y/N', model)

        return model

    return build_model()


def _get_what_model():
    """Predict to answer to `what` questions."""
    model = None

    def build_model():
        nonlocal model

        if model is None:
            cfg = Config.MODELS['WHAT']
            model = QuestionAnswerModel(
                units=cfg.get('units'),
                ans_length=7,
                vocab_size=cfg.get('vocab_size'),
                embedding_dim=cfg.get('embedding_dim'),
                model_type='WHAT'
            )

        return model

    return build_model()


def awake_models(models):
    """Set up models initially."""
    for func in models:
        log.info(f'Loding model: {func.__name__}')
        func()


def convert_output_to_sentence(sequence):
    """Converting network sequence to readable output.
    Args:
        sequence: 1-d or 2-d numpy.ndarray(dtype=np.int32)
            sequence represents the predict outputs
            if 2-d, apply argmax to convert 1-d array
    Returns:
        str: converted sentence to human-readable string
    """
    dim = len(sequence.shape)
    if dim == 2:
        sequence = np.argmax(sequence, axis=-1)
    elif dim > 2:
        raise ValueError('Invalid input shape. Expected 2-dim '
                         f'but {dim}-dim is given')

    result = []
    for idx in sequence:
        # stop if reaches end of sentence
        if idx == _tokens['eos']:
            break
        # skip if token is unknown
        elif idx == _tokens['unk'] or idx == _tokens['pad']:
            continue

        result.append(processor.index_word[idx])

    return ' '.join(result)


# TODO: temporary prediction function
def predict_question_type(sequence):
    model = _get_q_type_model()

    pred = model.predict(sequence)
    pred = np.argmax(pred, axis=1)
    pred = int(pred[0])
    return pred


def predict_yes_or_no(sequence, img_path):
    model = _get_y_n_model()
    img = load_image(img_path)
    img = tf.reshape(img, (1, 224, 224, 3))
    img = img_encoder(img)

    # predicted result shape is (1, 3)
    pred, weights = model(sequence, img)
    pred = np.argmax(pred, axis=1)
    return pred, weights


def predict_what(sequence, img_path):
    model = _get_what_model()
    img = load_image(img_path)
    img = tf.reshape(img, (1, 224, 224, 3))
    img = img_encoder(img)

    x = np.array([processor.word_index['<bos>']])
    hidden = np.zeros((1, 256))

    pred, weights = model(x, sequence, img, hidden)
    pred = np.argmax(pred, axis=-1)
    return pred, weights
