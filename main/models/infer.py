#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os.path
from functools import partial

import numpy as np

from main.settings import Config
from main.utils.loader import fetch_question_types, load_image
from main.utils.preprocess import text_processor
from ._base import BaseModel
from ._models import (
    QuestionTypeClassification,
    ClassificationModel,
    QuestionAnswerModel,
)

log = logging.getLogger(__name__)


processor = text_processor(num_words=20000, from_config=True)
# TODO: tmp
classes = fetch_question_types()
id2q = [q for q in classes]
num_classes = len(classes)

# tokens storing IDs for specific tokens
_tokens = {
    'bos': processor.word_index['<bos>'],
    'eos': processor.word_index['<eos>'],
    'unk': processor.word_index['<unk>'],
}


class PredictionModel(BaseModel):
    __instance = None

    _model_class = {
        'Y/N': {2, 6, 8, 10, 11, 13, 14, 15, 16, 17,
                23, 27, 29, 33, 35, 39, 40, 41, 42,
                43, 44, 45, 46, 47, 50, 52, 56, 57,
                58, 59, 62, 64, 66, 68, 76, 79},
        'WHAT': {1, 4, 5, 7, 12, 19, 20, 21, 22, 24,
                 25, 26, 28, 37, 38, 48, 51, 53, 54,
                 67, 75, 76, 77},
        # TODO:
        'WHY': set(),
        'WHICH': set(),
        'WHO': set(),
        'WHERE': set(),
        'HOW': set(),
        'COUNT': set(),
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

    def predict(self, sentence):
        # processor handles list of sentences
        sequence = self._processor(sentence)

        # if no word is detected including oov,
        # skip prediction
        if len(sequence.shape) == 1 or \
                len([s for s in sequence[0] if s > 1]) == 0:
            # last id in category(none of the above)
            log.warning('not found any word from vocabulary')
            return 'Can not understand the question'

        pred = predict_question_type(sequence)

        if pred in self._model_class['Y/N']:
            pred = predict_yes_or_no(sequence)
        elif pred in self._model_class['WHAT']:
            pred = predict_what(sequence)
        return pred

    @classmethod
    def get_model(cls):
        if PredictionModel.__instance is None:
            PredictionModel()
        return PredictionModel.__instance

    def _build_model(self):
        self._models = {
            'QTYPE': _get_q_type_model(),
            'Y/N': _get_y_n_model(),
            'WHAT': _get_what_model(),
        }


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
                num_classes=num_classes,
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
                seq_length=15,
                ans_length=7,
                vocab_size=cfg.get('vocab_size'),
                embedding_dim=cfg.get('embedding_dim'),
                model_type='WHAT')
            _set_weights_by_config('WHAT', model)

        return model

    return build_model()


def awake_models():
    """Set up models initially."""
    # ignore tqdm if not installed
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = lambda x: x

    # TODO: add models
    for func in tqdm([_get_q_type_model, _get_y_n_model]):
        log.info(f'Loding model: {func.__name__}')
        func()


def convert_output_to_sentence(sequence):
    """Converting network sequence to readable output.
    Args:
        sequence: 2-d numpy.ndarray(dtype=np.int32)
            sequence represents the predict outputs
    Returns:
        str: converted sentence to human-readable string
    """
    dim = len(sequence.shape)
    if dim != 2:
        raise ValueError('Invalid input shape. Expected 2-dim'
                         f'but {dim}-dim is given')

    sequence = np.argmax(sequence, axis=-1)
    result = []
    for idx in sequence:
        # stop if reaches end of sentence
        if idx == _tokens['eos']:
            break
        # skip if token is unknown
        elif idx == _tokens['unk']:
            continue

        result.append(processor.index_word[idx])

    return ' '.join(result)


# TODO: temporary prediction function
def predict_question_type(sequence):
    model = _get_q_type_model()

    pred = model.predict(sequence)
    pred = np.argmax(pred, axis=1)
    pred = int(pred[0])
    return id2q[pred]


def predict_yes_or_no(sequence, img_path):
    model = _get_y_n_model()
    img = load_image(img_path)

    # predicted result shape is (1, 3)
    pred = model(sequence, img)
    pred = np.argmax(pred, axis=-1)
    return pred
