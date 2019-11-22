#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path

import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

ROOT_DIR = Path(__file__).parents[2]


class Config(object):
    """Base config for web app."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get(
        'SECRET_KEY',
        b'\x93P\x8d\xaa8d\xce\xa1J\xca\x1d\xea\x88r\xfbH~\xfd\xb81f\xb3\xc3$'
    )

    # prediction server
    MODEL_SERVER = {
        'host': 'localhost',
        'port': 12345,
    }

    # model information
    MODELS = {
        # question type classification
        'QTYPE': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'question_types'),
            'seq_length': 15,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 64,
        },
        # yes/no
        'Y/N': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'y_n'),
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 256,
        },
        # what
        'WHAT': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'what'),
            'ans_length': 7,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 256,
        },
        # text_tokenizer config file path
        'TOKENIZER': {
            'path': os.path.join(ROOT_DIR, 'data', 'tokenizer_config.json'),
        }
    }

    # upload directory
    UPLOAD_DIR = os.path.join(ROOT_DIR, 'main/web/static/media/uploaded')

    # Database settings
    DATABASE_HOST = os.environ.get('DATABASE_HOST', 'localhost')
    DATABASE_PORT = os.environ.get('DATABASE_PORT', 3306)
    # mysqlclient
    DATABASE_URI = 'mysql+mysqldb://appuser:password@{}:{}/app' \
        .format(DATABASE_HOST, DATABASE_PORT)
    # PyMySql
    # DATABASE_URI = 'mysql+pymysql://appuser:password@{}:{}/app' \
    #     .format(DATABASE_HOST, DATABASE_PORT)

    @staticmethod
    def init_app(app):
        pass
