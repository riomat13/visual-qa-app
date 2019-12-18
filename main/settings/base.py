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

    # celery
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')
    CELERY_IMPORTS = 'main.tasks'

    # Mail settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = os.environ.get('MAIL_PORT', 567)
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')

    # file server
    FILE_SERVER_URL = os.environ.get('FILE_SERVER_URL')

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
            'seq_length': 10,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 64,
        },
        # yes/no
        'Y/N': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'y_n'),
            'vocab_size': 15000,
            'embedding_dim': 256,
            'units': 256,
        },
        # TODO: adjust hyper-parameter settings for each model
        'WHAT': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'what'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 512,
            'units': 512,
        },
        'WHERE': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'where'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 512,
        },
        'WHICH': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'which'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 512,
        },
        'WHO': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'who'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 512,
        },
        'WHY': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'why'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 512,
        },
        'COUNT': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'count'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 512,
        },
        # non-categorized group
        'NONE': {
            'path': os.path.join(ROOT_DIR, 'checkpoints', 'none'),
            'seq_length': 10,
            'ans_length': 5,
            'vocab_size': 20000,
            'embedding_dim': 256,
            'units': 512,
        },
        # text_tokenizer config file path
        'TOKENIZER': {
            'path': os.path.join(ROOT_DIR, 'data', 'tokenizer_config.json'),
        }
    }

    # upload directory
    STATIC_DIR = os.path.join(ROOT_DIR, 'main/web/static')
    UPLOAD_DIR = 'media/uploaded'
    FIG_DIR = 'media/figs'

    # target directory to send data by tasks
    SEND_DATA_DIR = 'data/tmp'
    REMOTE_USERNAME = os.environ.get('REMOTE_USERNAME')
    REMOTE_PASSWORD = os.environ.get('REMOTE_PASSWORD')
    SSH_HOSTKEYS = os.environ.get('SSH_HOSTKEYS')
    REMOTE_SERVER = os.environ.get('REMOTE_SERVER')
    REMOTE_DIR = 'data/tmp/target'

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
