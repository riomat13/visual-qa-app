#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import uuid

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
