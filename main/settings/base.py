#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]


class Config(object):
    """Base config for web app."""
    DEBUG = False
    TESTING = False
    SECRET_KET = os.environ.get('SECRET_KEY', 'temp_key')

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
