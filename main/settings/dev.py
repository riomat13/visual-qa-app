#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from .base import Config


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = True
    ENV = 'development'

    DATABASE_HOST = os.environ.get('DATABASE_HOST', '127.0.0.1')
    DATABASE_PORT = os.environ.get('DATABASE_PORT', 5432)
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')

    DATABASE_URI = f'postgres://postgres:{POSTGRES_PASSWORD}@' \
                   f'{DATABASE_HOST}:{DATABASE_PORT}/app_db_dev'
