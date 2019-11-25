#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import Config

class ProductionConfig(Config):
    DATABASE_HOST = os.environ.get('DATABASE_HOST')
    DATABASE_PORT = os.environ.get('DATABASE_PORT', 5432)
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')

    DATABASE_URI = f'postgres://postgres:{POSTGRES_PASSWORD}@' \
                   f'{DATABASE_HOST}:{DATABASE_PORT}/app_db'
