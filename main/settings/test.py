#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import Config


class TestConfig(Config):
    DEBUG = True
    TESTING = True
    ENV = 'testing'

    # Database settings
    DATABASE_URI = 'sqlite:////tmp/tmp-app-test.db'