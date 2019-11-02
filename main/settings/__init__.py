#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import ROOT_DIR, Config


def set_config(config_type):
    global Config

    if config_type == 'production':
        from .production import ProductionConfig
        Config = ProductionConfig

    elif config_type == 'local':
        from .local import LocalConfig
        Config = LocalConfig

    elif config_type == 'development':
        from .dev import DevelopmentConfig
        Config = DevelopmentConfig

    elif config_type == 'test':
        from .test import TestConfig
        Config = TestConfig
