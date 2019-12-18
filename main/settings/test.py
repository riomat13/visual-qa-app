#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import getpass

from .base import ROOT_DIR, Config


class TestConfig(Config):
    DEBUG = True
    TESTING = True
    ENV = 'testing'
    WTF_CSRF_ENABLED = False

    # Database settings
    DATABASE_URI = 'sqlite:////tmp/tmp-app-test.db'

    TEST_UPLOAD_DIR = os.path.join(ROOT_DIR, 'tests/data/uploaded')
    UPLOAD_DIR = 'media/tmp/uploaded'
    FIG_DIR = 'media/tmp/figs'
    SEND_DATA_DIR = 'tests/data/tmp'
    REMOTE_USERNAME = getpass.getuser()
    REMOTE_PASSWORD = ''
    SSH_HOSTKEYS = 'test_ssh_hostkey'
    REMOTE_SERVER = 'localhost'
    REMOTE_DIR = 'tests/data/tmp/target'
