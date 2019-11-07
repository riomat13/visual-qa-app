#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Blueprint

api = Blueprint('api', __name__)

from . import _api
