#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Blueprint

base = Blueprint('base', __name__)

from . import views
