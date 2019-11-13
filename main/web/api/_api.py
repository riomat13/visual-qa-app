#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import jsonify

from . import api
from main.orm.models.ml import MLModel
from main.orm.db import provide_session


@api.route('/model_list')
@provide_session
def model_list(session=None):
    """Return model list availabel."""
    models = MLModel.query().all()
    return jsonify([model.to_dict() for model in models])
