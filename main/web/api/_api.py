#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio

from flask import request, jsonify, session

from . import api
from main.orm.models.ml import MLModel
from main.orm.db import provide_session
from main.models.client import run_model


@api.route('/model_list')
@provide_session
def model_list(session=None):
    """Return model list availabel."""
    models = MLModel.query().all()
    return jsonify([model.to_dict() for model in models])


@api.route('/question_type', methods=['POST'])
@provide_session
def predict_question_type(session=None):
    """Predict question type by given question."""
    question = request.values.get('question')
    # empty path is trigger to execute only question type prediction
    pred = asyncio.run(run_model('', question))
    return jsonify({'question': question,
                    'answer': pred})
