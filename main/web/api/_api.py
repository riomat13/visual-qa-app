#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import asyncio

from flask import request, jsonify, session

from . import api
from main.orm.models.ml import MLModel, RequestLog
from main.orm.db import provide_session
from main.models.client import run_model

log = logging.getLogger(__name__)


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


# TODO: add POST to filter by question type
@api.route('/question_type/logs')
@provide_session
def question_type_logs(session=None):
    logs = RequestLog.query(session=session)
    return jsonify([log.to_dict() for log in logs])


@api.app_errorhandler(400)
def bad_request(e):
    log.error(e)
    response = jsonify(error=str(e))
    response.status_code = 400
    return response


@api.app_errorhandler(404)
def page_not_found(e):
    log.error(e)
    response = jsonify(error='404 Page Not Found')
    response.status_code = 404
    return response
