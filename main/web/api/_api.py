#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import asyncio

from flask import request, jsonify, session

from . import api
from main.orm.models.ml import MLModel, RequestLog
from main.models.client import run_model

log = logging.getLogger(__name__)


@api.route('/models/all')
def model_list():
    """Return model list availabel."""
    models = MLModel.query().all()
    response = jsonify([model.to_dict() for model in models])
    response.status_code = 200
    return response


@api.route('/register/model', methods=['POST'])
def register_model():
    name = request.values.get('name')
    type_ = request.values.get('type')
    cat = request.values.get('category')
    module = request.values.get('module')
    obj = request.values.get('object')
    path = request.values.get('path')
    metrics = request.values.get('metrics')
    score = request.values.get('score')
    if score is not None:
        score = float(score)

    try:
        model = MLModel(name=name,
                        type=type_,
                        category=cat,
                        module=module,
                        object=obj,
                        path=path,
                        metrics=metrics,
                        score=score)
        model.save()
        response = jsonify({
            'success': 'created model successfully',
            'model': MLModel.query().filter_by(name=name).first().to_dict()
        })
        response.status_code = 200
    except Exception as e:
        response = jsonify({
            'error': str(e)
        })
        response.status_code = 400
    return response


@api.route('/model/<int:model_id>')
def get_model_info(model_id):
    model = MLModel.query().filter_by(id=model_id).first()
    if model is None:
        data = {}
    else:
        data = model.to_dict()
    response = jsonify(data)
    response.status_code = 200
    return response


@api.route('/question_type', methods=['POST'])
def predict_question_type():
    """Predict question type by given question."""
    question = request.values.get('question')
    # empty path is trigger to execute only question type prediction
    pred = asyncio.run(run_model('', question))

    response = jsonify({'question': question, 'answer': pred})
    response.status_code = 200
    return response


# TODO: add POST to filter by question type
@api.route('/question_type/logs')
def question_type_logs():
    logs = RequestLog.query().all()
    response = jsonify([log.to_dict() for log in logs])
    response.status_code = 200
    return response


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


@api.app_errorhandler(405)
def method_not_allowed(e):
    log.error(e)
    response = jsonify(error='405 Method Not Allowed')
    response.status_code = 405
    return response
