#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import asyncio

from flask import abort, jsonify, request, session

from . import api
from main.orm.models.ml import MLModel, RequestLog, PredictionScore, QuestionType
from main.models.client import run_model
from main.web.auth import verify_user

log = logging.getLogger(__name__)


@api.route('/models/all')
def model_list():
    """Return model list availabel."""
    authorization = request.authorization
    if authorization is None or not verify_user(**authorization):
        abort(403)

    models = MLModel.query().all()
    response = jsonify([model.to_dict() for model in models])
    response.status_code = 200
    return response


@api.route('/register/model', methods=['POST'])
def register_model():
    authorization = request.authorization
    if authorization is None or not verify_user(**authorization):
        abort(403)

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
    authorization = request.authorization
    if authorization is None or not verify_user(**authorization):
        abort(403)

    model = MLModel.get(model_id)
    if model is None:
        data = {}
    else:
        data = model.to_dict()
    response = jsonify(data)
    response.status_code = 200
    return response


@api.route('/predict/question_type', methods=['POST'])
def predict_question_type():
    """Predict question type by given question."""
    question = request.values.get('question')
    # empty path is trigger to execute only question type prediction
    pred = asyncio.run(run_model('', question))
    if pred == '<e>':
        response = jsonify({'question': question, 'error': 'error occured'})
    else:
        response = jsonify({'question': question, 'answer': pred})
    response.status_code = 200
    return response


@api.route('/logs/requests', methods=['GET', 'POST'])
def extract_requests_logs():
    authorization = request.authorization
    if authorization is None or not verify_user(**authorization):
        abort(403)

    logs = RequestLog.query()

    if request.method == 'POST':
        q_type = request.values.get('question_type')
        logs = logs.filter(RequestLog.question_type.has(type=q_type))

    response = jsonify([log.to_dict() for log in logs.all()])
    response.status_code = 200
    return response


@api.route('/logs/predictions', methods=['GET', 'POST'])
def extract_prediction_logs():
    authorization = request.authorization
    if authorization is None or not verify_user(**authorization):
        abort(403)

    scores = PredictionScore.query()

    if request.method == 'POST':
        q_type = request.values.get('question_type')
        scores = scores.outerjoin(PredictionScore.log).filter(
            RequestLog.question_type.has(type=q_type)
        )
    response = jsonify([
        {
            'rate': log.rate,
            'prediction': log.prediction,
            'probability': log.probability,
            'answer': log.answer,
            'predicted_time': log.predicted_time,
            'log_id': log.log_id
        } for log in scores.all()])
    response.status_code = 200
    return response


@api.app_errorhandler(400)
def bad_request(e):
    log.error(e)
    response = jsonify(error=str(e))
    response.status_code = 400
    return response


@api.app_errorhandler(403)
def forbidden(e):
    log.error(e)
    response = jsonify(error='403 Permission Denied')
    response.status_code = 403
    return response


@api.route('/<path:invalid_url>')
def page_not_found(invalid_url):
    log.error(f'404 Not Found: /api/{invalid_url}')
    response = jsonify(error='404 Page Not Found')
    response.status_code = 404
    return response


@api.app_errorhandler(405)
def method_not_allowed(e):
    log.error(e)
    response = jsonify(error='405 Method Not Allowed')
    response.status_code = 405
    return response
