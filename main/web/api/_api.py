#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import asyncio

from flask import abort, jsonify, request, session

from . import api
from main.orm.models.ml import MLModel, RequestLog, PredictionScore, QuestionType
from main.orm.models.data import Image, Question, WeightFigure
from main.orm.models.web import Update, Citation
from main.models.client import run_model
from main.web.auth import verify_user

log = logging.getLogger(__name__)


def _is_authorized():
    authorization = request.authorization
    if authorization is None or not verify_user(**authorization):
        abort(403)


@api.route('/models/all')
def model_list():
    """Return model list availabel."""
    _is_authorized()

    kwargs = {
        'task': 'read-only',
        'type': 'model',
        'done': True
    }

    models = MLModel.query().all()
    response = jsonify(data=[model.to_dict() for model in models],
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/register/model', methods=['POST'])
def register_model():
    _is_authorized()

    kwargs = {
        'task': 'register',
        'type': 'model',
        'done': True
    }

    try:
        model = MLModel(name=request.values.get('name'),
                        type=request.values.get('type'),
                        category=request.values.get('category'),
                        module=request.values.get('module'),
                        object=request.values.get('object'),
                        path=request.values.get('path'),
                        metrics=request.values.get('metrics'),
                        score=request.values.get('score'))
        model.save()
        response = jsonify(data=model.to_dict(), **kwargs)
        response.status_code = 200
    except Exception as e:
        kwargs['done'] = False
        response = jsonify(
            message='failed to upload',
            error=str(e),
            **kwargs
        )
        response.status_code = 400
    return response


@api.route('/model/<int:model_id>')
def get_model_info(model_id):
    _is_authorized()

    kwargs = {
        'task': 'read-only',
        'type': 'model',
        'done': True
    }

    model = MLModel.get(model_id)
    if model is None:
        data = {}
    else:
        data = model.to_dict()
    response = jsonify(data=data, **kwargs)
    response.status_code = 200
    return response


@api.route('/predict/question_type', methods=['POST'])
def predict_question_type():
    """Predict question type by given question."""
    question = request.values.get('question')

    kwargs = {
        'task': 'prediction',
        'type': 'question type',
        'done': True
    }

    # empty path is trigger to execute only question type prediction
    pred = asyncio.run(run_model('', question))
    if pred == '<e>':
        kwargs['done'] = False
        response = jsonify(question=question,
                           message='error occured',
                           **kwargs)
        response.status_code = 400
    else:
        response = jsonify(question=question, answer=pred, **kwargs)
        response.status_code = 200
    return response


@api.route('/logs/requests', methods=['GET', 'POST'])
def extract_requests_logs():
    _is_authorized()

    logs = RequestLog.query()

    if request.method == 'POST':
        q_type = request.values.get('question_type')
        if q_type:
            logs = logs.filter(RequestLog.question_type.has(type=q_type))
        
        img_id = request.values.get('image_id')
        if img_id:
            logs = logs.filter(RequestLog.image_id==img_id)

    kwargs = {
        'task': 'read-only',
        'type': 'request log',
        'done': True
    }

    response = jsonify(data=[log.to_dict() for log in logs.all()],
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/logs/request/<int:req_id>')
def extract_requst_log(req_id):
    _is_authorized()

    log = RequestLog.get(req_id)
    kwargs = {
        'task': 'read-only',
        'type': 'request log',
        'done': True
    }

    response = jsonify(data=log.to_dict(),
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/logs/qa/<int:req_id>')
def extract_qa_by_request_log(req_id):
    _is_authorized()

    log = RequestLog.get(req_id)
    kwargs = {
        'task': 'read-only',
        'type': 'request log',
        'done': True
    }

    data = {
        'request_id': req_id,
        'question': Question.get(log.question_id).question,
        'prediction': log.score.prediction,
        'image': Image.get(log.image_id).filename,
        'figure': WeightFigure.get(log.fig_id).filename,
    }

    response = jsonify(data=data, **kwargs)
    response.status_code = 200
    return response


@api.route('/logs/predictions', methods=['GET', 'POST'])
def extract_prediction_logs():
    _is_authorized()

    logs = RequestLog.query()

    if request.method == 'POST':
        q_type = request.values.get('question_type')
        logs = logs.filter(RequestLog.question_type.has(type=q_type))

    kwargs = {
        'task': 'read-only',
        'type': 'prediction log',
        'done': True
    }

    response = jsonify(data=[
        {
            'request_id': log.id,
            'rate': log.score.rate,
            'prediction': log.score.prediction,
            'probability': log.score.probability,
            'answer': log.score.answer,
            'predicted_time': log.score.predicted_time,
        } for log in logs.all() if log.score is not None],
        **kwargs)
    response.status_code = 200
    return response


@api.route('/updates/all')
def update_list():
    _is_authorized()

    kwargs = {
        'task': 'read-only',
        'type': 'update',
        'done': True
    }

    response = jsonify(data=[u.to_dict() for u in Update.query().all()],
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/update/register', methods=['POST'])
def add_update():
    _is_authorized()

    content = request.values.get('content')
    u = Update(content=content)

    kwargs = {
        'task': 'register',
        'type': 'update',
        'done': True
    }

    try:
        u.save()
        response = jsonify(data=u.to_dict(), **kwargs)
        response.status_code = 200
    except Exception as e:
        kwargs['done'] = False
        response = jsonify(message='failed to upload',
                           error=str(e),
                           **kwargs)
        response.status_code = 400

    return response


@api.route('/references/all')
def reference_list():
    _is_authorized()

    kwargs = {
        'task': 'read-only',
        'type': 'reference',
        'done': True
    }

    response = jsonify(data=[c.to_dict() for c in Citation.query().all()],
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/reference/register', methods=['POST'])
def add_reference():
    _is_authorized()

    c = Citation(author=request.values.get('author'),
                 title=request.values.get('title'),
                 year=request.values.get('year'),
                 url=request.values.get('url'))

    kwargs = {
        'task': 'register',
        'type': 'reference',
        'done': True
    }

    try:
        c.save()
        response = jsonify(data=c.to_dict(), **kwargs)
        response.status_code = 200
    except Exception as e:
        kwargs['done'] = False
        response = jsonify(message='failed to upload',
                           error=str(e),
                           **kwargs)
        response.status_code = 400
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
