#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import os
import asyncio

from sqlalchemy.exc import IntegrityError

from flask import Blueprint
from flask import abort, jsonify, request, session

from . import api
from main.settings import Config
from werkzeug.utils import secure_filename

from main.web.auth import verify_user, generate_token, login_required
from main.ml.client import run_model
from main.orm.models.base import User
from main.orm.models.ml import MLModel, RequestLog, PredictionScore, QuestionType
from main.orm.models.web import Update, Citation
from main.orm.models.data import Image, Question, WeightFigure

log = logging.getLogger(__name__)

api = Blueprint('api', __name__)


#def _is_authorized(token):
#    """Used for first login."""
#    authorization = request.authorization
#    user = verify_user(**authorization)
#    if authorization is None or not user:
#        abort(403)


@api.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    email = request.json.get('email')
    password = request.json.get('password')

    user = verify_user(username=username, email=email, password=password)

    if user is not None:
        token = generate_token(user)
        kwargs = {'authenticated': True,
                  'status': 'success',
                  'message': 'successfully logged in',
                  'token': token.decode('ascii')}
        response = jsonify(kwargs)
        response.status_code = 200

    else:
        kwargs = {'status': 'error',
                  'message': 'failed to log in'}
        response = jsonify(kwargs)
        response.status_code = 403

    return response


@api.route('/upload/image', methods=['POST'])
def upload_image():
    img_file = request.files['file']

    if img_file:
        filename = secure_filename(img_file.filename)

        # rename if the same file name does exist
        if Image.query().filter_by(filename=filename).count() > 0:
            from datetime import datetime
            dttm = datetime.now()
            dttm = dttm.strftime('%Y%m%s%H%M%S')
            filename, ext = os.path.splitext(filename)
            # max length of filename is 64
            filename = f'{filename[:64-len(dttm)-len(ext)-1]}_{dttm}{ext}'
        elif len(filename) > 64:
            filename, ext = os.path.splitext(filename)
            # max length of filename is 64
            filename = f'{filename[:64-len(ext)]}{ext}'

        try:
            img = Image(filename=filename)
            img.save()
        except Exception:
            kwargs = {
                'status': 'error',
                'message': 'failed to save image'
            }
            response = jsonify(kwargs)
            response.status_code = 400
        else:
            filepath = os.path.join(Config.UPLOAD_DIR, filename)
            img_file.save(os.path.join(Config.STATIC_DIR, filepath))

            kwargs = {
                'status': 'success',
                'message': 'uploaded image'
            }
            data = {'img_id': img.id}
            response = jsonify(data=data, **kwargs)
            response.status_code = 201
    else:
        kwargs = {
            'status': 'error',
            'message': 'failed to upload image'
        }
        response = jsonify(data=data, **kwargs)
        response.status_code = 400

    return response


@api.route('/prediction', methods=['POST'])
def prediction():
    image_id = request.json.get('img_id')
    question = request.json.get('question')

    img = Image.get(image_id)

    if img is None:
        kwargs = {'status': 'error',
                  'message': 'invalid image id'}
        response = jsonify(kwargs)
        response.status_code = 400
    else:
        filepath = os.path.join(Config.UPLOAD_DIR, img.filename)
        path = os.path.join(Config.STATIC_DIR, f'{filepath}')
        pred, fig_id = asyncio.run(run_model(path, question))

        if fig_id > 0:
            #fig = WeightFigure.get(fig_id)
            #figfile = fig.filename
            #figpath = os.path.join(Config.FIG_DIR, figfile)
            pass

        kwargs = {'status': 'success'}
        data = {'prediction': pred}
        response = jsonify(data=data, **kwargs)
        response.status_code = 200

    return response


@api.route('/note')
def note():
    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'note',
    }

    updates = [update.to_dict(date=True) for update in Update.query().order_by(Update.id.desc()).all()]
    references = [ref.to_dict() for ref in Citation.query().all()]

    data = {'updates': updates,
            'references': references}

    response = jsonify(data=data, **kwargs)
    response.status_code = 200

    return response


@api.route('/update/register', methods=['POST'])
@login_required(admin=True)
def update_item_register():
    kwargs = {
        'status': 'success',
        'task': 'register',
        'type': 'update',
    }

    try:
        content = request.json.get('content')
        summary = request.json.get('summary')
        Update(content=content, summary=summary).save()
        response = jsonify(**kwargs)
        response.status_code = 201
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)

        response = jsonify(**kwargs)
        response.status_code = 400

    return response


@api.route('/update/items/all', methods=['POST'])
@login_required()
def fetch_update_items_all():
    items = Update.query().all()

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'update',
    }

    response = jsonify(data=[item.to_dict() for item in items], **kwargs)
    response.status_code = 200

    return response


@api.route('/update/item/<int:item_id>', methods=['POST'])
@login_required()
def fetch_update_item(item_id):
    item = Update.get(item_id)

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'update',
    }

    if item is None:
        data = {}
    else:
        data = item.to_dict()

    response = jsonify(data=data, **kwargs)
    response.status_code = 200

    return response


@api.route('/update/edit/<int:item_id>', methods=['PUT'])
@login_required(admin=True)
def update_item_edit(item_id):
    update = Update.get(item_id)

    kwargs = {
        'status': 'success',
        'task': 'edit',
        'type': 'update',
    }

    if update is None:
        kwargs['status'] = 'error'
        kwargs['message'] = 'item not found'
        response = jsonify(**kwargs)
        response.status_code = 400
        return response

    try:
        content = request.json.get('content')
        summary = request.json.get('summary')
        if content is not None:
            update.content = content
        if summary is not None:
            update.summary = summary
        update.save()

        response = jsonify(**kwargs)
        response.status_code = 201
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)
        response = jsonify(**kwargs)
        response.status_code = 400

    return response


@api.route('/reference/register', methods=['POST'])
@login_required(admin=True)
def ref_item_register():
    kwargs = {
        'status': 'success',
        'task': 'write',
        'type': 'reference',
    }

    try:
        author = request.json.get('author')
        title = request.json.get('title')
        year = request.json.get('year')
        url = request.json.get('url')
        summary = request.json.get('summary')
        Citation(author=author,
                 title=title,
                 year=year,
                 url=url,
                 summary=summary).save()

        response = jsonify(**kwargs)
        response.status_code = 201
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)

        response = jsonify(**kwargs)
        response.status_code = 400

    return response


@api.route('/reference/items/all', methods=['POST'])
@login_required
def fetch_ref_items_all():
    items = Citation.query().all()

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'reference',
    }

    response = jsonify(data=[item.to_dict() for item in items], **kwargs)
    response.status_code = 200

    return response


@api.route('/reference/item/<int:item_id>', methods=['POST'])
@login_required
def fetch_ref_item(item_id):
    item = Citation.get(item_id)

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'reference',
    }

    if item is None:
        data = {}
    else:
        data = item.to_dict()

    response = jsonify(data=data, **kwargs)
    response.status_code = 200

    return response


@api.route('/reference/edit/<int:item_id>', methods=['PUT'])
@login_required(admin=True)
def ref_item_edit(item_id):
    cite = Citation.get(item_id)

    kwargs = {
        'status': 'success',
        'task': 'edit',
        'type': 'reference',
    }

    if cite is None:
        kwargs['status'] = 'error'
        kwargs['message'] = 'item not found'
        response = jsonify(**kwargs)
        response.status_code = 400
        return response

    try:
        author = request.json.get('author')
        if author is not None:
            cite.author = author
        title = request.json.get('title')
        if title is not None:
            cite.title = title
        year = request.json.get('year')
        if year is not None:
            cite.year = year
        url = request.json.get('url')
        if url is not None:
            cite.url = url
        cite.save()

        response = jsonify(**kwargs)
        response.status_code = 201
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)
        response = jsonify(**kwargs)
        response.status_code = 400

    return response


@api.route('/models/all', methods=['POST'])
@login_required
def model_list():
    """Return model list availabel."""
    #_is_authorized()

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'model',
    }

    try:
        models = MLModel.query().all()
        response = jsonify(data=[model.to_dict() for model in models],
                           **kwargs)
        response.status_code = 200
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)
        response = jsonify(**kwargs)
        response.status_code = 400

    return response


@api.route('/register/model', methods=['POST'])
@login_required(admin=True)
def register_model():

    kwargs = {
        'status': 'success',
        'task': 'register',
        'type': 'model',
    }

    try:
        MLModel(name=request.json.get('name'),
                type=request.json.get('type'),
                category=request.json.get('category'),
                module=request.json.get('module'),
                object=request.json.get('object'),
                path=request.json.get('path'),
                metrics=request.json.get('metrics'),
                score=request.json.get('score')).save()
        response = jsonify(**kwargs)
        response.status_code = 201
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)
        response = jsonify(
            message='failed to upload',
            **kwargs
        )
        response.status_code = 400
    return response


@api.route('/model/<int:model_id>', methods=['POST'])
@login_required
def get_model_info(model_id):

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'model',
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
    question = request.json.get('question')

    kwargs = {
        'status': 'success',
        'task': 'prediction',
        'type': 'question type',
    }

    # empty path is trigger to execute only question type prediction
    pred = asyncio.run(run_model('', question))
    if pred == '<e>':
        kwargs['status'] = 'error'
        data = {'question': question}
        response = jsonify(data=data,
                           message='error occured',
                           **kwargs)
        response.status_code = 400
    else:
        data = {
            'question': question,
            'prediction': pred
        }
        response = jsonify(data=data, **kwargs)
        response.status_code = 201
    return response


@api.route('/logs/requests', methods=['POST'])
@login_required
def extract_requests_logs():

    logs = RequestLog.query()

    q_type = request.json.get('question_type')
    if q_type is not None:
        logs = logs.filter(RequestLog.question_type.has(type=q_type))

    img_id = request.json.get('image_id')
    if img_id is not None:
        logs = logs.filter(RequestLog.image_id==img_id)

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'request log',
    }

    response = jsonify(data=[log.to_dict() for log in logs.all()],
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/logs/request/<int:req_id>', methods=['POST'])
@login_required
def extract_requst_log(req_id):

    log = RequestLog.get(req_id)
    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'request log',
    }

    response = jsonify(data=log.to_dict(),
                       **kwargs)
    response.status_code = 200
    return response


@api.route('/logs/qa/<int:req_id>', methods=['POST'])
@login_required
def extract_qa_by_request_log(req_id):

    log = RequestLog.get(req_id)
    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'request log',
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


@api.route('/logs/predictions', methods=['POST'])
@login_required
def extract_prediction_logs():

    logs = RequestLog.query()

    q_type = request.json.get('question_type')
    if q_type is not None:
        logs = logs.filter_by(question_type=q_type)

    kwargs = {
        'status': 'success',
        'task': 'read-only',
        'type': 'prediction log',
    }

    try:
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
    except Exception as e:
        kwargs['status'] = 'error'
        kwargs['error'] = str(e)
        response = jsonify(**kwargs)
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
