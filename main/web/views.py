#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio

from flask import request, render_template, redirect, url_for, session, flash
from werkzeug import secure_filename

from . import base
from main.settings import Config
from main.web.forms import QuestionForm
from main.models.client import run_model


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/prediction', methods=['GET', 'POST'])
def prediction():
    filename = None
    question = None
    pred = None

    form = QuestionForm()

    if request.method == 'POST':
        # if submitted to preduct
        if form.validate_on_submit():
            question = form.question.data
            form.question.data = ''

            # handle uploaded image
            filename = session.get('image')
            # TODO: make reusable, add remove current image button
            if filename is not None:
                session.pop('image')
            else:
                flash('Image is not provided')
                return redirect(url_for('base.prediction'))

            # TODO: add form for sentence
            path = os.path.join(Config.UPLOAD_DIR, filename)
            pred = asyncio.run(run_model(path, question))

        # uploaded an image
        if request.form['action'] == 'upload':
            # image upload
            f = request.files['file']
            # save a file to be saved safely in file system
            filename = secure_filename(f.filename)
            
            if not filename:
                return redirect(url_for('base.prediction'))

            # save file name to predict
            session['image'] = filename

            # save image data for later investigation
            f.save(os.path.join(Config.UPLOAD_DIR, '{filename}'))

    return render_template('prediction.html',
                           filename=filename,
                           question=question,
                           prediction=pred,
                           form=form)
