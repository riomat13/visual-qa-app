#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio

from flask import request, render_template, redirect, url_for, session, flash
from werkzeug import secure_filename

from . import base
from main.settings import Config
from main.web.forms import QuestionForm
from main.web.auth import verify_user
from main.models.client import run_model


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        user = verify_user(username, password, email=email)
        if user is None:
            # if not valid user
            flash('Provided information is incorrect')
            return redirect(url_for('base.login'))

        # set user to session after clear all hold information
        session.clear()
        session['user_id'] = user.id
        return redirect(url_for('base.index'))

    return render_template('login.html')


@base.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('base.index'))


@base.route('/prediction', methods=['GET', 'POST'])
def prediction():
    filename = session.get('image', None)
    question = None
    pred = None

    form = QuestionForm()

    if request.method == 'POST':
        # if submitted to preduct
        if form.validate_on_submit():
            question = form.question.data
            form.question.data = ''

            if filename is None:
                flash('Image is not provided')
                return redirect(url_for('base.prediction'))

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
    else:
        # clear when re-enter the page
        if 'image' in session:
            session.pop('image')

    return render_template('prediction.html',
                           filename=filename,
                           question=question,
                           prediction=pred,
                           form=form)
