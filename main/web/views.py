#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path

from flask import request, render_template, redirect, url_for, session
from werkzeug import secure_filename

from . import base
from main.settings import ROOT_DIR


# TODO: replace with actual function
run_model = lambda x: 'test result'


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/prediction', methods=['GET', 'POST'])
def prediction():
    filename = None
    pred = None

    if request.method == 'POST':
        # if submitted to preduct
        if request.form['action'] == 'Submit':
            # if image is not uploaded
            filename = session.get('image')
            if not filename:
                # TODO: add warning
                return redirect(url_for('base.prediction'))

            # TODO:
            pred = run_model(filename)

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
            f.save(os.path.join(ROOT_DIR, f'main/web/static/media/uploaded/{filename}'))

    return render_template('prediction.html', filename=filename, prediction=pred)
