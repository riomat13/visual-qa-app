#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio

from flask import g, request, render_template, redirect, url_for, session, flash
from werkzeug import secure_filename

from . import base
from main.settings import Config
from main.web.forms import QuestionForm, UpdateForm, CitationForm
from main.web.auth import verify_user, login_required
from main.models.client import run_model
from main.orm.models.web import Update, Citation
from main.orm.models.data import Image, Question


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not verify_user(username, password, email=email):
            # if not valid user
            flash('Provided information is incorrect')
            return redirect(url_for('base.login'))

        user = g.user

        # set user to session after clear all hold information
        session.clear()
        session['user_id'] = user.id
        return redirect(url_for('base.index'))

    return render_template('login.html')


@base.route('/logout')
def logout():
    if 'user' in g:
        g.pop('user')
    session.clear()
    return redirect(url_for('base.index'))


@base.route('/prediction', methods=['GET', 'POST'])
def prediction():
    filename = session.get('image', None)
    question = None
    pred = None

    form = QuestionForm()

    if request.method == 'POST':
        # if submitted to predict
        if form.validate_on_submit():
            question = form.question.data
            form.question.data = ''

            if filename is None:
                flash('Image is not provided')
                return redirect(url_for('base.prediction'))

            Question(question=question).save()
            path = os.path.join(Config.UPLOAD_DIR, filename)
            pred = asyncio.run(run_model(path, question))

        # uploaded an image
        if request.form['action'] == 'upload':
            # image upload
            f = request.files['file']
            # save a file to be saved safely in file system
            filename = secure_filename(f.filename)
            Image(filename=filename).save()
            
            if not filename:
                return redirect(url_for('base.prediction'))

            # save file name to predict
            session['image'] = filename

            # save image data for later investigation
            f.save(os.path.join(Config.UPLOAD_DIR, f'{filename}'))
    else:
        # clear when re-enter the page
        if 'image' in session:
            session.pop('image')

    return render_template('prediction.html',
                           filename=filename,
                           question=question,
                           prediction=pred,
                           form=form)


@base.route('/note')
def note():
    updates = [update.to_dict() for update in Update.query().all()]
    references = [ref.to_dict() for ref in Citation.query().all()]

    return render_template('note.html',
                           updates=updates,
                           references=references)


@base.route('/update/register', methods=['GET', 'POST'])
@login_required(admin=True)
def update_register():
    form = UpdateForm()

    if form.validate_on_submit():
        content = form.content.data
        Update(content=content).save()
        return redirect(url_for('base.note'))
    return render_template('update_form.html', form=form)


@base.route('/update/edit/<int:update_id>', methods=['GET', 'PUT'])
@login_required(admin=True)
def update_edit(update_id):
    update = Update.get(update_id)
    form = UpdateForm()

    if form.validate_on_submit():
        update.content = form.content.data
        update.save()
        return redirect(url_for('base.note'))
    return render_template('update_form.html', form=form)


@base.route('/reference/register', methods=['GET', 'POST'])
@login_required(admin=True)
def ref_register():
    form = CitationForm()
    if form.validate_on_submit():
        author = form.author.data
        title = form.title.data
        year = form.year.data
        url = form.url.data
        Citation(author=author, name=name, year=year, link=link).save()
        return redirect(url_for('base.note'))
    return render_template('citation_form.html', form=form)


@base.route('/reference/edit/<int:ref_id>', methods=['GET', 'PUT'])
@login_required(admin=True)
def ref_edit(ref_id):
    cite = Citation.get(ref_id)

    form = CitationForm()
    if form.validate_on_submit():
        cite.author = form.author.data
        cite.title = form.title.data
        cite.year = form.year.data
        cite.url = form.url.data
        cite.save()
        return redirect(url_for('base.note'))
    return render_template('citation_form.html', form=form)
