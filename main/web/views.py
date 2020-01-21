#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import asyncio
import logging

from flask import g, request, render_template, redirect, url_for, session, flash
from werkzeug import secure_filename

from . import base
from main.settings import Config
from main.web.forms import UserForm, QuestionForm, UpdateForm, CitationForm
from main.web.auth import verify_user, login_required
from main.ml.client import run_model
from main.orm.models.web import Update, Citation
from main.orm.models.data import WeightFigure

log = logging.getLogger(__name__)


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/login', methods=['GET', 'POST'])
def login():
    form = UserForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            username = form.username.data
            email = form.email.data
            password = form.password.data

        if not verify_user(username, password, email=email):
            # if not valid user
            flash('Provided information is incorrect')
            return redirect(url_for('base.login'))

        user = g.user

        # set user to session after clear all hold information
        session.clear()
        session['user_id'] = user.id
        return redirect(url_for('base.index'))

    return render_template('login.html', form=form)


@base.route('/logout')
def logout():
    if 'user' in g:
        g.pop('user')
    session.clear()
    return redirect(url_for('base.index'))


@base.route('/prediction', methods=['GET', 'POST'])
def prediction():
    filepath = session.get('image', None)
    question = None
    pred = None
    figpath = None

    form = QuestionForm()

    if request.method == 'POST':
        # if submitted to predict
        if form.validate_on_submit():
            question = form.question.data
            form.question.data = ''

            if filepath is None:
                flash('Image is not provided')
                return redirect(url_for('base.prediction'))

            path = os.path.join(Config.STATIC_DIR, f'{filepath}')
            pred, fig_id = asyncio.run(run_model(path, question))
            if fig_id > 0:
                fig = WeightFigure.get(fig_id)
                figfile = fig.filename
                figpath = os.path.join(Config.FIG_DIR, figfile)

        # uploaded an image
        if request.form['action'] == 'upload':
            # image upload
            f = request.files['file']
            # save a file to be saved safely in file system
            filename = secure_filename(f.filename)
            filepath = os.path.join(Config.UPLOAD_DIR, filename)
            
            if not filename:
                return redirect(url_for('base.prediction'))

            # save file name to predict
            session['image'] = filepath

            # save image data for later investigation
            f.save(os.path.join(Config.STATIC_DIR, f'{filepath}'))
    else:
        # clear when re-enter the page
        if 'image' in session:
            session.pop('image')

    return render_template('prediction.html',
                           filepath=filepath,
                           question=question,
                           prediction=pred,
                           figpath=figpath,
                           form=form)


@base.route('/note')
def note():
    updates = [update.to_dict() for update in Update.query().order_by(Update.id.desc()).all()]
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
        summary = form.summary.data
        Update(content=content, summary=summary).save()
        return redirect(url_for('base.note'))
    return render_template('update_form.html', form=form)


@base.route('/update/items/all')
@login_required(admin=True)
def list_update_items_all():
    items = Update.query().all()
    return render_template('update_items_all.html', items=items)


@base.route('/update/item/<int:item_id>')
@login_required(admin=True)
def list_update_item(item_id):
    item = Update.get(item_id)
    if item is None:
        return redirect(url_for('base.note'))
    return render_template('update_item.html', item=item)


@base.route('/update/edit/<int:item_id>', methods=['GET', 'PUT'])
@login_required(admin=True)
def update_edit(item_id):
    update = Update.get(item_id)
    form = UpdateForm()

    if form.validate_on_submit():
        update.content = form.content.data
        update.save()
        return redirect(url_for('base.note'))

    # set initial values
    form.content.data = update.content
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
        summary = form.summary.data
        Citation(author=author,
                 title=title,
                 year=year,
                 url=url,
                 summary=summary).save()
        return redirect(url_for('base.note'))
    return render_template('citation_form.html', form=form)


@base.route('/reference/items/all')
@login_required(admin=True)
def list_ref_items_all():
    items = Citation.query().all()
    return render_template('citation_items_all.html', items=items)


@base.route('/reference/item/<int:item_id>')
@login_required(admin=True)
def list_ref_item(item_id):
    item = Citation.get(item_id)
    if item is None:
        return redirect(url_for('base.note'))
    return render_template('citation_item.html', item=item)


@base.route('/reference/edit/<int:item_id>', methods=['GET', 'PUT'])
@login_required(admin=True)
def ref_edit(item_id):
    cite = Citation.get(item_id)

    form = CitationForm()
    if form.validate_on_submit():
        cite.author = form.author.data
        cite.title = form.title.data
        cite.year = form.year.data
        cite.url = form.url.data
        cite.save()
        return redirect(url_for('base.note'))

    # set initial values
    form.author.data = cite.author
    form.title.data = cite.title
    form.year.data = cite.year
    form.url.data = cite.url
    return render_template('citation_form.html', form=form)


@base.app_errorhandler(400)
def bad_request(e):
    log.error(e)
    return '<h2>400 Bad Request</h2>', 400


@base.app_errorhandler(403)
def forbidden(e):
    log.error(e)
    return '<h2>403 Forbidden</h2>', 403


@base.app_errorhandler(404)
def page_not_found(e):
    log.error(e)
    return '<h2>404 Page Not Found</h2>', 404
