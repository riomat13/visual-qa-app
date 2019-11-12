#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import Form, StringField, SubmitField, validators
from wtforms.validators import Length, DataRequired


class QuestionForm(FlaskForm):
    question = StringField('Question', validators=[DataRequired()])
    submit = SubmitField('Submit')
