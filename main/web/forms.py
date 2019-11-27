#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import (
    Form,
    StringField, IntegerField, PasswordField,
    SubmitField,
    validators
)
from wtforms.validators import Length, DataRequired


class UserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class QuestionForm(FlaskForm):
    question = StringField('Question', validators=[DataRequired()])
    submit = SubmitField('Submit')


class UpdateForm(FlaskForm):
    content = StringField('Content', validators=[DataRequired()])
    submit = SubmitField('Submit')


class CitationForm(FlaskForm):
    author = StringField('Author')
    title = StringField('Title', validators=[DataRequired()])
    year = IntegerField('Year')
    url = StringField('URL')
    submit = SubmitField('Submit')
