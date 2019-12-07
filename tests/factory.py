#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob
import logging
import random

from faker import Faker

from main.settings.test import TestConfig
from main.orm.db import provide_session, session_scope
from main.orm.models.base import User
from main.orm.models.ml import PredictionScore, RequestLog, QuestionType
from main.orm.models.data import Image, Question
from main.orm.models.web import Citation, Update

log = logging.getLogger(__name__)

fake = Faker()


def _QuestionTypeFactory():
    factory = None

    def func():
        nonlocal factory

        if factory is None:
            group = ['what', 'which', 'when', 'who', 'why', 'how']
            for type_ in group:
                q = QuestionType(type=type_)
                try:
                    q.save()
                except Exception as e:
                    continue

            def _factory():
                return QuestionType.get(fake.random_int(1, QuestionType.query().count()))

            factory = _factory

        return factory
    return func

QuestionTypeFactory = _QuestionTypeFactory()


class UserFactory(object):
    def __call__(self):
        return User(username=fake.user_name,
                    email=fake.email(),
                    password=fake.password())


class UserAdminFactory(object):
    def __call__(self):
        return User(username=fake.user_name(),
                    email=fake.email(),
                    password=fake.password(),
                    is_admin=True)


class RequestLogFactory(object):
    def __call__(self):
        logged_time = fake.past_date(start_date='-30d')
        log_type = fake.word(ext_word_list=['success', 'error', 'warning', 'debug'])
        log_text = fake.text(max_nb_chars=64)
        return RequestLog(logged_time=logged_time,
                          log_type=log_type,
                          log_text=log_text)


class PredictionScoreFactory(object):
    def __call__(self):
        qtype_factory = QuestionTypeFactory()
        prediction = fake.text(max_nb_chars=64)
        log = qtype_factory()
        predicted_time = fake.past_date(start_date='-30d')
        return PredictionScore(prediction=prediction,
                               log_id=log.id,
                               predicted_time=predicted_time)


class ImageFactory(object):
    def __init__(self):
        self._test_files = glob.glob(os.path.join(TestConfig.TEST_UPLOAD_DIR, '*.jpg'))

    def __call__(self):
        if not self._test_files:
            raise RuntimeError('data already created')

        img_file = self._test_files.pop()
        filename = os.path.basename(img_file)
        target_path = os.path.join(TestConfig.STATIC_DIR, TestConfig.UPLOAD_DIR, filename)

        # if not exist, copy image to actual directory
        if not os.path.isfile(target_path):
            shutil.copy(img_file, target_path)
        saved_at = fake.past_date(start_date='-30d')

        return Image(filename=filename, saved_at=saved_at)


class QuestionFactory(object):
    def __call__(self):
        question = fake.text(max_nb_chars=64)
        return Question(question=question)


class CitationFactory(object):
    def __call__(self):
        author = fake.name()
        title = fake.text(max_nb_chars=64)
        year = fake.random_int(min=1995, max=2019, step=1)
        url = fake.url()
        return Citation(author=author, title=title, year=year, url=url)


class UpdateFactory(object):
    def __call__(self):
        content = fake.text(max_nb_chars=64)
        update_at = fake.past_date(start_date='-30d')
        return Update(content=content, update_at=update_at)


def generate_fake_logs(n=1):
    request_log_factory = RequestLogFactory()
    score_factory = PredictionScoreFactory()

    for _ in range(n):
        req = request_log_factory()
        req.save()
        pred = score_factory()
        pred.log_id = req.id
        pred.save()


def generate_fake_notes(n=1):
    citation_factory = CitationFactory()
    update_factory = UpdateFactory()

    for _ in range(n):
        cit = citation_factory()
        cit.save()
        up = update_factory()
        up.save()


def generate_fake_dataset(n=1):
    img_factory = ImageFactory()
    q_factory = QuestionFactory()

    for _ in range(n):
        img = img_factory()
        img.save()
        q = q_factory()
        q.save()
