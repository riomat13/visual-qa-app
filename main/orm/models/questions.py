#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy import Column, String

from main.orm.db import Base
from main.mixins.models import BaseMixin


class QuestionType(BaseMixin, Base):
    __tablename__ = 'question_type'

    type = Column(String(64), unique=True, nullable=False)

    @classmethod
    def register(cls, type):
        """Shortcut to register new question type."""
        model = cls(type=type)
        model.save()
