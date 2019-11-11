#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Column, String, Integer, Boolean, DateTime

from main.orm.db import Base
from main.mixins.models import BaseMixin


class Image(BaseMixin, Base):
    __tablename__ = 'image'

    filename = Column(String(64), nullable=False)
    # image is processed and stored as numpy array
    processed = Column(Boolean, nullable=False, default=False)
    saved_at = Column(DateTime, default=datetime.utcnow())
