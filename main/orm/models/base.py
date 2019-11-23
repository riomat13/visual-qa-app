#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy import Column, String
from werkzeug.security import generate_password_hash, check_password_hash

from main.orm.db import Base
from main.mixins.models import BaseMixin


class User(BaseMixin, Base):
    __tablename__ = 'user'

    username = Column(String(32), nullable=False, unique=True)
    email = Column(String(128), nullable=False)
    password_hash = Column(String(128), nullable=False)

    @property
    def password(self):
        raise AttributeError('not allowed to read password')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
