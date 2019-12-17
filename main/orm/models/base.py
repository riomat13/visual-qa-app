#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Column, String, Boolean, DateTime, Text
from werkzeug.security import generate_password_hash, check_password_hash

from main.orm.db import Base
from main.mixins.models import BaseMixin


class User(BaseMixin, Base):
    __tablename__ = 'user'

    username = Column(String(32), nullable=False, unique=True)
    email = Column(String(128), nullable=False)
    password_hash = Column(String(128), nullable=False)
    is_admin = Column(Boolean, default=False)

    @property
    def password(self):
        raise AttributeError('not allowed to read password')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


class AppLog(BaseMixin, Base):
    __tablename__ = 'app_log'

    logged_time = Column(DateTime, default=datetime.utcnow())
    # Log Type: e.g. Info, Warning, Error
    log_type = Column(String(32), nullable=False)
    # class name if error. e.g. ValueError
    log_class = Column(String(32), nullable=True)
    # Log Detail
    log_text = Column(Text, nullable=False)

    @classmethod
    def fetch_logs(cls, log_type=None, log_class=None, log_time=None):
        """Fetch logs by type, class and/or time

        Args:
            log_type: str
                log type such as success, warning, error
            log_class: str
                class name of log type such as exception class name
            log_time: str, datetime object, or list of str or datetime
                start time to fetch logs
                if it is list, it will be [start_time, end_time]
        """
        logs = cls.query()

        if log_type is not None:
            logs = logs.filter_by(log_type=log_type)

        if log_class is not None:
            logs = logs.filter_by(log_class=log_class)

        if log_time is not None:
            if isinstance(log_time, (list, tuple)):
                logs = logs.filter(
                    AppLog.logged_time >= log_time[0],
                    AppLog.logged_time <= log_time[1]
                )
            else:
                logs = logs.filter(AppLog.logged_time>=log_time)

        return logs.all()

