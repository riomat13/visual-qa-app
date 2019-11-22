#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
from functools import wraps

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

engine = None
Session = None
Base = declarative_base()


def session_builder():
    global engine
    global Session

    if engine is None:
        from main.settings import Config
        engine_uri = Config.DATABASE_URI
        engine = create_engine(engine_uri)

    Session = scoped_session(
        sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    )
    return Session


def session_removal():
    global Session

    if Session is not None:
        Session.remove()
        Session = None


@contextlib.contextmanager
def session_scope():
    global Session

    if Session is None:
        session_builder()

    session = Session()

    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# Reference:
#   https://github.com/apache/airflow/blob/master/airflow/utils/db.py
def provide_session(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_session = 'session'

        # extract all names of args and kwargs
        func_params = func.__code__.co_varnames

        # if 'session' is in args and the value is given
        session_in_args = arg_session in func_params and \
            func_params.index(arg_session) < len(args)

        # if 'session' is provided as kwargs
        session_in_kwargs = arg_session in kwargs

        if session_in_kwargs or session_in_args:
            return func(*args, **kwargs)
        else:
            with session_scope() as sess:
                kwargs[arg_session] = sess
                return func(*args, **kwargs)
    return wrapper


def init_db():
    global engine
    Base.metadata.create_all(engine)


def reset_db(conn=None):
    conn = conn or engine
    Base.metadata.drop_all(conn)
    init_db()
