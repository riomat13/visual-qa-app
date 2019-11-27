#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import wraps, partial

from flask import g, redirect, url_for, session, abort

from . import base
from main.orm.models.base import User


def verify_user(username, password, email=None):
    """Check if verify password is correct by username and password."""
    user = User.query() \
        .filter_by(username=username)

    if email is not None:
        user = user.filter_by(email=email)

    user = user.first()

    if user is None or not user.verify_password(password):
        return False

    g.user = user

    return True


def login_required(view=None, *, admin=False):
    """Decorator for authentication check.

    Args:
        admin: bool
            if this is set to True
            User logging in must be admin.
    """
    if view is None:
        return partial(login_required, admin=admin)

    @wraps(view)
    def wrapper(*, _admin=admin, **kwargs):
        if g.user is None:
            return redirect(url_for('base.login'))
        if _admin and not g.user.is_admin:
            return abort(403)
        return view(**kwargs)
    return wrapper


@base.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = User.query().filter_by(id=user_id).first()
