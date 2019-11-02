#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask


def create_app(mode='local'):
    # update current config based on given mode
    from main.settings import set_config
    set_config(mode)

    # import updated config
    from main.settings import Config

    app = Flask(__name__)
    app.config.from_object(Config)

    from main.web import base as base_bp
    app.register_blueprint(base_bp)

    from main.web.api import api as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    engine = app.config['DATABASE_URI']

    from main.orm.db import session_builder
    session_builder()

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        from main.orm.db import session_removal
        session_removal()

    return app
