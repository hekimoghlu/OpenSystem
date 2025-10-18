#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# pylint: disable=invalid-name,missing-docstring,wrong-import-order,wrong-import-position
# pylint: disable=import-outside-toplevel
#
# Copyright (C) 2015-2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+

import os

from flask import Flask, flash, render_template, g, redirect, url_for
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.local import LocalProxy

from .dbutils import drop_db, init_db

app = Flask(__name__)
app.config.from_object(__name__)
if "ODRS_CONFIG" in os.environ:
    app.config.from_envvar("ODRS_CONFIG")
if "ODRS_REVIEWS_SECRET" in os.environ:
    app.secret_key = os.environ["ODRS_REVIEWS_SECRET"]
for key in ["SQLALCHEMY_DATABASE_URI", "SQLALCHEMY_TRACK_MODIFICATIONS"]:
    if key in os.environ:
        app.config[key] = os.environ[key]

db = SQLAlchemy(app)

migrate = Migrate(app, db)

csrf = CSRFProtect(app)


@app.cli.command("initdb")
def initdb_command():
    init_db(db)


@app.cli.command("dropdb")
def dropdb_command():
    drop_db(db)


lm = LoginManager(app)
lm.login_view = "odrs_login"


@app.teardown_appcontext
def shutdown_session(unused_exception=None):
    db.session.remove()


@lm.user_loader
def load_user(user_id):
    from .models import Moderator

    g.user = (
        db.session.query(Moderator).filter(Moderator.moderator_id == user_id).first()
    )
    return g.user


@app.errorhandler(404)
def error_page_not_found(msg=None):
    """Error handler: File not found"""
    flash(msg)
    return render_template("error.html"), 404


@app.errorhandler(CSRFError)
def error_csrf(e):
    flash(str(e), "danger")
    return redirect(url_for(".odrs_index"))


from odrs import views
from odrs import views_api
from odrs import views_admin
