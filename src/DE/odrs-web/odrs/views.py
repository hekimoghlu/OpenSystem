#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# pylint: disable=invalid-name,missing-docstring
#
# Copyright (C) 2015-2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+

import os

from flask import (
    request,
    url_for,
    redirect,
    flash,
    render_template,
    send_from_directory,
    g,
)
from flask_login import login_user, logout_user

from odrs import app, db

from .models import Moderator


@app.context_processor
def _utility_processor():
    def format_rating(rating):
        nr_stars = int(rating / 20)
        tmp = ""
        for _ in range(0, nr_stars):
            tmp += "★"
        for _ in range(0, 5 - nr_stars):
            tmp += "☆"
        return tmp

    def format_truncate(tmp, length):
        if len(tmp) <= length:
            return tmp
        return tmp[:length] + "…"

    def url_for_other_page(page):
        args = request.view_args.copy()
        args["page"] = page
        return url_for(request.endpoint, **args)

    return dict(
        format_rating=format_rating,
        format_truncate=format_truncate,
        url_for_other_page=url_for_other_page,
    )


@app.route("/login", methods=["GET", "POST"])
def odrs_login():
    if request.method != "POST":
        return render_template("login.html")
    user = (
        db.session.query(Moderator)
        .filter(Moderator.username == request.form["username"])
        .first()
    )
    if not user:
        flash("Incorrect username")
        return redirect(url_for(".odrs_login"))
    if not user.verify_password(request.form["password"]):
        flash("Incorrect password")
        return redirect(url_for(".odrs_login"))
    login_user(user, remember=False)
    g.user = user
    flash("Logged in")
    return redirect(url_for(".odrs_index"))


@app.route("/logout")
def odrs_logout():
    logout_user()
    flash("Logged out.")
    return redirect(url_for(".odrs_index"))


@app.errorhandler(400)
def _error_internal(msg=None, errcode=400):
    """Error handler: Internal"""
    flash("Internal error: %s" % msg)
    return render_template("error.html"), errcode


@app.errorhandler(401)
def _error_permission_denied(msg=None):
    """Error handler: Permission Denied"""
    flash("Permission denied: %s" % msg)
    return render_template("error.html"), 401


@app.errorhandler(404)
def odrs_error_page_not_found(msg=None):
    """Error handler: File not found"""
    flash(msg)
    return render_template("error.html"), 404


@app.route("/")
def odrs_index():
    """start page"""
    return render_template("index.html")


@app.route("/privacy")
def odrs_privacy():
    """odrs_privacy page"""
    return render_template("privacy.html")


@app.route("/oars")
def oars_index():
    """OARS page"""
    return render_template("oars.html")


@app.route("/<path:resource>")
def odrs_static_resource(resource):
    """Return a static image or resource"""
    return send_from_directory(
        "%s/odrs/static/" % os.environ["HOME"], os.path.basename(resource)
    )
