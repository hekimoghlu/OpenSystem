#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Richard Hughes <richard@hughsie.com>
#
# pylint: disable=import-outside-toplevel
#
# SPDX-License-Identifier: GPL-3.0+

import datetime
import json
import hashlib

from sqlalchemy import or_
from sqlalchemy.orm import load_only

from flask import Response

ODRS_CUTOFF_YEARS = 5
ODRS_REPORTED_CNT = 2


def json_success(msg=None, errcode=200):
    """Success handler: JSON output"""
    item = {}
    item["success"] = True
    if msg:
        item["msg"] = msg
    dat = json.dumps(item, sort_keys=True, indent=4, separators=(",", ": "))
    return Response(response=dat, status=errcode, mimetype="application/json")


def json_error(msg=None, errcode=400):
    """Error handler: JSON output"""
    item = {}
    item["success"] = False
    if msg:
        item["msg"] = msg
    dat = json.dumps(item, sort_keys=True, indent=4, separators=(",", ": "))
    return Response(response=dat, status=errcode, mimetype="application/json")


def _get_datestr_from_dt(when):
    return int("%04i%02i%02i" % (when.year, when.month, when.day))


def _get_user_key(user_hash, app_id):
    from odrs import app

    key = "invalid"
    try:
        key = hashlib.sha1(
            app.secret_key.encode("utf-8")
            + user_hash.encode("utf-8")
            + app_id.encode("utf-8")
        ).hexdigest()
    except UnicodeEncodeError as e:
        print("invalid input: %s,%s: %s" % (user_hash, app_id, str(e)))
    return key


def _eventlog_add(
    user_addr=None, user_id=None, app_id=None, message=None, important=False
):
    """Adds a warning to the event log"""
    from .models import Event
    from odrs import db

    db.session.add(Event(user_addr, user_id, app_id, message, important))
    db.session.commit()


def _query_reviews_for_app(app_ids):
    """Return all valid reviews for the given app IDs"""
    from odrs import db
    from odrs.models import Review, Component

    cutoff_days = ODRS_CUTOFF_YEARS * 365
    cutoff = datetime.date.today() - datetime.timedelta(days=cutoff_days)

    # Note that fields here should probably be indexed for performance.
    return (
        db.session.query(Review)
            .join(Component)
            .filter(Component.app_id.in_(app_ids))
            .filter(Review.reported < ODRS_REPORTED_CNT)
            .filter(Review.date_created > cutoff)
    )

def _get_rating_for_component(component, min_total=1):
    """Gets the ratings information for the application"""
    from odrs.models import Review

    # get all ratings for app
    array = [0] * 6
    for review in (
        _query_reviews_for_app(component.app_ids)
        .options(load_only(Review.rating))
    ):
        idx = int(review.rating / 20)
        if idx > 5:
            continue
        array[idx] += 1

    # nothing found
    if sum(array) < min_total:
        return []

    # return as dict
    item = {"total": sum(array)}
    for idx in range(6):
        item["star{}".format(idx)] = array[idx]
    return item


def _password_hash(value):
    """Generate a legacy salted hash of the password string"""
    salt = "odrs%%%"
    return hashlib.sha1(salt.encode("utf-8") + value.encode("utf-8")).hexdigest()


def _addr_hash(value):
    """Generate a salted hash of the IP address"""
    from odrs import app

    return hashlib.sha1((app.secret_key + value).encode("utf-8")).hexdigest()


def _get_taboos_for_locale(locale):
    from .models import Taboo
    from odrs import db

    if locale.find("_") != -1:
        lang, _ = locale.split("_", maxsplit=1)
        return (
            db.session.query(Taboo)
            .filter(
                or_(Taboo.locale == locale, Taboo.locale == lang, Taboo.locale == "en")
            )
            .all()
        )
    return (
        db.session.query(Taboo)
        .filter(or_(Taboo.locale == locale, Taboo.locale == "en"))
        .all()
    )


def _sanitised_input(val):

    # remove trailing whitespace
    val = val.strip()

    # fix up style issues
    val = val.replace("!!!", "!")
    val = val.replace(":)", "")
    val = val.replace("  ", " ")

    return val


def _sanitised_summary(val):
    val = _sanitised_input(val)
    if val.endswith("."):
        val = val[: len(val) - 1]
    return val


def _sanitised_description(val):
    return _sanitised_input(val)


def _sanitised_version(val):

    # remove epoch
    idx = val.find(":")
    if idx != -1:
        val = val[idx + 1 :]

    # remove distro addition
    idx = val.find("+")
    if idx != -1:
        val = val[:idx]
    idx = val.find("~")
    if idx != -1:
        val = val[:idx]

    return val


def _locale_is_compatible(l1, l2):
    """Returns True if the locale is compatible"""

    # trivial case
    if l1 == l2:
        return True

    # language code matches
    lang1 = l1.split("_")[0]
    lang2 = l2.split("_")[0]
    if lang1 == lang2:
        return True

    # LANG=C
    en_langs = ["C", "en"]
    if lang1 in en_langs and lang2 in en_langs:
        return True

    # perhaps include other locale quirks here?
    return False
