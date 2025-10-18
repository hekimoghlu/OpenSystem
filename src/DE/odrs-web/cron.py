#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# pylint: disable=singleton-comparison
#
# Copyright (C) 2015-2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+

import json
import sys
import datetime
import csv
import gzip

from lxml import etree as ET

from odrs import db

from odrs.models import Review, Taboo, Component
from odrs.util import _get_rating_for_component, _get_taboos_for_locale


def _fsck_components():

    # get all existing components
    components = {}
    for component in (
        db.session.query(Component)
        .filter(Component.app_id != "")
        .order_by(Component.app_id.asc())
    ):
        components[component.app_id] = component

    # guessed, thanks Canonical :/
    for app_id in components:
        if not app_id.startswith("io.snapcraft."):
            continue
        if components[app_id].component_id_parent:
            continue
        name, _ = app_id[13:].rsplit("-", maxsplit=1)
        parent = components.get(name + ".desktop")
        if not parent:
            continue
        print(
            "adding snapcraft parent for {} -> {}".format(
                components[app_id].app_id, parent.app_id
            )
        )
        parent.adopt(components[app_id])

    # upstream drops the .desktop sometimes
    for app_id in components:
        if components[app_id].component_id_parent:
            continue
        app_id_new = app_id.replace(".desktop", "")
        if app_id == app_id_new:
            continue
        parent = components.get(app_id_new)
        if not parent:
            continue
        print(
            "adding parent for {} -> {}".format(
                components[app_id].app_id, parent.app_id
            )
        )
        parent.adopt(components[app_id])

    # API change :/
    for app_id in components:
        if not app_id.endswith(".shell-extension"):
            continue
        if components[app_id].component_id_parent:
            continue
        app_id_new = app_id.replace(".shell-extension", "")
        app_id_new = app_id_new.replace("@", "_")
        parent = components.get(app_id_new)
        if not parent:
            continue
        print(
            "adding shell parent for {} -> {}".format(
                components[app_id].app_id, parent.app_id
            )
        )
        parent.adopt(components[app_id])
    db.session.commit()


def _auto_delete(days=31):

    # delete all reviews with taboo, otherwise the moderatorators get overwhelmed
    for review in (
        db.session.query(Review)
        .filter(Review.reported == 5)
        .order_by(Review.date_created.asc())
        .limit(1000)
    ):
        db.session.delete(review)
    db.session.commit()

    # clean up all old deleted reviews
    since = datetime.datetime.now() - datetime.timedelta(days=days)
    for review in (
        db.session.query(Review)
        .filter(Review.date_deleted != None)
        .filter(Review.date_deleted < since)
        .order_by(Review.date_created.asc())
        .limit(1000)
    ):
        db.session.delete(review)
    db.session.commit()


def _fsck():
    _auto_delete()
    _fsck_components()


def _regenerate_ratings(fn):
    item = {}
    for component in db.session.query(Component).order_by(Component.app_id.asc()):
        ratings = _get_rating_for_component(component, 2)
        if len(ratings) == 0:
            continue
        item[component.app_id] = ratings

    # dump to file
    with open(fn, "w") as outfd:
        outfd.write(json.dumps(item, sort_keys=True, indent=4, separators=(",", ": ")))


def _taboo_check():

    # this is moderately expensive, so cache for each locale
    taboos = {}
    for review in db.session.query(Review).filter(Review.reported < 5):
        if review.locale not in taboos:
            taboos[review.locale] = _get_taboos_for_locale(review.locale)
        matched_taboos = review.matches_taboos(taboos[review.locale])
        if matched_taboos:
            for taboo in matched_taboos:
                print(review.review_id, review.locale, taboo.value)
            review.reported = 5
    db.session.commit()


def _appstream_import(fn):

    # get existing components
    app_ids = {}
    for component in db.session.query(Component):
        app_ids[component.app_id] = component

    # parse xml
    with gzip.open(fn, "rb") as f:
        for component in ET.fromstring(f.read()).xpath("/components/component"):
            app_id = component.xpath("id")[0].text
            if app_id not in app_ids:
                continue
            children = []
            for provide in component.xpath("provides/id"):
                child_id = provide.text
                if child_id == app_id:
                    continue
                if child_id not in app_ids:
                    continue
                if app_ids[child_id].component_id_parent:
                    continue
                children.append(app_ids[child_id])
            if not children:
                continue
            parent = app_ids[app_id]
            for child in children:
                print(
                    "adding AppStream parent for {} -> {}".format(
                        child.app_id, parent.app_id
                    )
                )
                parent.adopt(child)
    db.session.commit()


def _taboo_import_item(taboos, locale, value, description, severity):
    key = locale + ":" + value
    if key in taboos:
        taboo = taboos[key]
        is_dirty = False
        if taboo.description != description:
            print(
                'Modifying {} description from "{}" to "{}"'.format(
                    key, taboo.description, description
                )
            )
            taboo.description = description
            is_dirty = True
        if taboo.severity != severity:
            print(
                'Modifying {} severity from "{}" to "{}"'.format(
                    key, taboo.severity, severity
                )
            )
            taboo.severity = severity
            is_dirty = True
        return is_dirty
    if value.find(" ") != -1:
        print("Ignoring", locale, value)
        return False
    if value.lower() != value:
        print("Ignoring", locale, value)
        return False
    taboo = Taboo(locale, value, description, severity=severity)
    taboos[key] = taboo
    print("Adding {}".format(key))
    db.session.add(taboo)
    return True


def _taboo_import(fn):

    # get all the taboos in one database call
    taboos = {}
    for taboo in db.session.query(Taboo):
        key = taboo.locale + ":" + taboo.value
        taboos[key] = taboo

    # add any new ones
    is_dirty = False
    with open(fn, newline="") as csvfile:
        for locale, values, description, severity in csv.reader(csvfile):
            locale = locale.strip()
            description = description.strip()
            severity = int(severity)
            for value in values.split("/"):
                value = value.strip()
                is_dirty = (
                    _taboo_import_item(taboos, locale, value, description, severity)
                    or is_dirty
                )
    db.session.commit()

    # if dirty, check all the existing reviews
    if is_dirty:
        _taboo_check()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: %s ratings|fsck|taboo-check|taboo-import" % sys.argv[0])
        sys.exit(1)

    # create the ratings data
    if sys.argv[1] == "ratings":
        if len(sys.argv) < 3:
            print("Usage: %s ratings filename" % sys.argv[0])
            sys.exit(1)
        _regenerate_ratings(sys.argv[2])
    elif sys.argv[1] == "fsck":
        _fsck()
    elif sys.argv[1] == "taboo-check":
        _taboo_check()
    elif sys.argv[1] == "taboo-import":
        if len(sys.argv) < 3:
            print("Usage: %s taboo-import filename" % sys.argv[0])
            sys.exit(1)
        _taboo_import(sys.argv[2])
    elif sys.argv[1] == "appstream-import":
        if len(sys.argv) < 3:
            print("Usage: %s taboo-import filename" % sys.argv[0])
            sys.exit(1)
        _appstream_import(sys.argv[2])
    else:
        print("cron mode %s not known" % sys.argv[1])
        sys.exit(1)

    # success
    sys.exit(0)
