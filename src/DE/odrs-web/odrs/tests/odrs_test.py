#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+
#
# pylint: disable=fixme,too-many-public-methods,line-too-long,too-many-lines
# pylint: disable=too-many-instance-attributes,wrong-import-position,import-outside-toplevel

import os
import json
import sys
import unittest
import tempfile

# allows us to run this from the project root
sys.path.append(os.path.realpath("."))

from odrs.util import _get_user_key


class OdrsTest(unittest.TestCase):
    def setUp(self):

        # create new database
        self.db_fd, self.db_filename = tempfile.mkstemp()
        self.db_uri = "sqlite:///" + self.db_filename
        self.user_hash = "deadbeef348c0f88529f3bfd937ec1a5d90aefc7"

        # write out custom settings file
        self.cfg_fd, self.cfg_filename = tempfile.mkstemp()
        with open(self.cfg_filename, "w") as cfgfile:
            cfgfile.write(
                "\n".join(
                    [
                        "SQLALCHEMY_DATABASE_URI = '%s'" % self.db_uri,
                        "SQLALCHEMY_TRACK_MODIFICATIONS = False",
                        "SECRET_KEY = 'not-secret4'",
                        "ODRS_REVIEWS_SECRET = '1'",
                        "WTF_CSRF_CHECK_DEFAULT = False",
                        "DEBUG = True",
                    ]
                )
            )

        # create instance
        import odrs
        from odrs import db
        from odrs.dbutils import init_db

        self.app = odrs.app.test_client()
        odrs.app.config.from_pyfile(self.cfg_filename)
        with odrs.app.app_context():
            init_db(db)

        # assign user_hash to this account
        self.login()
        rv = self.app.post(
            "/admin/moderator/1/modify_by_admin",
            data=dict(
                is_enabled=True,
                is_admin=True,
                locales="en",
                user_hash=self.user_hash,
            ),
            follow_redirects=True,
        )
        assert b"Updated profile" in rv.data, rv.data
        self.logout()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_filename)
        os.close(self.cfg_fd)
        os.unlink(self.cfg_filename)

    def _login(self, username, password="Pa$$w0rd"):
        return self.app.post(
            "/login",
            data=dict(username=username, password=password),
            follow_redirects=True,
        )

    def _logout(self):
        return self.app.get("/logout", follow_redirects=True)

    def login(self, username="admin@test.com", password="Pa$$w0rd"):
        rv = self._login(username, password)
        assert b"Logged in" in rv.data, rv.data
        assert b"/admin/show/reported" in rv.data, rv.data
        assert b"Incorrect username" not in rv.data, rv.data

    def logout(self):
        rv = self._logout()
        assert b"Logged out" in rv.data, rv.data
        assert b"/admin/show/reported" not in rv.data, rv.data

    def test_admin_show_review_for_app(self):

        self.review_submit()
        self.login()
        rv = self.app.get("/admin/show/app/inkscape.desktop")
        assert b"n essential part of my daily" in rv.data, rv.data

    def test_admin_graphs(self):

        self.review_submit()
        self.review_fetch()
        self.review_fetch()
        self.review_fetch()

        self.login()
        rv = self.app.get("/admin/graph_month")
        assert b"Chart.js" in rv.data, rv.data
        assert b"0, 1" in rv.data, rv.data

        rv = self.app.get("/admin/graph_year")
        assert b"Chart.js" in rv.data, rv.data
        assert b"0, 1" in rv.data, rv.data

        rv = self.app.get("/admin/stats")
        assert b"Chart.js" in rv.data, rv.data
        assert b"Active reviews</td>\n    <td>1</td>" in rv.data, rv.data
        assert b"Haters Gonna Hate" in rv.data, rv.data

    def test_admin_unreport(self):

        self.review_submit()
        self.review_report()

        self.login()
        rv = self.app.get("/admin/unreport/1", follow_redirects=True)
        assert b"Review unreported" in rv.data, rv.data

    def test_admin_review(self):

        rv = self._review_submit(locale="in_IN")
        assert b'"success": true' in rv.data, rv.data

        self.login()
        rv = self.app.get("/admin/review/1")
        assert (
            b"Inkscape has been a essential part of my workflow for many years"
            in rv.data
        ), rv.data
        assert b"Somebody Important" in rv.data, rv.data
        assert b"Fedora" in rv.data, rv.data
        rv = self.app.post(
            "/admin/modify/1",
            data=dict(
                distro="Ubuntu",
            ),
            follow_redirects=True,
        )
        assert (
            b"Inkscape has been a essential part of my workflow for many years"
            in rv.data
        ), rv.data
        assert b"Ubuntu" in rv.data, rv.data

        rv = self.app.get("/admin/englishify/1", follow_redirects=True)
        assert b"en_IN" in rv.data, rv.data

        rv = self.app.get("/admin/anonify/1", follow_redirects=True)
        assert b"Somebody Important" not in rv.data, rv.data

        rv = self.app.get("/admin/vote/1/down", follow_redirects=True)
        assert b"Recorded vote" in rv.data, rv.data

        # delete
        rv = self.app.get("/admin/delete/1", follow_redirects=True)
        assert b"Confirm Removal?" in rv.data, rv.data
        rv = self.app.get("/admin/delete/1/force", follow_redirects=True)
        assert b"Deleted review" in rv.data, rv.data
        rv = self.app.get("/admin/review/1", follow_redirects=True)
        assert b"No review with that ID" in rv.data, rv.data

    def _admin_moderator_add(self, username="dave@dave.com", password="foobarbaz123."):

        return self.app.post(
            "/admin/moderator/add",
            data=dict(
                password_new=password,
                username_new=username,
                display_name="Dave",
            ),
            follow_redirects=True,
        )

    def test_admin_add_moderator(self):

        self.login()

        # bad values
        rv = self._admin_moderator_add(username="1")
        assert b"Username invalid" in rv.data, rv.data
        rv = self._admin_moderator_add(password="foo")
        assert b"The password is too short" in rv.data, rv.data
        rv = self._admin_moderator_add(password="foobarbaz")
        assert b"requires at least one non-alphanumeric" in rv.data, rv.data
        rv = self._admin_moderator_add(username="foo")
        assert b"Invalid email address" in rv.data, rv.data

        # good values
        rv = self._admin_moderator_add()
        assert b"Added user" in rv.data, rv.data

        # duplicate
        rv = self._admin_moderator_add()
        assert b"Already a entry with that username" in rv.data, rv.data
        self.logout()

        # test this actually works
        self.login(username="dave@dave.com", password="foobarbaz123.")
        self.logout()

        # remove
        self.login()
        rv = self.app.get("admin/moderator/2/delete", follow_redirects=True)
        assert b"Deleted user" in rv.data, rv.data

    def test_admin_show_reviews(self):

        self.review_submit()
        self.login()
        rv = self.app.get("/admin/show/all")
        assert b"An essential " in rv.data, rv.data

        rv = self.app.get("/admin/show/lang/en_US")
        assert b"An essential " in rv.data, rv.data
        rv = self.app.get("/admin/show/lang/fr_FR")
        assert b"An essential " not in rv.data, rv.data

    def test_admin_moderators(self):

        self.login()
        rv = self.app.get("/admin/moderators/all")
        assert self.user_hash.encode() in rv.data, rv.data

    def test_admin_search(self):

        self.review_submit()
        self.login()
        rv = self.app.get("/admin/search?value=notgoingtoexist")
        assert b"There are no results for this query" in rv.data, rv.data
        rv = self.app.get("/admin/search?value=inkscape+notgoingtoexist")
        assert b"Somebody Import" in rv.data, rv.data

    def _admin_taboo_add(
        self, locale="en", value="inkscape", description="ola!", severity=0
    ):
        data = {
            "locale": locale,
            "value": value,
            "description": description,
            "severity": severity,
        }
        return self.app.post("/admin/taboo/add", data=data, follow_redirects=True)

    def test_admin_components(self):

        self.review_submit()
        self.review_submit(app_id="inkscape-ubuntu-lts.desktop")
        self.login()
        rv = self.app.get("/admin/component/all")
        assert b"inkscape.desktop" in rv.data, rv.data
        assert b"inkscape-ubuntu-lts.desktop" in rv.data, rv.data

        rv = self.app.get(
            "/admin/component/join/notgoingtoexist.desktop/inkscape-ubuntu-lts.desktop",
            follow_redirects=True,
        )
        assert b"No parent component found" in rv.data, rv.data
        rv = self.app.get(
            "/admin/component/join/inkscape.desktop/notgoingtoexist.desktop",
            follow_redirects=True,
        )
        assert b"No child component found" in rv.data, rv.data
        rv = self.app.get(
            "/admin/component/join/inkscape.desktop/inkscape.desktop",
            follow_redirects=True,
        )
        assert b"Parent and child components were the same" in rv.data, rv.data
        rv = self.app.get(
            "/admin/component/join/inkscape.desktop/inkscape-ubuntu-lts.desktop",
            follow_redirects=True,
        )
        assert b"Joined components" in rv.data, rv.data

        # again
        rv = self.app.get(
            "/admin/component/join/inkscape.desktop/inkscape-ubuntu-lts.desktop",
            follow_redirects=True,
        )
        assert b"Parent and child already set up" in rv.data, rv.data

        # delete inkscape.desktop
        rv = self._api_review_delete()
        assert b"removed review #1" in rv.data, rv.data

        # still match for the alternate name
        self.review_fetch()

    def test_admin_component_delete(self):
        self.review_submit()
        self.review_submit(app_id="inkscape-ubuntu-lts.desktop")
        self.login()

        # delete one, causing the review to get deleted too
        rv = self.app.get("/admin/component/delete/99999", follow_redirects=True)
        assert b"Unable to find component" in rv.data, rv.data
        rv = self.app.get("/admin/component/delete/2", follow_redirects=True)
        assert b"Deleted component with 1 reviews" in rv.data, rv.data

        # still match for the alternate name
        self.review_fetch()

        rv = self.app.get("/admin/component/delete/1", follow_redirects=True)
        assert b"Deleted component with 1 reviews" in rv.data, rv.data

    def test_admin_taboo(self):

        self.login()

        rv = self.app.get("/admin/taboo/all")
        assert b"There are no taboos stored" in rv.data, rv.data

        # add taboos
        rv = self._admin_taboo_add()
        assert b"Added taboo" in rv.data, rv.data
        assert b"inkscape" in rv.data, rv.data
        rv = self._admin_taboo_add()
        assert b"Already added that taboo" in rv.data, rv.data
        rv = self._admin_taboo_add(locale="fr_FR")
        assert b"Added taboo" in rv.data, rv.data

        # submit something, and ensure it's flagged
        rv = self._review_submit()
        assert b"review contains taboo word" in rv.data, rv.data

        # delete
        rv = self.app.get("/admin/taboo/1/delete", follow_redirects=True)
        assert b"Deleted taboo" in rv.data, rv.data
        rv = self.app.get("/admin/taboo/1/delete", follow_redirects=True)
        assert b"No taboo with ID" in rv.data, rv.data

    def test_api_submit_when_banned(self):

        # submit abusive review
        self.review_submit()

        # add user to the ban list
        self.login()
        rv = self.app.get(
            "/admin/user_ban/{}".format(self.user_hash), follow_redirects=True
        )
        assert b"Banned user" in rv.data, rv.data
        assert b"deleted 1 reviews" in rv.data, rv.data
        self.logout()

        # try to submit another review
        rv = self._review_submit(app_id="gimp.desktop")
        assert b"account has been disabled due to abuse" in rv.data, rv.data

    def test_login_logout(self):

        # test logging in and out
        rv = self._login("admin@test.com", "Pa$$w0rd")
        assert b"/admin/show/reported" in rv.data, rv.data
        rv = self._logout()
        rv = self._login("admin@test.com", "Pa$$w0rd")
        assert b"/admin/show/reported" in rv.data, rv.data
        rv = self._logout()
        assert b"/admin/show/reported" not in rv.data, rv.data
        rv = self._login("FAILED@test.com", "default")
        assert b"Incorrect username" in rv.data, rv.data
        rv = self._login("admin@test.com", "defaultx")
        assert b"Incorrect password" in rv.data, rv.data

    @staticmethod
    def run_cron_regenerate_ratings(fn="test.json"):

        from odrs import app
        from cron import _regenerate_ratings

        with app.test_request_context():
            _regenerate_ratings(fn)

    @staticmethod
    def run_cron_auto_delete():

        from odrs import app
        from cron import _auto_delete

        with app.test_request_context():
            _auto_delete(0)

    def test_nologin_required(self):

        # all these are viewable without being logged in
        uris = [
            "/",
            "/privacy",
        ]
        for uri in uris:
            rv = self.app.get(uri, follow_redirects=True)
            assert b"favicon.ico" in rv.data, rv.data
            assert b"Error!" not in rv.data, rv.data

    def _review_submit(
        self,
        app_id=None,
        locale="en_US",
        distro="Fedora",
        version="2:1.2.3~dsg",
        summary=" An essential part of my daily workflow",
        user_hash=None,
        user_skey=None,
        user_display="Somebody Important",
    ):
        if not app_id:
            app_id = "inkscape.desktop"
        if not user_hash:
            user_hash = self.user_hash
        if not user_skey:
            user_skey = _get_user_key(user_hash, app_id)
        # upload a review
        data = {
            "app_id": app_id,
            "locale": locale,
            "summary": summary,
            "description": "Inkscape has been a essential part of my workflow for many years now.",
            "user_hash": user_hash,
            "user_skey": user_skey,
            "user_display": user_display,
            "distro": distro,
            "rating": 100,
            "version": version,
        }
        return self.app.post(
            "/1.0/reviews/api/submit", data=json.dumps(data), follow_redirects=True
        )

    def review_submit(self, app_id=None, user_hash=None):
        rv = self._review_submit(app_id=app_id, user_hash=user_hash)
        assert b'"success": true' in rv.data, rv.data

    def _review_fetch(
        self,
        app_id="inkscape.desktop",
        user_hash=None,
        locale="en_US",
        distro="Fedora",
        compat_ids=None,
        version="1.2.3",
    ):
        if not user_hash:
            user_hash = self.user_hash
        # fetch some reviews
        data = {
            "app_id": app_id,
            "user_hash": user_hash,
            "locale": locale,
            "distro": distro,
            "limit": 5,
            "version": version,
        }
        if compat_ids:
            data["compat_ids"] = compat_ids
        return self.app.post(
            "/1.0/reviews/api/fetch", data=json.dumps(data), follow_redirects=True
        )

    def review_fetch(self):
        rv = self._review_fetch(app_id="inkscape.desktop")
        assert b"An essential part of my daily workflow" in rv.data, rv.data

    def test_api_moderate_locale(self):

        rv = self.app.get("/1.0/reviews/api/moderate/{}/en_GB".format(self.user_hash))
        assert rv.data == b"[]", rv.data
        self.review_submit()
        rv = self.app.get("/1.0/reviews/api/moderate/{}/en_GB".format(self.user_hash))
        assert b"Somebody Important" in rv.data, rv.data
        rv = self.app.get("/1.0/reviews/api/moderate/{}/fr_FR".format(self.user_hash))
        assert rv.data == b"[]", rv.data

    def test_api_fetch_no_results(self):

        # get the skey back for an app with no reviews
        rv = self._review_fetch(app_id="not-going-to-exist.desktop")
        assert b"An essential part of my daily workflow" not in rv.data, rv.data
        assert b"user_skey" in rv.data, rv.data

    def test_api_fetch_compat_id(self):

        self.review_submit()

        # get the reviews back for the app using compat IDs
        rv = self._review_fetch(app_id="foo.desktop", compat_ids=["inkscape.desktop"])
        assert b"An essential part of my daily workflow" in rv.data, rv.data
        assert b"user_skey" in rv.data, rv.data

    def review_upvote(self, user_hash=None):
        if not user_hash:
            user_hash = self.user_hash
        data = {
            "review_id": 1,
            "app_id": "inkscape.desktop",
            "user_hash": user_hash,
            "user_skey": _get_user_key(user_hash, "inkscape.desktop"),
        }
        return self.app.post("/1.0/reviews/api/upvote", data=json.dumps(data))

    def review_report(self, user_hash=None):
        if not user_hash:
            user_hash = self.user_hash
        data = {
            "review_id": 1,
            "app_id": "inkscape.desktop",
            "user_hash": user_hash,
            "user_skey": _get_user_key(user_hash, "inkscape.desktop"),
        }
        return self.app.post("/1.0/reviews/api/report", data=json.dumps(data))

    def test_api_upvote(self):

        # does not exist
        rv = self.review_upvote()
        assert b"invalid review ID" in rv.data, rv.data

        # first upvote
        self.review_submit()
        rv = self.review_upvote()
        assert b'success": true' in rv.data, rv.data
        assert b"voted #1 1" in rv.data, rv.data

        # duplicate upvote
        rv = self.review_upvote()
        assert b'success": false' in rv.data, rv.data
        assert b"already voted on this app" in rv.data, rv.data

        # check vote_id is set
        rv = self._review_fetch(app_id="inkscape.desktop")
        assert b'vote_id": 1' in rv.data, rv.data

        # delete review, hopefully deleting vote too
        rv = self._api_review_delete()
        assert b"removed review #1" in rv.data, rv.data
        self.run_cron_auto_delete()
        rv = self._review_fetch(app_id="inkscape.desktop")
        assert b"vote_id" not in rv.data, rv.data

    def test_api_report(self):

        # submit and verify
        self.review_submit()
        self.review_fetch()

        # should not appear again
        rv = self.review_report()
        assert b'success": true' in rv.data, rv.data
        assert b"voted #1 -5" in rv.data, rv.data
        rv = self.review_report(user_hash="729342d6a7c477bb1ea0186f8c60804a3d783183")
        assert b'success": true' in rv.data, rv.data
        assert b"voted #1 -5" in rv.data, rv.data
        rv = self._review_fetch(app_id="inkscape.desktop")
        assert b"An essential part of my daily workflow" not in rv.data, rv.data

        # duplicate upvote
        rv = self.review_upvote()
        assert b'success": false' in rv.data, rv.data
        assert b"already voted on this app" in rv.data, rv.data

    def test_api_app_rating(self):

        # nothing
        rv = self.app.get("/1.0/reviews/api/ratings/not-going-to-exist.desktop")
        assert rv.data == b"[]", rv.data

        # something
        self.review_submit()
        rv = self.app.get("/1.0/reviews/api/ratings/inkscape.desktop")
        assert b'star1": 0' in rv.data, rv.data
        assert b'star5": 1' in rv.data, rv.data
        assert b'total": 1' in rv.data, rv.data

        # all
        self.review_submit(user_hash="0000000000000000000000000000000000000000")
        rv = self.app.get("/1.0/reviews/api/ratings")
        assert b"inkscape.desktop" in rv.data, rv.data
        assert b'star1": 0' in rv.data, rv.data
        assert b'star5": 2' in rv.data, rv.data
        assert b'total": 2' in rv.data, rv.data

    def _api_review_delete(self):
        data = {
            "review_id": 1,
            "app_id": "inkscape.desktop",
            "user_hash": self.user_hash,
            "user_skey": _get_user_key(self.user_hash, "inkscape.desktop"),
        }
        return self.app.post("/1.0/reviews/api/remove", data=json.dumps(data))

    def test_api_remove(self):

        self.review_submit()

        # wrong app_id
        data = {
            "review_id": 1,
            "app_id": "dave.desktop",
            "user_hash": self.user_hash,
            "user_skey": _get_user_key(self.user_hash, "dave.desktop"),
        }
        rv = self.app.post("/1.0/reviews/api/remove", data=json.dumps(data))
        assert b"the app_id is invalid" in rv.data, rv.data

        # wrong user_hash
        data = {
            "review_id": 1,
            "app_id": "inkscape.desktop",
            "user_hash": _get_user_key(self.user_hash, "inkscape.desktop"),
            "user_skey": _get_user_key(self.user_hash, "inkscape.desktop"),
        }
        rv = self.app.post("/1.0/reviews/api/remove", data=json.dumps(data))
        assert b"no review" in rv.data, rv.data

        # wrong user_skey
        data = {
            "review_id": 1,
            "app_id": "inkscape.desktop",
            "user_hash": self.user_hash,
            "user_skey": self.user_hash,
        }
        rv = self.app.post("/1.0/reviews/api/remove", data=json.dumps(data))
        assert b"invalid user_skey" in rv.data, rv.data

        # delete a review
        rv = self._api_review_delete()
        assert b"removed review #1" in rv.data, rv.data

    def test_api_submit(self):

        # upload a report
        rv = self._review_submit()
        assert b'"success": true' in rv.data, rv.data

        # upload a 2nd report
        rv = self._review_submit(app_id="gimp.desktop")
        assert b'"success": true' in rv.data, rv.data

        # upload a duplicate report
        rv = self._review_submit()
        assert b'success": false' in rv.data, rv.data
        assert b"already reviewed this app" in rv.data, rv.data

        # upload an invalid report
        rv = self._review_submit(summary="<html>foo</html>")
        assert b'success": false' in rv.data, rv.data
        assert b"is not a valid string" in rv.data, rv.data
        rv = self._review_submit(summary="")
        assert b'success": false' in rv.data, rv.data
        assert b"missing data" in rv.data, rv.data

        # get the review back
        rv = self.app.get("/1.0/reviews/api/app/inkscape.desktop")
        assert b"Somebody Important" in rv.data, rv.data
        assert b"An essential part of my daily workflow" in rv.data, rv.data
        assert b"user_skey" not in rv.data, rv.data

        # get the review back with skey
        rv = self.app.get(
            "/1.0/reviews/api/app/inkscape.desktop/{}".format(self.user_hash)
        )
        assert b"An essential part of my daily workflow" in rv.data, rv.data
        assert b"user_skey" in rv.data, rv.data

        # get the reviews back for the app
        rv = self._review_fetch(distro="Ubuntu", version="1.2.4")
        assert b"An essential part of my daily workflow" in rv.data, rv.data
        assert b"user_skey" in rv.data, rv.data

    def test_fail_when_login_required(self):

        # all these are an error when not logged in
        uris = [
            "/admin/graph_month",
            "/admin/graph_year",
            "/admin/stats",
            "/admin/user_ban/1",
            "/admin/show/reported",
            "/admin/stats",
            "/admin/moderators/all",
        ]
        for uri in uris:
            rv = self.app.get(uri, follow_redirects=True)
            assert b"favicon.ico" in rv.data, rv.data
            assert b"Please log in to access this page" in rv.data, (uri, rv.data)

        # POST only
        uris = ["/admin/modify/1"]
        for uri in uris:
            rv = self.app.post(uri, follow_redirects=True)
            assert b"favicon.ico" in rv.data, rv.data
            assert b"Please log in to access this page" in rv.data, rv.data


if __name__ == "__main__":
    unittest.main()
