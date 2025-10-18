#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# pylint: disable=invalid-name,missing-docstring,chained-comparison,singleton-comparison
#
# Copyright (C) 2015-2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+

import datetime
import calendar
from math import ceil

from sqlalchemy import text, or_

from flask import abort, request, flash, render_template, redirect, url_for
from flask_login import login_required, current_user

from odrs import app, db
from .models import Review, User, Moderator, Vote, Taboo, Component
from .models import _vote_exists
from .util import _get_datestr_from_dt, _get_taboos_for_locale


def _get_chart_labels_months():
    """Gets the chart labels"""
    now = datetime.date.today()
    labels = []
    offset = 0
    for i in range(0, 12):
        if now.month - i == 0:
            offset = 1
        labels.append(calendar.month_name[now.month - i - offset])
    return labels


def _get_chart_labels_days():
    """Gets the chart labels"""
    now = datetime.date.today()
    labels = []
    for i in range(0, 30):
        then = now - datetime.timedelta(i)
        labels.append("%02i-%02i-%02i" % (then.year, then.month, then.day))
    return labels


def _get_langs_for_user(user):
    if not user:
        return None
    if not getattr(user, "locales", None):
        return None
    if not user.locales:
        return None
    if user.locales == "*":
        return None
    return user.locales.split(",")


def _get_hash_for_user(user):
    if not user:
        return None
    if not getattr(user, "user_hash", None):
        return None
    return user.user_hash


def _password_check(value):
    """Check the password for suitability"""
    success = True
    if len(value) < 8:
        success = False
        flash("The password is too short, the minimum is 8 characters", "warning")
    if value.isalnum():
        success = False
        flash(
            "The password requires at least one non-alphanumeric character", "warning"
        )
    return success


def _email_check(value):
    """Do a quick and dirty check on the email address"""
    if len(value) < 5 or value.find("@") == -1 or value.find(".") == -1:
        return False
    return True


class Pagination:
    def __init__(self, page, per_page, total_count):
        self.page = page
        self.per_page = per_page
        self.total_count = total_count

    @property
    def pages(self):
        return int(ceil(self.total_count / float(self.per_page)))

    @property
    def has_prev(self):
        return self.page > 1

    @property
    def has_next(self):
        return self.page < self.pages

    def iter_pages(self, left_edge=2, left_current=2, right_current=5, right_edge=2):
        last = 0
        for num in range(1, self.pages + 1):
            if (
                num <= left_edge
                or (
                    num > self.page - left_current - 1
                    and num < self.page + right_current
                )
                or num > self.pages - right_edge
            ):
                if last + 1 != num:
                    yield None
                yield num
                last = num


def _get_analytics_by_interval(size, interval):
    """Gets analytics data"""
    array = []
    now = datetime.date.today()

    # yes, there's probably a way to do this in one query
    for i in range(size):
        start = _get_datestr_from_dt(
            now - datetime.timedelta((i * interval) + interval - 1)
        )
        end = _get_datestr_from_dt(now - datetime.timedelta((i * interval) - 1))
        stmt = text(
            "SELECT fetch_cnt FROM analytics WHERE datestr BETWEEN :start AND :end"
        )
        res = db.session.execute(  # pylint: disable=no-member
            stmt.bindparams(start=start, end=end)
        )

        # add all these up
        tmp = 0
        for r in res:
            tmp = tmp + r[0]
        array.append(tmp)
    return array


def _get_stats_by_interval(size, interval, msg):
    """Gets stats data"""
    cnt = []
    now = datetime.date.today()

    # yes, there's probably a way to do this in one query
    for i in range(size):
        start = now - datetime.timedelta((i * interval) + interval - 1)
        end = now - datetime.timedelta((i * interval) - 1)
        stmt = text(
            "SELECT COUNT(*) FROM eventlog "
            "WHERE message = :msg AND date_created BETWEEN :start AND :end"
        )
        res = db.session.execute(  # pylint: disable=no-member
            stmt.bindparams(start=start, end=end, msg=msg)
        )
        cnt.append(res.fetchone()[0])
    return cnt


@app.route("/admin/graph_month")
@login_required
def admin_graph_month():
    """
    Show nice graph graphs.
    """
    data_fetch = _get_analytics_by_interval(30, 1)
    data_review = _get_stats_by_interval(30, 1, "reviewed")
    return render_template(
        "graph-month.html",
        labels=_get_chart_labels_days()[::-1],
        data_requests=data_fetch[::-1],
        data_submitted=data_review[::-1],
    )


@app.route("/admin/graph_year")
@login_required
def admin_graph_year():
    """
    Show nice graph graphs.
    """
    data_fetch = _get_analytics_by_interval(12, 30)
    data_review = _get_stats_by_interval(12, 30, "reviewed")
    return render_template(
        "graph-year.html",
        labels=_get_chart_labels_months()[::-1],
        data_requests=data_fetch[::-1],
        data_submitted=data_review[::-1],
    )


@app.route("/admin/stats")
@login_required
def admin_show_stats():
    """
    Return the statistics page as HTML.
    """
    # security check
    if not current_user.is_admin:
        flash("Unable to show stats as non-admin", "error")
        return redirect(url_for(".odrs_index"))

    stats = {}

    # get the total number of reviews
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(*) FROM reviews;"
    )
    stats["Active reviews"] = rs.fetchone()[0]

    # unique reviewers
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(DISTINCT(user_id)) FROM reviews;"
    )
    stats["Unique reviewers"] = rs.fetchone()[0]

    # total votes
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(*) FROM votes WHERE val = 1;"
    )
    stats["User upvotes"] = rs.fetchone()[0]
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(*) FROM votes WHERE val = -1;"
    )
    stats["User downvotes"] = rs.fetchone()[0]

    # unique voters
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(DISTINCT(user_id)) FROM votes;"
    )
    stats["Unique voters"] = rs.fetchone()[0]

    # unique languages
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(DISTINCT(locale)) FROM reviews;"
    )
    stats["Unique languages"] = rs.fetchone()[0]

    # unique distros
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(DISTINCT(distro)) FROM reviews;"
    )
    stats["Unique distros"] = rs.fetchone()[0]

    # unique apps
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(*) FROM components;"
    )
    stats["Unique apps reviewed"] = rs.fetchone()[0]

    # unique distros
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT COUNT(*) FROM reviews WHERE reported > 0;"
    )
    stats["Reported reviews"] = rs.fetchone()[0]

    # star reviews
    for star in range(1, 6):
        rs = db.session.execute(  # pylint: disable=no-member
            "SELECT COUNT(*) FROM reviews WHERE rating = {};".format(star * 20)
        )
        stats["%i star reviews" % star] = rs.fetchone()[0]

    # popularity view
    viewed = (
        db.session.query(Component.app_id, Component.fetch_cnt)
        .filter(Component.app_id != None)
        .order_by(Component.fetch_cnt.desc())
        .limit(50)
        .all()
    )

    # popularity reviews
    submitted = (
        db.session.query(Component.app_id, Component.review_cnt)
        .filter(Component.app_id != None)
        .order_by(Component.review_cnt.desc())
        .limit(50)
        .all()
    )

    # users
    users_awesome = (
        db.session.query(User)
        .filter(User.karma != 0)
        .order_by(User.karma.desc())
        .limit(10)
        .all()
    )
    users_haters = (
        db.session.query(User)
        .filter(User.karma != 0)
        .order_by(User.karma.asc())
        .limit(10)
        .all()
    )

    # distros
    rs = db.session.execute(  # pylint: disable=no-member
        "SELECT DISTINCT(distro), COUNT(distro) AS total "
        "FROM reviews GROUP BY distro ORDER BY total DESC "
        "LIMIT 8;"
    )
    labels = []
    data = []
    for s in rs:
        name = s[0]
        for suffix in [" Linux", " GNU/Linux", " OS", " Linux"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        labels.append(name)
        data.append(s[1])

    return render_template(
        "stats.html",
        users_awesome=users_awesome,
        users_haters=users_haters,
        results_stats=stats,
        results_viewed=viewed,
        results_submitted=submitted,
        labels=labels,
        data=data,
    )


@app.route("/admin/review/<int:review_id>")
@login_required
def admin_show_review(review_id):
    """
    Show a specific review as HTML.
    """
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))

    # has the user already voted
    if current_user.user:
        vote = _vote_exists(review_id, current_user.user.user_id)
    else:
        vote = None

    # does the review contain any banned keywords
    matched_taboos = review.matches_taboos(_get_taboos_for_locale(review.locale))
    return render_template(
        "show.html", r=review, vote_exists=vote, matched_taboos=matched_taboos
    )


@app.route("/admin/modify/<int:review_id>", methods=["POST"])
@login_required
def admin_modify(review_id):
    """Change details about a review"""
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))
    if "distro" in request.form:
        review.distro = request.form["distro"]
    if "locale" in request.form:
        review.locale = request.form["locale"]
    if "user_display" in request.form:
        if len(request.form["user_display"]) == 0:
            review.user_display = None
        else:
            review.user_display = request.form["user_display"]
    if "description" in request.form:
        review.description = request.form["description"]
    if "summary" in request.form:
        review.summary = request.form["summary"]
    if "version" in request.form:
        review.version = request.form["version"]
    db.session.commit()
    return redirect(url_for(".admin_show_review", review_id=review_id))


@app.route("/admin/user_ban/<user_hash>")
@login_required
def admin_user_ban(user_hash):
    """Change details about a review"""
    # security check
    if not current_user.is_admin:
        flash("Unable to ban user as non-admin", "error")
        return redirect(url_for(".odrs_index"))
    user = db.session.query(User).filter(User.user_hash == user_hash).first()
    if not user:
        flash("No user with that user_hash")
        return redirect(url_for(".odrs_index"))
    user.is_banned = True

    # delete any of the users reviews
    nr_delete = len(user.reviews)
    for review in user.reviews:
        db.session.delete(review)
    db.session.commit()
    flash("Banned user and deleted {} reviews".format(nr_delete))
    return redirect(url_for(".odrs_show_reported"))


@app.route("/admin/unreport/<int:review_id>")
@login_required
def admin_unreport(review_id):
    """Unreport a perfectly valid review"""
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))
    review.reported = 0
    db.session.commit()
    flash("Review unreported")
    return redirect(url_for(".admin_show_review", review_id=review_id))


@app.route("/admin/unremove/<int:review_id>")
@login_required
def admin_unremove(review_id):
    """Unreport a perfectly valid review"""
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))
    review.date_deleted = None
    db.session.commit()
    flash("Review unremoved")
    return redirect(url_for(".admin_show_review", review_id=review_id))


@app.route("/admin/englishify/<int:review_id>")
@login_required
def admin_englishify(review_id):
    """Marks a review as writen in English"""
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))
    parts = review.locale.split("_")
    if len(parts) == 1:
        review.locale = "en"
    else:
        review.locale = "en_" + parts[1]
    db.session.commit()
    return redirect(url_for(".admin_show_review", review_id=review_id))


@app.route("/admin/anonify/<int:review_id>")
@login_required
def admin_anonify(review_id):
    """Removes the username from the review"""
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))
    review.user_display = None
    db.session.commit()
    return redirect(url_for(".admin_show_review", review_id=review_id))


@app.route("/admin/delete/<review_id>/force")
@login_required
def admin_delete_force(review_id):
    """Delete a review"""
    review = db.session.query(Review).filter(Review.review_id == review_id).first()
    if not review:
        flash("No review with that ID")
        return redirect(url_for(".odrs_index"))
    db.session.delete(review)
    db.session.commit()
    flash("Deleted review")
    return redirect(url_for(".odrs_show_reported"))


@app.route("/admin/delete/<review_id>")
@login_required
def admin_delete(review_id):
    """Ask for confirmation to delete a review"""
    return render_template("delete.html", review_id=review_id)


@app.route("/admin/show/all")
@app.route("/admin/show/all/page/<int:page>")
@login_required
def admin_show_all(page=1):
    """
    Return all the reviews on the server as HTML.
    """

    reviews = db.session.query(Review).order_by(Review.date_created.desc()).all()
    if not reviews and page != 1:
        abort(404)

    reviews_per_page = 20
    pagination = Pagination(page, reviews_per_page, len(reviews))
    reviews = reviews[(page - 1) * reviews_per_page : page * reviews_per_page]
    return render_template("show-all.html", pagination=pagination, reviews=reviews)


def _review_filter_keys(keys):
    cond = []
    for key in keys:
        cond.append(Review.user_display.like("%{}%".format(key)))
        cond.append(Review.summary.like("%{}%".format(key)))
        cond.append(Review.description.like("%{}%".format(key)))
    return or_(*cond)


@app.route("/admin/search")
@app.route("/admin/search/<int:max_results>")
def admin_search(max_results=19):

    # no search results
    if "value" not in request.args:
        return render_template("search.html")

    keys = request.args["value"].split(" ")
    reviews = (
        db.session.query(Review)
        .filter(_review_filter_keys(keys))
        .order_by(Review.date_created.desc())
        .limit(max_results)
        .all()
    )
    return render_template("show-all.html", reviews=reviews)


@app.route("/admin/show/reported")
@app.route("/admin/show/reported/<int:limit>")
@login_required
def odrs_show_reported(limit=1):
    """
    Return all the reported reviews on the server as HTML.
    """
    reviews = (
        db.session.query(Review)
        .filter(Review.reported >= limit)
        .order_by(Review.date_created.desc())
        .all()
    )
    return render_template("show-all.html", reviews=reviews)


@app.route("/admin/show/user/<user_hash>")
@login_required
def admin_show_user(user_hash):
    """
    Return all the reviews from a user on the server as HTML.
    """
    user = db.session.query(User).filter(User.user_hash == user_hash).first()
    if not user:
        flash("No user with that user_hash")
        return redirect(url_for(".admin_show_all"))
    reviews = db.session.query(Review).filter(Review.user_id == user.user_id).all()
    return render_template("show-all.html", reviews=reviews)


@app.route("/admin/show/app/<app_id>")
@login_required
def admin_show_app(app_id):
    """
    Return all the reviews from a user on the server as HTML.
    """
    reviews = (
        db.session.query(Review)
        .join(Component)
        .filter(Component.app_id == app_id)
        .all()
    )
    return render_template("show-all.html", reviews=reviews)


@app.route("/admin/show/lang/<locale>")
@login_required
def admin_show_lang(locale):
    """
    Return all the reviews from a user on the server as HTML.
    """
    reviews = db.session.query(Review).filter(Review.locale == locale).all()
    return render_template("show-all.html", reviews=reviews)


@app.route("/admin/moderators/all")
@login_required
def admin_moderator_show_all():
    """
    Return all the moderators as HTML.
    """
    # security check
    if not current_user.is_admin:
        flash("Unable to show all moderators", "error")
        return redirect(url_for(".odrs_index"))
    mods = db.session.query(Moderator).all()
    return render_template("mods.html", mods=mods)


@app.route("/admin/moderator/add", methods=["GET", "POST"])
@login_required
def admin_moderator_add():
    """Add a moderator [ADMIN ONLY]"""

    # only accept form data
    if request.method != "POST":
        return redirect(url_for(".profile"))

    # security check
    if not current_user.is_admin:
        flash("Unable to add moderator as non-admin", "error")
        return redirect(url_for(".odrs_index"))

    for key in ["username_new", "password_new", "display_name"]:
        if not key in request.form:
            flash("Unable to add moderator as {} missing".format(key), "error")
            return redirect(url_for(".odrs_index"))
    if (
        db.session.query(Moderator)
        .filter(Moderator.username == request.form["username_new"])
        .first()
    ):
        flash("Already a entry with that username", "warning")
        return redirect(url_for(".admin_moderator_show_all"))

    # verify password
    password = request.form["password_new"]
    if not _password_check(password):
        return redirect(url_for(".admin_moderator_show_all"))

    # verify username
    username_new = request.form["username_new"]
    if len(username_new) < 3:
        flash("Username invalid", "warning")
        return redirect(url_for(".admin_moderator_show_all"))
    if not _email_check(username_new):
        flash("Invalid email address", "warning")
        return redirect(url_for(".admin_moderator_show_all"))

    # verify name
    display_name = request.form["display_name"]
    if len(display_name) < 3:
        flash("Name invalid", "warning")
        return redirect(url_for(".admin_moderator_show_all"))

    # verify username
    db.session.add(Moderator(username_new, password, display_name))
    db.session.commit()
    flash("Added user")
    return redirect(url_for(".admin_moderator_show_all"))


@app.route("/admin/moderator/<int:moderator_id>/admin")
@login_required
def odrs_moderator_show(moderator_id):
    """
    Shows an admin panel for a moderator
    """
    if moderator_id != current_user.moderator_id and not current_user.is_admin:
        flash("Unable to show moderator information", "error")
        return redirect(url_for(".odrs_index"))
    mod = (
        db.session.query(Moderator)
        .filter(Moderator.moderator_id == moderator_id)
        .first()
    )
    if not mod:
        flash("No entry with moderator ID {}".format(moderator_id), "warning")
        return redirect(url_for(".admin_moderator_show_all"))
    return render_template("modadmin.html", u=mod)


@app.route("/admin/moderator/<int:moderator_id>/delete")
@login_required
def admin_moderate_delete(moderator_id):
    """Delete a moderator"""

    # security check
    if not current_user.is_admin:
        flash("Unable to delete moderator as not admin", "error")
        return redirect(url_for(".odrs_index"))

    # check whether exists in database
    mod = (
        db.session.query(Moderator)
        .filter(Moderator.moderator_id == moderator_id)
        .first()
    )
    if not mod:
        flash("No entry with moderator ID {}".format(moderator_id), "warning")
        return redirect(url_for(".admin_moderator_show_all"))
    db.session.delete(mod)
    db.session.commit()
    flash("Deleted user")
    return redirect(url_for(".admin_moderator_show_all"))


@app.route("/admin/taboo/all")
@login_required
def admin_taboo_show_all():
    """
    Return all the taboos.
    """
    # security check
    if not current_user.is_admin:
        flash("Unable to show all taboos", "error")
        return redirect(url_for(".odrs_index"))
    taboos = (
        db.session.query(Taboo)
        .order_by(Taboo.locale.asc())
        .order_by(Taboo.value.asc())
        .all()
    )
    return render_template("taboos.html", taboos=taboos)


@app.route("/admin/taboo/add", methods=["GET", "POST"])
@login_required
def admin_taboo_add():
    """Add a taboo [ADMIN ONLY]"""

    # only accept form data
    if request.method != "POST":
        return redirect(url_for(".admin_taboo_show_all"))

    # security check
    if not current_user.is_admin:
        flash("Unable to add taboo as non-admin", "error")
        return redirect(url_for(".odrs_index"))

    for key in ["locale", "value", "description", "severity"]:
        if not key in request.form:
            flash("Unable to add taboo as {} missing".format(key), "error")
            return redirect(url_for(".odrs_index"))
    if (
        db.session.query(Taboo)
        .filter(Taboo.locale == request.form["locale"])
        .filter(Taboo.value == request.form["value"])
        .first()
    ):
        flash("Already added that taboo", "warning")
        return redirect(url_for(".admin_taboo_show_all"))

    # verify username
    db.session.add(
        Taboo(
            request.form["locale"],
            request.form["value"],
            request.form["description"],
            int(request.form["severity"]),
        )
    )
    db.session.commit()
    flash("Added taboo")
    return redirect(url_for(".admin_taboo_show_all"))


@app.route("/admin/taboo/<int:taboo_id>/delete")
@login_required
def admin_taboo_delete(taboo_id):
    """Delete an taboo"""

    # security check
    if not current_user.is_admin:
        flash("Unable to delete taboo as not admin", "error")
        return redirect(url_for(".odrs_index"))

    # check whether exists in database
    taboo = db.session.query(Taboo).filter(Taboo.taboo_id == taboo_id).first()
    if not taboo:
        flash("No taboo with ID {}".format(taboo_id), "warning")
        return redirect(url_for(".admin_taboo_show_all"))
    db.session.delete(taboo)
    db.session.commit()
    flash("Deleted taboo")
    return redirect(url_for(".admin_taboo_show_all"))


@app.route("/admin/component/all")
@login_required
def admin_component_show_all():
    """
    Return all the components.
    """
    # security check
    if not current_user.is_admin:
        flash("Unable to show all components", "error")
        return redirect(url_for(".odrs_index"))
    components = (
        db.session.query(Component)
        .order_by(Component.app_id.asc())
        .order_by(Component.review_cnt.asc())
        .all()
    )
    return render_template("components.html", components=components)


@app.route("/admin/component/join/<component_id_parent>/<component_id_child>")
@login_required
def admin_component_join(component_id_parent, component_id_child):
    """
    Join components.
    """
    # security check
    if not current_user.is_admin:
        flash("Unable to join components", "error")
        return redirect(url_for(".odrs_index"))
    parent = (
        db.session.query(Component)
        .filter(Component.app_id == component_id_parent)
        .first()
    )
    if not parent:
        flash("No parent component found", "warning")
        return redirect(url_for(".admin_component_show_all"))
    child = (
        db.session.query(Component)
        .filter(Component.app_id == component_id_child)
        .first()
    )
    if not child:
        flash("No child component found", "warning")
        return redirect(url_for(".admin_component_show_all"))
    if parent.component_id == child.component_id:
        flash("Parent and child components were the same", "warning")
        return redirect(url_for(".admin_component_show_all"))
    if parent.component_id == child.component_id_parent:
        flash("Parent and child already set up", "warning")
        return redirect(url_for(".admin_component_show_all"))

    # return best message
    adopted = parent.adopt(child)
    db.session.commit()
    if adopted:
        flash(
            "Joined components, adopting {} additional components".format(adopted),
            "info",
        )
    else:
        flash("Joined components", "info")
    return redirect(url_for(".admin_component_show_all"))


@app.route("/admin/component/join", methods=["POST"])
@login_required
def admin_component_join2():
    """Change details about the any user"""

    # security check
    if not current_user.is_admin:
        flash("Unable to join components", "error")
        return redirect(url_for(".odrs_index"))

    # set each thing in turn
    parent = None
    children = []
    for key in request.form:
        if key == "parent":
            parent = (
                db.session.query(Component)
                .filter(Component.app_id == request.form[key])
                .first()
            )
        if key == "child":
            for component_id in request.form.getlist(key):
                child = (
                    db.session.query(Component)
                    .filter(Component.app_id == component_id)
                    .first()
                )
                if child:
                    children.append(child)
    if not parent:
        flash("No parent component found", "warning")
        return redirect(url_for(".admin_component_show_all"))
    if not children:
        flash("No child components found", "warning")
        return redirect(url_for(".admin_component_show_all"))

    # adopt each child
    adopted = 0
    for child in children:
        if parent.component_id == child.component_id:
            child.component_id_parent = None
            continue
        adopted += parent.adopt(child)
    db.session.commit()
    if adopted:
        flash(
            "Joined {} components, "
            "adopting {} additional components".format(len(children), adopted),
            "info",
        )
    else:
        flash("Joined {} components".format(len(children)), "info")
    return redirect(url_for(".admin_component_show_all"))


@app.route("/admin/component/delete/<int:component_id>")
@login_required
def admin_component_delete(component_id):
    """
    Delete component, and any reviews.
    """
    if not current_user.is_admin:
        flash("Unable to delete component", "error")
        return redirect(url_for(".odrs_index"))
    component = (
        db.session.query(Component)
        .filter(Component.component_id == component_id)
        .first()
    )
    if not component:
        flash("Unable to find component", "error")
        return redirect(url_for(".admin_component_show_all"))

    flash("Deleted component with {} reviews".format(len(component.reviews)), "info")
    db.session.delete(component)
    db.session.commit()
    return redirect(url_for(".admin_component_show_all"))


@app.route("/admin/vote/<int:review_id>/<val_str>")
@login_required
def admin_vote(review_id, val_str):
    """Up or downvote an existing review by @val karma points"""
    if not current_user.user:
        flash("No user for moderator")
        return redirect(url_for(".admin_show_review", review_id=review_id))
    if val_str == "up":
        val = 1
    elif val_str == "down":
        val = -1
    elif val_str == "meh":
        val = 0
    else:
        flash("Invalid vote value")
        return redirect(url_for(".admin_show_review", review_id=review_id))

    # the user already has a review
    if _vote_exists(review_id, current_user.user_id):
        flash("already voted on this app")
        return redirect(url_for(".admin_show_review", review_id=review_id))

    current_user.user.karma += val
    db.session.add(Vote(current_user.user_id, val, review_id=review_id))
    db.session.commit()
    flash("Recorded vote")
    return redirect(url_for(".admin_show_review", review_id=review_id))


@app.route("/admin/moderator/<int:moderator_id>/modify_by_admin", methods=["POST"])
@login_required
def admin_user_modify_by_admin(moderator_id):
    """Change details about the any user"""

    # security check
    if moderator_id != current_user.moderator_id and not current_user.is_admin:
        flash("Unable to modify user as non-admin", "error")
        return redirect(url_for(".odrs_index"))

    mod = (
        db.session.query(Moderator)
        .filter(Moderator.moderator_id == moderator_id)
        .first()
    )
    if not mod:
        flash("moderator_id invalid", "warning")
        return redirect(url_for(".admin_moderator_show_all"))

    # set each thing in turn
    mod.is_enabled = "is_enabled" in request.form
    mod.is_admin = "is_admin" in request.form
    for key in ["display_name", "password", "user_hash", "locales"]:
        # unchecked checkbuttons are not included in the form data
        if key not in request.form:
            continue

        val = request.form[key]
        # don't set the optional password
        if key == "password" and len(val) == 0:
            continue
        setattr(mod, key, val)
    db.session.commit()
    flash("Updated profile")
    return redirect(url_for(".odrs_moderator_show", moderator_id=moderator_id))
