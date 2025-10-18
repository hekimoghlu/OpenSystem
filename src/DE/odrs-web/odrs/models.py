#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# pylint: disable=invalid-name,missing-docstring,too-few-public-methods,too-many-instance-attributes
#
# Copyright (C) 2015-2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+

import datetime
import re

from werkzeug.security import generate_password_hash, check_password_hash

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Index,
    ForeignKey,
    CheckConstraint,
)
from sqlalchemy.orm import relationship

from odrs import db

from .util import _password_hash, _get_user_key, _addr_hash


def _vote_exists(review_id, user_id):
    """Checks to see if a vote exists for the review+user"""
    return (
        db.session.query(Vote)
        .filter(Vote.review_id == review_id)
        .filter(Vote.user_id == user_id)
        .first()
    )


class Analytic(db.Model):

    # sqlalchemy metadata
    __tablename__ = "analytics"
    __table_args__ = (
        Index("datestr", "datestr", "app_id", unique=True),
        {"mysql_character_set": "utf8mb4"},
    )

    datestr = Column(Integer, default=0, primary_key=True)
    app_id = Column(String(128), primary_key=True)
    fetch_cnt = Column(Integer, default=1)

    def __init__(self):
        self.datestr = None

    def __repr__(self):
        return "Analytic object %s" % self.analytic_id


class Taboo(db.Model):

    # sqlalchemy metadata
    __tablename__ = "taboos"
    __table_args__ = {"mysql_character_set": "utf8mb4"}

    taboo_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    locale = Column(String(8), nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(Integer, default=0)

    def __init__(self, locale, value, description=None, severity=0):
        self.locale = locale
        self.value = value
        self.description = description
        self.severity = severity

    def asdict(self):
        item = {"value": self.value}
        if self.severity:
            item["severity"] = self.severity
        if self.description:
            item["description"] = self.description
        return item

    @property
    def color(self):
        if self.severity == 3:
            return "danger"
        if self.severity == 2:
            return "warning"
        return "info"

    def __repr__(self):
        return "Taboo object %s" % self.taboo_id


class Vote(db.Model):

    # sqlalchemy metadata
    __tablename__ = "votes"
    __table_args__ = {"mysql_character_set": "utf8mb4"}

    vote_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    date_created = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    review_id = Column(Integer, ForeignKey("reviews.review_id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    val = Column(Integer, default=0)

    user = relationship("User")
    review = relationship("Review")

    def __init__(self, user_id, val, review_id=0):
        self.review_id = review_id
        self.user_id = user_id
        self.val = val

    def __repr__(self):
        return "Vote object %s" % self.vote_id


class User(db.Model):

    # sqlalchemy metadata
    __tablename__ = "users"
    __table_args__ = {"mysql_character_set": "utf8mb4"}
    __table_args__ = (
        Index("users_hash_idx", "user_hash"),
        {"mysql_character_set": "utf8mb4"},
    )

    user_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    date_created = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    user_hash = Column(String(40))
    karma = Column(Integer, default=0)
    is_banned = Column(Boolean, default=False)

    reviews = relationship("Review", back_populates="user", cascade="all,delete-orphan")

    def __init__(self, user_hash=None):
        self.user_hash = user_hash
        self.karma = 0
        self.is_banned = False

    def __repr__(self):
        return "User object %s" % self.user_id


def _tokenize(val):
    return [token.lower() for token in re.findall(r"[\w']+", val)]


class Component(db.Model):

    # sqlalchemy metadata
    __tablename__ = "components"
    __table_args__ = {"mysql_character_set": "utf8mb4"}

    component_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    component_id_parent = Column(Integer, ForeignKey("components.component_id"))
    app_id = Column(Text)
    fetch_cnt = Column(Integer, default=0)
    review_cnt = Column(Integer, default=1)

    reviews = relationship(
        "Review", back_populates="component", cascade="all,delete-orphan"
    )
    parent = relationship(
        "Component",
        uselist=False,
        remote_side="Component.component_id",
        backref="children",
        lazy="joined",
    )

    def __init__(self, app_id):
        self.app_id = app_id
        self.fetch_cnt = 0
        self.review_cnt = 1

    def adopt(self, child):

        # adopt any of the childs existing children
        adopted = 0
        for component in child.children:
            component.component_id_parent = self.component_id
            adopted += 1

        # set the child parent
        child.component_id_parent = self.component_id

        return adopted

    @property
    def app_ids(self):
        app_ids = [self.app_id]
        if self.parent:
            if self.parent.app_id not in app_ids:
                app_ids.append(self.parent.app_id)
        for child in self.children:
            if child.app_id not in app_ids:
                app_ids.append(child.app_id)
        return app_ids

    def __repr__(self):
        return "Component object %s" % self.component_id


class Review(db.Model):

    # sqlalchemy metadata
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint("rating >=0 and rating <= 100", name="rating_constraint"),
        {"mysql_character_set": "utf8mb4"}
    )

    review_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    date_created = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    date_deleted = Column(DateTime)
    component_id = Column(
        Integer, ForeignKey("components.component_id"), nullable=False
    )
    locale = Column(Text)
    summary = Column(Text)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    user_addr_hash = Column("user_addr", Text)
    user_display = Column(Text)
    version = Column(Text)
    distro = Column(Text)
    rating = Column(Integer, default=0)
    karma_up = Column(Integer, default=0)
    karma_down = Column(Integer, default=0)
    reported = Column(Integer, default=0, index=True)

    user = relationship("User", back_populates="reviews")
    component = relationship(
        "Component",  # the one used for submit()
        back_populates="reviews",
        lazy="joined",
    )
    votes = relationship("Vote", back_populates="review", cascade="all,delete-orphan")

    def __init__(self):
        self.locale = None
        self.summary = None
        self.description = None
        self.version = None
        self.distro = None
        self.karma_up = 0
        self.karma_down = 0
        self.user_id = 0
        self.user_display = None
        self.rating = 0
        self.reported = 0

    @property
    def component_parent(self):
        if self.component.parent:
            return self.component.parent
        return self.component

    def _generate_keywords(self):

        # tokenize anything the user can specify
        tokens = []
        if self.summary:
            tokens.extend(_tokenize(self.summary))
        if self.description:
            tokens.extend(_tokenize(self.description))
        if self.user_display:
            tokens.extend(_tokenize(self.user_display))

        # dedupe, and remove anything invalid
        tokens = set(tokens)
        if None in tokens:
            tokens.remove(None)
        return tokens

    def matches_taboos(self, taboos):

        # does the review contain any banned keywords
        kws = self._generate_keywords()
        matches = []
        for taboo in taboos:
            if taboo.value in kws:
                matches.append(taboo)
        return matches

    @property
    def user_addr(self):
        raise AttributeError("user_addr is not a readable attribute")

    @user_addr.setter
    def user_addr(self, user_addr):
        self.user_addr_hash = _addr_hash(user_addr)

    def asdict(self, user_hash=None):
        item = {
            "app_id": self.component.app_id,
            "date_created": self.date_created.timestamp(),
            "description": self.description,
            "distro": self.distro,
            "karma_down": self.karma_down,
            "karma_up": self.karma_up,
            "locale": self.locale,
            "rating": self.rating,
            "reported": self.reported,
            "review_id": self.review_id,
            "summary": self.summary,
            "user_display": self.user_display,
            "version": self.version,
        }
        if self.user:
            item["user_hash"] = self.user.user_hash
        if user_hash:
            item["user_skey"] = _get_user_key(user_hash, self.component.app_id)
        return item

    def __repr__(self):
        return "Review object %s" % self.review_id


class Event(db.Model):

    # sqlalchemy metadata
    __tablename__ = "eventlog"
    __table_args__ = (
        Index("message_idx", "message", mysql_length=8),
        Index("date_created_idx", "date_created"),
        {"mysql_character_set": "utf8mb4"},
    )

    eventlog_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    date_created = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    user_addr = Column(Text)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    message = Column(Text)
    app_id = Column(Text)
    important = Column(Boolean, default=False)

    user = relationship("User")

    def __init__(
        self, user_addr, user_id=None, app_id=None, message=None, important=False
    ):
        self.user_addr = user_addr
        self.user_id = user_id
        self.message = message
        self.app_id = app_id
        self.important = important

    def __repr__(self):
        return "Event object %s" % self.eventlog_id


class Moderator(db.Model):

    # sqlalchemy metadata
    __tablename__ = "moderators"
    __table_args__ = {"mysql_character_set": "utf8mb4"}

    moderator_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    username = Column(Text)
    password_hash = Column("password", Text)
    display_name = Column(Text)
    is_enabled = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    locales = Column(Text)

    user = relationship("User")

    def __init__(self, username=None, password=None, display_name=None):
        self.username = username
        self.display_name = display_name
        self.is_enabled = True
        self.is_admin = False
        self.locales = None
        if password:
            self.password = password

    @property
    def password(self):
        raise AttributeError("password is not a readable attribute")

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        if not self.password_hash:
            return False
        # on success, upgrade the old hashing function to the new secure one
        if len(self.password_hash) == 40:
            if self.password_hash != _password_hash(password):
                return False
            self.password = password
            return True
        return check_password_hash(self.password_hash, password)

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.moderator_id)

    def __repr__(self):
        return "Moderator object %s" % self.moderator_id
