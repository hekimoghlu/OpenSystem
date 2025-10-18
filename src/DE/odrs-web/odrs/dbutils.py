#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2019 Richard Hughes <richard@hughsie.com>
#
# pylint: disable=import-outside-toplevel
#
# SPDX-License-Identifier: GPL-3.0+


def init_db(db):

    # ensure all tables exist
    db.metadata.create_all(bind=db.engine)

    # ensure admin user exists
    from .models import Moderator, User

    user = (
        db.session.query(User)
        .filter(User.user_hash == "deadbeef348c0f88529f3bfd937ec1a5d90aefc7")
        .first()
    )
    if not user:
        user = User("deadbeef348c0f88529f3bfd937ec1a5d90aefc7")
        db.session.add(user)
        db.session.commit()
    if (
        not db.session.query(Moderator)
        .filter(Moderator.username == "admin@test.com")
        .first()
    ):
        mod = Moderator(username="admin@test.com")
        mod.password = "Pa$$w0rd"
        mod.is_admin = True
        mod.user_id = user.user_id
        db.session.add(mod)
        db.session.commit()


def drop_db(db):
    db.metadata.drop_all(bind=db.engine)
