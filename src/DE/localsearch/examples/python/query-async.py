#!/usr/bin/env python3
# Copyright (C) 2020, Sam Thursfield <sam@afuera.me.uk>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

"""Example script to query all documents from the filesystem index.

This uses an asynchronous query, which runs the query in a new thread and
notifies the main loop when results are ready.

"""

import gi
gi.require_version('Tracker', '3.0')
from gi.repository import Tracker, GLib


def results_ready_cb(obj, result, user_data):
    cursor = obj.query_finish(result)
    loop = user_data

    # This can also be done asynchronously using next_async().
    print("Documents:")
    while (cursor.next(None)):
        location = cursor.get_string(0)[0]
        title = cursor.get_string(1)[0]
        print(f"  * {location} (title: {title})")
    loop.quit ()


if __name__ == "__main__":
    loop = GLib.MainLoop ()
    cancellable = None

    miner_fs = Tracker.SparqlConnection.bus_new(
        "org.freedesktop.LocalSearch3", None, cancellable)
    cursor = miner_fs.query_async(
        "SELECT nie:isStoredAs(?doc) ?title "
        "FROM tracker:Documents "
        "WHERE { ?doc a nfo:Document ; nie:title ?title . }",
        cancellable,
        results_ready_cb,
        loop)

    loop.run ()
