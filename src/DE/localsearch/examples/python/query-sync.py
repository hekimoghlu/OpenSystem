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

"""Example script to query all music albums from the filesystem index.

This uses a synchronous query, which blocks until results are ready. This can
be useful in commandline applications, automated tests, and multithreaded
apps.

"""

import gi
gi.require_version('Tracker', '3.0')
from gi.repository import Tracker

cancellable = None

miner_fs = Tracker.SparqlConnection.bus_new(
    "org.freedesktop.LocalSearch3", None, cancellable)
cursor = miner_fs.query(
    "SELECT ?album ?title "
    "FROM tracker:Audio "
    "WHERE { ?album a nmm:MusicAlbum ; nie:title ?title . }",
    cancellable)

print("Music albums:")
while (cursor.next(None)):
    album_id = cursor.get_string(0)[0]
    album_title = cursor.get_string(1)[0]
    print(f"  * {album_title} (ID: {album_id})")
