# Copyright (C) 2025, Red Hat Inc.
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
#
# Author: Carlos Garnacho <carlosg@gnome.org>

"""
Test offline configuration changes
"""

import gi
from gi.repository import GLib

import os
import pathlib
import time
import configuration as cfg
import fixtures
from trackertestutils.dconf import DConfClient
from fixtures import MountFlags
from minerhelper import MinerFsHelper

class TestOfflineConfigChanges(fixtures.TrackerMinerTest):
    def setUp(self):
        os.makedirs(self.non_recursive_dir, exist_ok=True)
        super(TestOfflineConfigChanges, self).setUp()


    def test_ignored_files(self):
        filename1 = "test-monitored/file1.txt"
        path1 = self.path(filename1)
        uri1 = self.uri(filename1)
        with self.await_document_inserted(path1):
            with open(path1, "w") as f:
                f.write("Foo bar baz")

        self.assertResourceExists(uri1)
        resource_id = self.tracker.get_content_resource_id(uri1)

        dconf = self.sandbox.get_dconf_client()
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'ignored-files', GLib.Variant.new_strv(['*.txt']))

        # Newly created filtered file should not appear either
        filename2 = "test-monitored/file2.txt"
        path2 = self.path(filename2)
        uri2 = self.uri(filename2)
        with open(path2, "w") as f:
            f.write("Foo bar baz")

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceMissing(uri2)

        # Ensure that both files are back by removing the filter
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'ignored-files', GLib.Variant.new_strv([]))

        with self.await_document_inserted(path1):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceExists(uri2)


    def test_ignored_directories(self):
        filename = "test-monitored/dir/file1.txt"
        f = pathlib.Path(self.path(filename))
        uri = self.uri(filename)
        dir_uri = self.uri("test-monitored/dir")
        with self.await_document_inserted(f):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("Foo bar baz")

        self.assertResourceExists(uri)
        resource_id = self.tracker.get_content_resource_id(dir_uri)

        dconf = self.sandbox.get_dconf_client()

        # Set ignored-directories filter, directory and file should disappear
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'ignored-directories', GLib.Variant.new_strv(['dir']))

        with self.tracker.await_delete(
            fixtures.FILESYSTEM_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        # Ensure the file/dir are back after removing the filter
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'ignored-directories', GLib.Variant.new_strv([]))

        with self.await_document_inserted(f):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceExists(dir_uri)
        self.assertResourceExists(uri)


    def test_ignored_directories_with_content(self):
        filename = "test-monitored/dir/foo.txt"
        f = pathlib.Path(self.path(filename))
        f.parent.mkdir(parents=True, exist_ok=True)
        uri = self.uri(filename)
        dir_uri = self.uri("test-monitored/dir")
        hidden = pathlib.Path(self.path("test-monitored/dir/hide-me"))
        hidden.write_text("")
        with self.await_document_inserted(f):
            f.write_text("Foo bar baz")

        self.assertResourceExists(uri)
        resource_id = self.tracker.get_content_resource_id(uri)

        dconf = self.sandbox.get_dconf_client()

        # Set ignored-directories-with-content filter, dir and file should disappear
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'ignored-directories-with-content', GLib.Variant.new_strv(['hide-me']))

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        # Ensure the file/dir are back after removing the filter
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'ignored-directories-with-content', GLib.Variant.new_strv([]))

        with self.await_document_inserted(f):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceExists(dir_uri)
        self.assertResourceExists(uri)


    def test_index_single_directories(self):
        filename = "test-non-recursive/file1.txt"
        path = self.path(filename)
        uri = self.uri(filename)
        dir_uri = self.uri("test-non-recursive")
        with self.await_document_inserted(path):
            with open(path, "w") as f:
                f.write("Foo bar baz")

        self.assertResourceExists(uri)
        self.assertResourceExists(dir_uri)
        resource_id = self.tracker.get_content_resource_id(dir_uri)

        dconf = self.sandbox.get_dconf_client()
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'index-single-directories', GLib.Variant.new_strv([]))

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'index-single-directories', GLib.Variant.new_strv([self.non_recursive_dir]))

        self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
        with self.await_document_inserted(path):
            self.miner_fs.start()

        self.assertResourceExists(dir_uri)
        self.assertResourceExists(uri)


    def test_index_recursive_directories(self):
        filename = "test-monitored/file1.txt"
        path = self.path(filename)
        uri = self.uri(filename)
        dir_uri = self.uri("test-monitored")
        with self.await_document_inserted(path):
            with open(path, "w") as f:
                f.write("Foo bar baz")

        self.assertResourceExists(uri)
        self.assertResourceExists(dir_uri)
        resource_id = self.tracker.get_content_resource_id(dir_uri)

        dconf = self.sandbox.get_dconf_client()
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'index-recursive-directories', GLib.Variant.new_strv([]))

        self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())

        # Remove directory from configuration, file should disappear
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            self.miner_fs.start()

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'index-recursive-directories', GLib.Variant.new_strv([self.indexed_dir]))

        self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())

        # Ensure that dir and file are back after configuring it again
        with self.await_document_inserted(path):
            self.miner_fs.start()

        self.assertResourceExists(dir_uri)
        self.assertResourceExists(uri)


    def test_enable_monitors(self):
        dconf = self.sandbox.get_dconf_client()
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'enable-monitors', GLib.Variant.new_boolean(False))

        # Start the indexer
        start_check = "test-monitored/start_check.txt"
        with open(self.path(start_check), "w") as f:
            f.write("Foo bar baz")

        with self.await_document_inserted(self.path(start_check)):
            self.miner_fs = MinerFsHelper(self.sandbox.get_session_bus_connection())
            self.miner_fs.start()

        # Create file, test that the file is not detected
        filename1 = "test-monitored/file1.txt"
        path1 = self.path(filename1)
        uri1 = self.uri(filename1)
        with open(path1, "w") as f:
            f.write("Foo bar baz")

        time.sleep(2)
        self.assertResourceMissing(uri1)


if __name__ == "__main__":
    fixtures.tracker_test_main()
