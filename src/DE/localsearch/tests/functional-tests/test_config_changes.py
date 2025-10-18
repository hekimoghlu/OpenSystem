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
Test configuration changes
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

class TestConfigChanges(fixtures.TrackerMinerTest):
    def setUp(self):
        os.makedirs(self.non_recursive_dir, exist_ok=True)
        super(TestConfigChanges, self).setUp()


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

        # Set ignored-files filter, file should disappear
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'ignored-files', GLib.Variant.new_strv(['*.txt']))

        # Newly created filtered file should not appear either
        filename2 = "test-monitored/file2.txt"
        path2 = self.path(filename2)
        uri2 = self.uri(filename2)
        with open(path2, "w") as f:
            f.write("Foo bar baz")

        time.sleep(3)
        self.assertResourceMissing(uri2)

        # Ensure that both files are back by removing the filter
        with self.await_document_inserted(path1):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'ignored-files', GLib.Variant.new_strv([]))

        self.ensure_document_inserted(path2)


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
        with self.tracker.await_delete(
            fixtures.FILESYSTEM_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'ignored-directories', GLib.Variant.new_strv(['dir']))

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        # Ensure the file/dir are back after removing the filter
        with self.await_document_inserted(f):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'ignored-directories', GLib.Variant.new_strv([]))

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
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'ignored-directories-with-content', GLib.Variant.new_strv(['hide-me']))

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)
        time.sleep(2)

        # Ensure the file/dir are back after removing the filter
        with self.await_document_inserted(f):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'ignored-directories-with-content', GLib.Variant.new_strv([]))

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
        resource_id = self.tracker.get_content_resource_id(uri)

        dconf = self.sandbox.get_dconf_client()

        # Remove directory from configuration, file should disappear
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-single-directories', GLib.Variant.new_strv([]))

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        # Ensure that dir and file are back after configuring it again
        with self.await_document_inserted(path):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-single-directories', GLib.Variant.new_strv([self.non_recursive_dir]))

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
        resource_id = self.tracker.get_content_resource_id(uri)

        dconf = self.sandbox.get_dconf_client()

        # Remove directory from configuration, file should disappear
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-recursive-directories', GLib.Variant.new_strv([]))

        self.assertResourceMissing(dir_uri)
        self.assertResourceMissing(uri)

        # Ensure that dir and file are back after configuring it again
        with self.await_document_inserted(path):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-recursive-directories', GLib.Variant.new_strv([self.indexed_dir]))

        self.assertResourceExists(dir_uri)
        self.assertResourceExists(uri)


    def test_enable_monitors(self):
        dconf = self.sandbox.get_dconf_client()
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'enable-monitors', GLib.Variant.new_boolean(False))

        time.sleep(1)

        filename1 = "test-monitored/file1.txt"
        path1 = self.path(filename1)
        uri1 = self.uri(filename1)
        with open(path1, "w") as f:
            f.write("Foo bar baz")

        # Test that the file is not detected
        time.sleep(3)
        self.assertResourceMissing(uri1)

        # Test that the missing file is inserted
        with self.await_document_inserted(path1):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'enable-monitors', GLib.Variant.new_boolean(True))

        self.assertResourceExists(uri1)

        # Test that a new file is detected after re-enabling monitors
        filename2 = "test-monitored/file2.txt"
        path2 = self.path(filename2)
        uri2 = self.uri(filename2)
        with self.await_document_inserted(path2):
            with open(path2, "w") as f:
                f.write("Foo bar baz")

        self.assertResourceExists(uri2)


class TestConfigMount(fixtures.TrackerMinerRemovableMediaTest):
    """Test configuration with (non-)removable mounts"""

    def config(self):
        settings = super(TestConfigMount, self).config()
        settings["org.freedesktop.Tracker3.Miner.Files"][
            "index-removable-devices"
        ] = GLib.Variant.new_boolean(False)
        return settings

    def setUp(self):
        super(TestConfigMount, self).setUp()
        self.device_path = pathlib.Path(self.workdir).joinpath("mount-1")

    def test_index_removable_devices(self):
        self.device_path.mkdir()
        path = self.device_path.joinpath("file1.txt")
        path.write_text("Foo bar baz")
        self.add_removable_device(self.device_path)

        # Ensure the file was not indexed
        time.sleep(3)
        self.assertResourceMissing(path.as_uri())

        dconf = self.sandbox.get_dconf_client()

        # File should be indexed with the config change
        with self.await_document_inserted(path):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-removable-devices', GLib.Variant.new_boolean(True))

        # The "device" should be "removed", with the config change
        with self.await_device_removed(self.device_path.as_uri()):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-removable-devices', GLib.Variant.new_boolean(False))


    def test_non_removable_in_index_single_directories(self):
        self.device_path.mkdir()
        path = self.device_path.joinpath("file1.txt")
        path.write_text("Foo bar baz")
        self.add_removable_device(self.device_path, MountFlags.NON_REMOVABLE)

        time.sleep(3)
        self.assertResourceMissing(path.as_uri())

        dconf = self.sandbox.get_dconf_client()

        with self.await_document_inserted(path):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-single-directories', GLib.Variant.new_strv([str(self.device_path)]))

        self.assertResourceExists(path.as_uri())
        resource_id = self.tracker.get_content_resource_id(path.as_uri())

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-single-directories', GLib.Variant.new_strv([]))

        self.assertResourceMissing(path.as_uri())


    def test_preconfigured_non_removable_in_index_single_directories(self):
        dconf = self.sandbox.get_dconf_client()
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'index-single-directories', GLib.Variant.new_strv([str(self.device_path)]))

        with self.await_insert_dir(self.device_path):
            self.device_path.mkdir()
            self.add_removable_device(self.device_path, MountFlags.NON_REMOVABLE)

        self.assertResourceExists(self.device_path.as_uri())
        resource_id = self.tracker.get_resource_id_by_uri(self.device_path.as_uri())

        with self.tracker.await_delete(
            fixtures.FILESYSTEM_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            self.remove_removable_device(self.device_path)
            self.device_path.rmdir()

        self.assertResourceMissing(self.device_path.as_uri())


    def test_non_removable_in_index_recursive_directories(self):
        path = self.device_path.joinpath("dir/file1.txt")
        path.parent.mkdir(parents=True)
        path.write_text("Foo bar baz")
        self.add_removable_device(self.device_path, MountFlags.NON_REMOVABLE)

        dconf = self.sandbox.get_dconf_client()

        with self.await_document_inserted(path):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-recursive-directories', GLib.Variant.new_strv([self.indexed_dir, str(self.device_path)]))

        self.assertResourceExists(path.as_uri())
        resource_id = self.tracker.get_content_resource_id(path.as_uri())

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, resource_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            dconf.write (
                'org.freedesktop.Tracker3.Miner.Files',
                'index-recursive-directories', GLib.Variant.new_strv([self.indexed_dir]))


    def test_preconfigured_non_removable_in_index_recursive_directories(self):
        dconf = self.sandbox.get_dconf_client()
        dconf.write (
            'org.freedesktop.Tracker3.Miner.Files',
            'index-recursive-directories',
            GLib.Variant.new_strv([self.indexed_dir, str(self.device_path)]))

        with self.await_insert_dir(self.device_path):
            self.device_path.mkdir()
            self.add_removable_device(self.device_path, MountFlags.NON_REMOVABLE)

        self.assertResourceExists(self.device_path.as_uri())
        resource_id = self.tracker.get_resource_id_by_uri(self.device_path.as_uri())

        with self.await_device_removed(self.device_path.as_uri()):
            self.remove_removable_device(self.device_path)
            self.device_path.rmdir()


if __name__ == "__main__":
    fixtures.tracker_test_main()
