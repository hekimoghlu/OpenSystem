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
Test startup being delayed via systemd state monitoring
"""

import time
import fixtures
import dbusmock
import subprocess

class TestSystemd(fixtures.TrackerMinerTest):
    def setUp(self):
        super(TestSystemd, self).setUp()
        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')
        (self.systemd, self.systemd_obj) = self.spawn_server_template(
            'systemd', {}, system_bus=False)

    def tearDown(self):
        self.systemd.terminate()
        self.systemd.wait()
        super(TestSystemd, self).tearDown()

    def test_systemd_ready(self):
        self.systemd_obj.AddProperty(
            'org.freedesktop.systemd1.Manager', 'SystemState', 'running')

        filename = "test-monitored/file1.txt"
        path = self.path(filename)
        uri = self.uri(filename)

        start = time.time()
        # This activates the indexer
        with self.await_document_inserted(path):
            with open(path, "w") as f:
                f.write("Foo bar baz")

        elapsed = time.time() - start
        self.assertTrue(elapsed < 5)

        # The file should be indexed as usual
        self.assertResourceExists(uri)


    def test_systemd_busy_then_settle(self):
        self.systemd_obj.AddProperty(
            'org.freedesktop.systemd1.Manager', 'SystemState', 'activating')

        filename = "test-monitored/file1.txt"
        path = self.path(filename)
        uri = self.uri(filename)
        with open(path, "w") as f:
            f.write("Foo bar baz")

        time.sleep(1)
        self.assertResourceMissing(uri)

        start = time.time()
        # This activates the indexer
        with self.await_document_inserted(path):
            self.systemd_obj.Set(
                'org.freedesktop.systemd1.Manager', 'SystemState', 'running')
            self.systemd_obj.EmitSignal(
                'org.freedesktop.systemd1.Manager', 'StartupFinished',
                'uuuuuu', [0, 0, 0, 0, 0, 0])

        elapsed = time.time() - start
        # We should ideally test for this, but cannot with slow CI runners
        # self.assertTrue(elapsed < 5)

        # The file should be indexed as usual
        self.assertResourceExists(uri)

    def test_systemd_busy(self):
        self.systemd_obj.AddProperty(
            'org.freedesktop.systemd1.Manager', 'SystemState', 'activating')

        filename = "test-monitored/file1.txt"
        path = self.path(filename)
        uri = self.uri(filename)

        start = time.time()

        # This activates the indexer
        with self.await_document_inserted(path):
            with open(path, "w") as f:
                f.write("Foo bar baz")

        elapsed = time.time() - start
        self.assertTrue(elapsed >= 5)

        # The file should be indexed as usual
        self.assertResourceExists(uri)

        # Check that belated startup finished does not have ill effects
        self.systemd_obj.Set(
            'org.freedesktop.systemd1.Manager', 'SystemState', 'running')
        self.systemd_obj.EmitSignal(
            'org.freedesktop.systemd1.Manager', 'StartupFinished',
            'uuuuuu', [0, 0, 0, 0, 0, 0])
        time.sleep(1)
        self.assertResourceExists(uri)


if __name__ == "__main__":
    fixtures.tracker_test_main()
