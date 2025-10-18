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
Test (somewhat) memory pressure handling
"""

import fixtures
import dbusmock
import subprocess

class TestMemoryPressure(fixtures.TrackerMinerTest):
    def setUp(self):
        self.start_system_bus()
        (self.low_memory_monitor, self.low_memory_monitor_obj) = self.spawn_server_template(
            'low_memory_monitor', {})
        super(TestMemoryPressure, self).setUp()

    def tearDown(self):
        self.low_memory_monitor.terminate()
        self.low_memory_monitor.wait()
        super(TestMemoryPressure, self).tearDown()

    def test_memory_pressure(self):
        self.low_memory_monitor_obj.EmitSignal(
            'org.freedesktop.LowMemoryMonitor',
            'LowMemoryWarning', 'y', [255])

        # We cannot test for the effects, test at least
        # that everything works as usual.
        filename1 = "test-monitored/file1.txt"
        path1 = self.path(filename1)
        uri1 = self.uri(filename1)

        with self.await_document_inserted(path1):
            with open(path1, "w") as f:
                f.write("Foo bar baz")

        self.assertResourceExists(uri1)
        self.low_memory_monitor_obj.EmitWarning(255)


if __name__ == "__main__":
    fixtures.tracker_test_main()
