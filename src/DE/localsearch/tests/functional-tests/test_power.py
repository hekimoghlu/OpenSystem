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
Test battery state detection
"""

import gi
gi.require_version('UPowerGlib', '1.0')
from gi.repository import UPowerGlib

import time
import fixtures
import dbusmock

class TestPower(fixtures.TrackerMinerTest):
    @classmethod
    def setUpClass(klass):
        klass.start_system_bus()
        super(TestPower, klass).setUpClass()

    def setUp(self):
        (self.upowerd, self.upowerd_obj) = self.spawn_server_template(
            'upower', {'OnBattery': True})
        super(TestPower, self).setUp()

    def tearDown(self):
        self.upowerd.terminate()
        self.upowerd.wait()
        super(TestPower, self).tearDown()

    def set_battery_state(self, percentage, state, level):
        self.upowerd_obj.SetupDisplayDevice(
            UPowerGlib.DeviceKind.BATTERY,
            state,
            percentage, percentage, 100., # Charge levels
            0.01, 60, 0, # Discharge rates
            True, # Present
            'battery-caution-symbolic', # Icon
            level)

    def test_battery(self):
        # Set discharging battery level
        self.set_battery_state(
            50., UPowerGlib.DeviceState.DISCHARGING,
            UPowerGlib.DeviceLevel.DISCHARGING)

        filename1 = "test-monitored/file1.txt"
        path1 = self.path(filename1)
        uri1 = self.uri(filename1)
        with self.await_document_inserted(path1):
            with open(path1, "w") as f:
                f.write("Roses are red")

        # The file should be indexed as usual
        self.assertResourceExists(uri1)

        self.set_battery_state(
            50., UPowerGlib.DeviceState.CHARGING,
            UPowerGlib.DeviceLevel.NONE)

        filename2 = "test-monitored/file2.txt"
        path2 = self.path(filename2)
        uri2 = self.uri(filename2)
        with self.await_document_inserted(path2):
            with open(path2, "w") as f:
                f.write("Violets are blue")

        self.assertResourceExists(uri2)


    def test_low_battery(self):
        filename1 = "test-monitored/file1.txt"
        path1 = self.path(filename1)
        uri1 = self.uri(filename1)
        with self.await_document_inserted(path1):
            with open(path1, "w") as f:
                f.write("Roses are red")

        # The file should be indexed as usual
        self.assertResourceExists(uri1)

        # Set critical battery level
        self.set_battery_state(
            5., UPowerGlib.DeviceState.DISCHARGING,
            UPowerGlib.DeviceLevel.CRITICAL)

        filename2 = "test-monitored/file2.txt"
        path2 = self.path(filename2)
        uri2 = self.uri(filename2)
        with open(path2, "w") as f:
            f.write("Violets are blue")
            # Wait some time and check that the resource is still missing
            time.sleep(3)
            self.assertResourceMissing(uri2)

        # Restore battery state to normal levels, and
        # expect the missing file to be indexed
        with self.await_document_inserted(path2):
            self.set_battery_state(
                50., UPowerGlib.DeviceState.CHARGING,
                UPowerGlib.DeviceLevel.NONE)

        # The second file should now be indexed
        self.assertResourceExists(uri2)


if __name__ == "__main__":
    fixtures.tracker_test_main()
