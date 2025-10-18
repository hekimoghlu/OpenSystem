# Copyright (C) 2023, Red Hat Inc.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
# Author: Carlos Garnacho <carlosg@gnome.org>

"""
Tests misbehavior of tracker-extract modules.
"""

import os
import shutil
import sys
import unittest as ut

import configuration as cfg
import fixtures

RULES_DIR = ""
EXTRACTOR_DIR = ""

class MisbehaviorTest(fixtures.TrackerMinerTest):
    """
    Tests crawling and monitoring of configured content locations.
    """

    def environment(self):
        extra_env = cfg.test_environment(self.workdir)
        extra_env["TRACKER_EXTRACTOR_RULES_DIR"] = RULES_DIR
        extra_env["TRACKER_EXTRACTORS_DIR"] = EXTRACTOR_DIR
        return extra_env

    def setUp(self):
        monitored_files = self.create_test_data()

        try:
            # Start the miner.
            fixtures.TrackerMinerTest.setUp(self)

            for tf in monitored_files:
                url = self.uri(tf)
                self.tracker.ensure_resource(
                    fixtures.FILESYSTEM_GRAPH,
                    f"a nfo:FileDataObject; nie:url '{url}' ; tracker:extractorHash ?hash",
                    timeout=cfg.AWAIT_TIMEOUT,
                )
        except Exception:
            cfg.remove_monitored_test_dir(self.workdir)
            raise

    def create_test_data(self):
        monitored_files = [
            "test-monitored/file1.txt",
        ]

        for tf in monitored_files:
            testfile = self.path(tf)
            os.makedirs(os.path.dirname(testfile), exist_ok=True)
            with open(testfile, "w") as f:
                f.write('Some text')

        return monitored_files

    def test_misbehavior(self):
        self.assertEqual(self.tracker.query('ASK { <fail://> a rdfs:Resource }'), [['false']]);

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("ERROR: missing rules dir and extractor module path arguments")
        sys.exit(1)

    RULES_DIR = sys.argv.pop(1)
    EXTRACTOR_DIR = sys.argv.pop(1)
    fixtures.tracker_test_main()
