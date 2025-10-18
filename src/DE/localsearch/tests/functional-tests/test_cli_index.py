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
Test `localsearch index` subcommand
"""

from typing import *
import dataclasses
import pathlib
import os

import configuration
import fixtures
import shutil

class TestCli(fixtures.TrackerCommandLineTestCase):
    def test_index(self):
        output = self.run_cli(["localsearch", "index"])
        self.assertIn("test-monitored", output)
        self.assertIn("test-non-recursive", output)

    def test_index_remove_add(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        self.run_cli(["localsearch", "index", "--remove", self.indexed_dir])
        output = self.run_cli(["localsearch", "index"])
        self.assertNotIn("test-monitored", output)

        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        shutil.copy(file, self.indexed_dir)

        # Add the directory back, and wait for the new file to be indexed
        with self.await_document_inserted(target):
            self.run_cli(["localsearch", "index", "--recursive", "--add", self.indexed_dir])

    def test_index_wrongargs(self):
        err = None
        out = None
        try:
            # Pass unknown arg
            out = self.run_cli(["localsearch", "index", "--asdf"])
        except Exception as e:
            err = str(e)
        finally:
            self.assertIn("--asdf", err);
            self.assertIsNone(out)

if __name__ == "__main__":
    fixtures.tracker_test_main()
