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
Test `localsearch status` subcommand
"""

from typing import *
import dataclasses
import pathlib
import os

import configuration
import fixtures
import shutil

class TestCli(fixtures.TrackerCommandLineTestCase):
    def test_status(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure idle state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, self.indexed_dir)

        output = self.run_cli(["localsearch", "status"])
        self.assertIn("1 file", output)
        self.assertIn("1 folder", output)
        self.assertIn("idle", output)

    def test_status_stat(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, target)

        output = self.run_cli(["localsearch", "status", "--stat"])
        self.assertIn("nfo:Folder", output)
        self.assertIn("nfo:Document", output)
        self.assertIn("tracker:FileSystem", output)
        self.assertIn("tracker:Documents", output)

    def test_status_wrongargs(self):
        err = None
        out = None
        try:
            # Pass unknown arg
            out = self.run_cli(["localsearch", "status", "--asdf"])
        except Exception as e:
            err = str(e)
        finally:
            self.assertIn("--asdf", err);
            self.assertIsNone(out)

if __name__ == "__main__":
    fixtures.tracker_test_main()
