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
Test `localsearch inhibit` subcommand
"""

import gi

from gi.repository import Gio

from typing import *
import dataclasses
import pathlib
import os
import subprocess
import time
import sys

import configuration
import fixtures
import shutil

class TestCli(fixtures.TrackerCommandLineTestCase):
    def test_inhibit(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        with subprocess.Popen(
                ["localsearch", "inhibit", "cat"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL) as proc:
            paused = False
            n_tries = 0
            while not paused:
                [apps, reasons] = self.miner_fs.miner_fs.GetPauseDetails()
                if len(apps) > 0:
                    paused = True
                else:
                    n_tries += 1
                    if n_tries > 30:
                        proc.stdin.close()
                        raise RuntimeError("did not pause")
                    time.sleep(1)

            # Check that the command appears in the list
            output = self.run_cli(["localsearch", "inhibit", "--list"])
            self.assertIn("cat", output)

            file = datadir.joinpath("text/mango.txt")
            target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
            shutil.copy(file, self.indexed_dir)

            # Wait some time and check that the resource is still missing
            time.sleep(3)
            self.assertResourceMissing(self.uri(target))

            # Check that the file gets indexed after the inhibition is gone
            with self.await_document_inserted(target):
                # Close stdin to end the cat process
                proc.stdin.close()

            # List should now be empty
            output = self.run_cli(["localsearch", "inhibit", "--list"])
            self.assertEqual("", output)

    def test_inhibit_wrongargs(self):
        err = None
        out = None
        try:
            # Pass unknown arg
            out = self.run_cli(["localsearch", "inhibit", "--asdf"])
        except Exception as e:
            err = str(e)
        finally:
            self.assertIn("--asdf", err);
            self.assertIsNone(out)

if __name__ == "__main__":
    fixtures.tracker_test_main()
