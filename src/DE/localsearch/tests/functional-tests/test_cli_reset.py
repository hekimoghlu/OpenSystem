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
Test `localsearch reset` subcommand
"""

from typing import *
import dataclasses
import pathlib
import os
import time

import configuration
import fixtures
import shutil

class TestCli(fixtures.TrackerCommandLineTestCase):
    def test_reset(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure idle state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, self.indexed_dir)

        pid = self.sandbox.session_bus.get_connection_unix_process_id_sync(
            'org.freedesktop.LocalSearch3')

        # Reset the database
        self.run_cli(["localsearch", "reset"])

        # Ensure the process is gone
        pid_exists = True
        attempts = 0
        while pid_exists:
            if attempts == 10:
                raise RuntimeError("Took too long to stop indexer")

            try:
                time.sleep(1)
                program = os.readlink("/proc/" + str(pid) + "/exe")
                pid_exists = os.path.basename(program) == "localsearch-3"
                if pid_exists:
                    attempts += 1
            except:
                pid_exists = False

        # Re-start the indexer, check that file is reindexed
        with self.await_document_inserted(target):
            output = self.run_cli(["localsearch", "status"])
            self.assertNotIn("idle", output)

    def test_reset_file(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, target)

        output = self.run_cli(["localsearch", "status"])
        # State should be idle
        self.assertIn("idle", output)

        resource_id = self.tracker.get_content_resource_id(self.uri(target))
        with self.tracker.await_delete(fixtures.DOCUMENTS_GRAPH, resource_id):
            self.run_cli(["localsearch", "reset", "--file", target])

        # Trigger another change, wait for it to be processed
        file = datadir.joinpath("text/mango.txt")
        target2 = pathlib.Path(os.path.join(self.indexed_dir, 'mango2.txt'))
        with self.await_document_inserted(target2):
            shutil.copy(file, target2)

        # Both files should exist
        print (self.uri(target))
        self.assertResourceExists(self.uri(target))
        print (self.uri(target2))
        self.assertResourceExists(self.uri(target2))

    def test_reset_wrongargs(self):
        err = None
        out = None
        try:
            # Pass unknown arg
            out = self.run_cli(["localsearch", "reset", "--asdf"])
        except Exception as e:
            err = str(e)
        finally:
            self.assertIn("--asdf", err);
            self.assertIsNone(out)

if __name__ == "__main__":
    fixtures.tracker_test_main()
