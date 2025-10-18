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
Test `localsearch info` subcommand
"""

from typing import *
import dataclasses
import pathlib
import os
import re
import time

import configuration
import fixtures
import shutil

class TestCli(fixtures.TrackerCommandLineTestCase):
    def test_info(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure idle state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, self.indexed_dir)

        output = self.run_cli(["localsearch", "info", self.uri(target)])
        self.assertIn("nfo:PlainTextDocument", output)
        self.assertIn("mango.txt", output)

    def test_info_plain_text(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure idle state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, self.indexed_dir)

        output = self.run_cli(["localsearch", "info", target, "--plain-text-content"])
        self.assertIn("nfo:PlainTextDocument", output)
        self.assertIn("mango.txt", output)
        self.assertIn("monkey", output)

    def test_info_full_namespaces(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure idle state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, self.indexed_dir)

        output = self.run_cli(["localsearch", "info", target, "--full-namespaces"])
        self.assertIn("http://tracker.api.gnome.org/ontology/v3/nfo#PlainTextDocument", output)
        self.assertNotIn("nfo:PlainTextDocument", output)

    def test_info_eligible(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # Copy a file and wait for it to be indexed, in order to ensure idle state
        file = datadir.joinpath("text/mango.txt")
        target = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file)))
        with self.await_document_inserted(target):
            shutil.copy(file, self.indexed_dir)

        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("currently exists", output)
        self.assertIn("File is eligible", output)

    def test_info_eligible_nonexistent(self):
        target = self.path("test-monitored/non-existent.txt")
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("does not exist", output)
        self.assertIn("File is eligible", output)

    def test_info_non_eligible(self):
        # Non-existent file
        target = "/non-existent.txt"
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("does not exist", output)
        self.assertIn("not an indexed folder", output)

        # Hidden file
        target = self.path("test-monitored/.hidden.txt")
        with open(target, "w") as f:
            f.write("Foo bar baz")
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("currently exist", output)
        self.assertIn("hidden file", output)

        # File ineligible by glob filter
        target = self.path("test-monitored/backup.txt~")
        with open(target, "w") as f:
            f.write("Foo bar baz")
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("currently exist", output)
        self.assertIn("based on filters", output)

        # Directory ineligible by glob filter
        target = pathlib.Path(self.path("test-monitored/lost+found"))
        target.mkdir(parents=True, exist_ok=True)
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("currently exist", output)
        self.assertIn("based on filters", output)

        # File filtered by parent-dir filter
        nomedia = self.path("test-monitored/.nomedia")
        with open(nomedia, "w") as f:
            f.write("")
        target = self.path("test-monitored/file.txt")
        with open(target, "w") as f:
            f.write("Foo bar baz")
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("currently exist", output)
        self.assertIn("based on content filters", output)


    def test_info_non_eligible_parent(self):
        # Hidden parent folder
        target = pathlib.Path(self.path("test-monitored/.folder/file.txt"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('foo bar baz')
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("Parent directory", output)
        self.assertIn("hidden file", output)

        # Parent folder ineligible by filter
        target = pathlib.Path(self.path("test-monitored/lost+found/file.txt"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('foo bar baz')
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("Parent directory", output)
        self.assertIn("based on filters", output)

        # Parent folder ineligible by content filter
        target = pathlib.Path(self.path("test-monitored/a/b/file.txt"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('foo bar baz')
        nomedia = pathlib.Path(self.path("test-monitored/a/.nomedia"))
        nomedia.write_text('')
        output = self.run_cli(["localsearch", "info", "--eligible", target])
        self.assertIn("Parent directory", output)
        self.assertIn("based on content filters", output)


    def test_info_noargs(self):
        err = None
        out = None
        try:
            out = self.run_cli(["localsearch", "info"])
        except Exception as e:
            err = str(e)
        finally:
            self.assertIn("Usage", err)
            self.assertIsNone(out)

    def test_info_wrongargs(self):
        err = None
        out = None
        try:
            # Pass unknown arg
            out = self.run_cli(["localsearch", "info", "--asdf"])
        except Exception as e:
            err = str(e)
        finally:
            self.assertIn("--asdf", err);
            self.assertIsNone(out)

if __name__ == "__main__":
    fixtures.tracker_test_main()
