# Copyright (C) 2020, Sam Thursfield <sam@afuera.me.uk>
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

"""
Test `localsearch search` subcommand
"""

from typing import *
import dataclasses
import pathlib
import os
import re

import configuration
import fixtures
import shutil

class TestCli(fixtures.TrackerCommandLineTestCase):
    def test_search(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        # FIXME: synchronous `tracker index` isn't ready yet;
        # see https://gitlab.gnome.org/GNOME/tracker/-/issues/188
        # in the meantime we manually wait for it to finish.

        file1 = datadir.joinpath("text/Document 1.txt")
        target1 = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file1)))
        with self.await_document_inserted(target1):
            shutil.copy(file1, self.indexed_dir)

        file2 = datadir.joinpath("text/Document 2.txt")
        target2 = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file2)))
        with self.await_document_inserted(target2):
            shutil.copy(file2, self.indexed_dir)

        folder_name = "test-folder"
        folder_path = pathlib.Path(os.path.join(self.indexed_dir, folder_name))
        with self.await_insert_dir(folder_path):
            try:
                os.mkdir(os.path.join(self.indexed_dir, folder_name))
            except OSError as error:
                print(error)

        # FIXME: the --all should NOT be needed.
        # See: https://gitlab.gnome.org/GNOME/tracker-miners/-/issues/116
        output = self.run_cli(["localsearch", "search", "--all", "banana"])
        self.assertIn(target1.as_uri(), output)
        self.assertNotIn(target2.as_uri(), output)

        folder_output = self.run_cli(
            ["localsearch", "search", "--folders", "test-monitored"]
        )
        self.assertIn(self.indexed_dir, folder_output)
        self.assertNotIn(folder_path.as_uri(), folder_output)

    def test_search_filename(self):
        datadir = pathlib.Path(__file__).parent.joinpath("data/content")

        file1 = datadir.joinpath("text/mango.txt")
        target1 = pathlib.Path(os.path.join(self.indexed_dir, os.path.basename(file1)))
        with self.await_document_inserted(target1):
            shutil.copy(file1, self.indexed_dir)

        target2 = pathlib.Path(os.path.join(self.indexed_dir, "Document 2.txt"))

        search_output = self.run_cli(["localsearch", "search", "mango"])
        self.assertIn(target1.as_uri(), search_output)
        self.assertNotIn(target2.as_uri(), search_output)

if __name__ == "__main__":
    fixtures.tracker_test_main()
