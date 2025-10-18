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
Test `localsearch` commandline tool
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
    def test_noargs(self):
        out = self.run_cli(["localsearch"])
        self.assertIn("Available localsearch commands are", out)

    def test_help(self):
        output = self.run_cli(["localsearch", "help", "info"])
        self.assertIn("localsearch", output)

    def test_help_wrongargs(self):
        out = ""
        err = ""
        try:
            out = self.run_cli(["localsearch", "help", "asdf"])
        except Exception as e:
            err = str(e)

        self.assertEqual("", out)
        self.assertIn("CLI command failed", err)

if __name__ == "__main__":
    fixtures.tracker_test_main()
