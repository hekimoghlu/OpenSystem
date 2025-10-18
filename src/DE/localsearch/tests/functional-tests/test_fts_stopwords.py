# -*- coding: utf-8 -*-

# Copyright (C) 2010, Nokia (ivan.frade@nokia.com)
# Copyright (C) 2019-2020, Sam Thursfield (sam@afuera.me.uk)
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

import locale
import os
import unittest as ut

import configuration as cfg
import fixtures


class MinerFTSStopwordsTest(fixtures.TrackerMinerFTSTest):
    """
    Stop words were removed from Tracker SPARQL in 3.6

    See: https://gitlab.gnome.org/GNOME/tracker/-/merge_requests/611
    """

    def test_01_stopwords(self):
        """Test that you can search for English stopwords."""

        TEXT = "The optimism of the action is better than the pessimism of the thought"

        self.set_text(TEXT)

        results = self.search_word("the")
        self.assertEqual(len(results), 1)

        results = self.search_word("of")
        self.assertEqual(len(results), 1)

        results = self.search_word("in")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    fixtures.tracker_test_main()
