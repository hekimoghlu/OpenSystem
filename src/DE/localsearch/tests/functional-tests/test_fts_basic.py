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

#
# TODO:
#     These tests are for files... we need to write them for folders!
#
"""
Monitor a directory, copy/move/remove/update text files and check that
the text contents are updated accordingly in the indexes.
"""

import pathlib
import unittest as ut

# Must import this for logging.
import configuration as cfg
import fixtures


class MinerFTSBasicTest(fixtures.TrackerMinerFTSTest):
    """
    Tests different contents in a single file
    """

    def test_01_single_word(self):
        TEXT = "automobile"
        self.basic_test(TEXT, TEXT)

    def test_02_multiple_words(self):
        TEXT = "automobile with unlimited power"
        self.set_text(TEXT)

        results = self.search_word("automobile")
        self.assertEqual(len(results), 1)
        self.assertIn(self.uri(self.testfile), results)

        results = self.search_word("unlimited")
        self.assertEqual(len(results), 1)
        self.assertIn(self.uri(self.testfile), results)

    def test_03_long_word(self):
        # TEXT is longer than the 200 characters specified in the fts configuration
        TEXT = "ai" * 200
        self.set_text(TEXT)

        results = self.search_word(TEXT)
        self.assertEqual(len(results), 0)

    def test_04_non_existent_word(self):
        TEXT = "This a trick"
        self.set_text(TEXT)
        results = self.search_word("trikc")
        self.assertEqual(len(results), 0)

    def test_05_word_multiple_times_in_file(self):
        TEXT = "automobile is red. automobile is big. automobile is great!"
        self.basic_test(TEXT, "automobile")

    def test_06_sentence(self):
        TEXT = "plastic is fantastic"
        self.basic_test(TEXT, TEXT)

    def test_07_partial_sentence(self):
        TEXT = "plastic is fantastic"
        self.basic_test(TEXT, "is fantastic")

    @ut.skip("Currently fails with: fts5: syntax error near '.'")
    def test_08_strange_word(self):
        # FIXME Not sure what are we testing here
        TEXT = "'summer.time'"
        self.basic_test(TEXT, "summer.time")

    # Skip the test 'search for .'

    def test_09_mixed_letters_and_numbers(self):
        TEXT = "abc123"
        self.basic_test(TEXT, "abc123")

    @ut.skip(
        "We don't ignore numbers by default since https://gitlab.gnome.org/GNOME/tracker/merge_requests/172."
    )
    def test_10_ignore_numbers(self):
        TEXT = "palabra 123123"
        self.set_text(TEXT)
        results = self.search_word("123123")
        self.assertEqual(len(results), 0)

    def test_11_utf16(self):
        TEXT = '你好'
        self.set_text(TEXT, 'utf-16')
        results = self.search_word(TEXT)
        self.assertEqual(len(results), 1)
        self.assertIn(self.uri(self.testfile), results)

    def test_12_windows1252(self):
        TEXT = 'Cañamón'
        self.set_text(TEXT, 'WINDOWS-1252')
        results = self.search_word(TEXT)
        self.assertEqual(len(results), 1)
        self.assertIn(self.uri(self.testfile), results)

    def test_13_long_text(self):
        TEXT = 'Hola caracola ' * 1000
        self.basic_test(TEXT, "hola")


class MinerFTSWAllowlistTest(fixtures.TrackerMinerFTSTest):
    """
    Test that only an allowlisted set of plaintext files get FTS indexed.

    We can create performance problems by indexing big trees of sourcecode,
    video game data, etc., so we use an extension-based allowlist to limit
    what gets FTS indexed.
    """

    def test_no_extension_file(self):
        self.testfile = "test-monitored/miner-fts-test-no-extension"

        path = pathlib.Path(self.path(self.testfile))
        url = self.uri(self.testfile)
        expected = f"a nfo:Document; nie:isStoredAs <{url}>"
        with self.tracker.await_insert(
            fixtures.DOCUMENTS_GRAPH, expected, timeout=cfg.AWAIT_TIMEOUT
        ):
            path.write_text("Definitely do not index this file.")

        # No results for full text search.
        results = self.search_word("Definitely")
        self.assertEqual(len(results), 0)

        # However, there should be an nfo:PlainTextDocument resource as normal.
        self.assertEqual(1, self.tracker.count_instances("nfo:PlainTextDocument"))


if __name__ == "__main__":
    fixtures.tracker_test_main()
