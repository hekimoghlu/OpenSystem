# Copyright (C) 2016, Sam Thursfield (sam@afuera.me.uk)
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

"""
Tests failure cases of tracker-extract.
"""

import os
import shutil
import unittest as ut

import configuration as cfg
import fixtures


VALID_FILE = os.path.join(
    os.path.dirname(__file__), "data", "extractor-content", "audio", "mp3-id3v2.4-1.mp3"
)
VALID_FILE_TITLE = "Simply Juvenile"

TRACKER_EXTRACT_FAILURE_DATA_SOURCE = "tracker:extractor-failure-data-source"


class ExtractorDecoratorTest(fixtures.TrackerMinerTest):
    def test_reextraction(self):
        """Tests whether known files are still re-extracted on user request."""
        miner_fs = self.miner_fs
        store = self.tracker

        # Insert a valid file and wait extraction of its metadata.
        file_path = os.path.join(self.indexed_dir, os.path.basename(VALID_FILE))
        expected = f'a nmm:MusicPiece ; nie:title "{VALID_FILE_TITLE}"'
        with self.tracker.await_insert(
            fixtures.AUDIO_GRAPH, expected, timeout=cfg.AWAIT_TIMEOUT
        ) as resource:
            shutil.copy(VALID_FILE, file_path)
        file_urn = resource.urn

        try:
            # Remove a key piece of metadata.
            #   (Writeback must be disabled in the config so that the file
            #   itself is not changed).
            store.update(
                "DELETE { GRAPH ?g { <%s> nie:title ?title } }"
                " WHERE { GRAPH ?g { <%s> nie:title ?title } }" % (file_urn, file_urn)
            )
            assert not store.ask("ASK { <%s> nie:title ?title }" % file_urn)

            file_uri = "file://" + os.path.join(self.indexed_dir, file_path)

            # Request re-indexing (same as `tracker index --file ...`)
            # The extractor should reindex the file and re-add the metadata that we
            # deleted, so we should see the nie:title property change.
            with self.tracker.await_insert(
                fixtures.AUDIO_GRAPH,
                f'nie:title "{VALID_FILE_TITLE}"',
                timeout=cfg.AWAIT_TIMEOUT,
            ):
                miner_fs.index_location(file_uri, [], [])

            title_result = store.query(
                "SELECT ?title { <%s> nie:interpretedAs/nie:title ?title }" % file_uri
            )
            assert len(title_result) == 1
            self.assertEqual(title_result[0][0], VALID_FILE_TITLE)
        finally:
            os.remove(file_path)


    def test_unknown_hash(self):
        """Tests whether unknown files are deleted from content graphs."""
        # This test simulates the conditions that might happen if a file
        # changes in a way that it might look like a candidate for extraction,
        # but then somehow this changed by the time tracker-extract-3 opens
        # it.
        miner_fs = self.miner_fs
        store = self.tracker

        expected = f'a nfo:FileDataObject ; nie:url ?url . FILTER (STRENDS (?url, "/unknown.nfo"))'
        file_path = os.path.join(self.indexed_dir, "unknown.nfo")

        # Unchecked in this test, just here to trigger tracker-extract
        audio_file_path = os.path.join(self.indexed_dir, os.path.basename(VALID_FILE))
        shutil.copy(VALID_FILE, audio_file_path)

        # Insert a file with unhandled mimetype
        with self.tracker.await_insert(
            fixtures.FILESYSTEM_GRAPH, expected, timeout=cfg.AWAIT_TIMEOUT
        ) as resource:
            fd = open(file_path, 'a')
            fd.write('foo')
            fd.close()

        file_urn = resource.urn
        file_id = resource.id

        try:
            store.update(
                "INSERT DATA { GRAPH tracker:Audio { <id:1234> a nie:InformationElement  . <%s> a nfo:FileDataObject; nie:interpretedAs <id:1234> } }" % (file_urn)
            )
            # Forcibly insert unrelated file into the audio graph, and check it is removed
            with self.tracker.await_delete(
                fixtures.AUDIO_GRAPH, file_id,
                timeout=cfg.AWAIT_TIMEOUT,
            ):
                store.update(
                    "INSERT DATA { GRAPH tracker:Audio { <%s> a nfo:FileDataObject ; nie:url '%s' } }" % (file_urn, file_urn)
                )

            # Assert that the file does not exist in the audio graph
            assert not store.ask("ASK { GRAPH tracker:Audio { <%s> a nfo:FileDataObject } }" % file_urn)
        finally:
            os.remove(file_path)


if __name__ == "__main__":
    fixtures.tracker_test_main()
