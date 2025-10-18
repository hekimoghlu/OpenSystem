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

"""
Test that resource removal does not leave debris or clobber too much,
especially in the case where nie:InformationElement != nie:DataObject
"""

import os
import pathlib
import unittest as ut

import configuration as cfg
import fixtures


class MinerResourceRemovalTest(fixtures.TrackerMinerTest):
    def prepare_directories(self):
        # Override content from the base class
        pass

    def create_extra_audio_content(self, file_urn, url, title):
        sparql = (
            'WITH <%s> INSERT { \
                    <%s> a nfo:FileDataObject . \
                    <%s> a nmm:MusicPiece ; \
                         nie:title "%s" ; \
                         nie:isStoredAs <%s> . \
                  } '
            % (fixtures.AUDIO_GRAPH, url, file_urn, title, url)
        )

        with self.tracker.await_insert(
            fixtures.AUDIO_GRAPH,
            f'a nmm:MusicPiece; nie:title "{title}"',
            timeout=cfg.AWAIT_TIMEOUT,
        ) as resource:
            self.tracker.update(sparql)
        return resource

    def create_text_file(self, file_name):
        path = pathlib.Path(self.path(file_name))
        text = "Test"

        with self.await_document_inserted(file_name, content=text) as resource:
            path.write_text(text)
        return resource

    def test_01_file_deletion(self):
        """
        Ensure every logical resource (nie:InformationElement) contained with
        in a file is deleted when the file is deleted.
        """

        file_1_name = "test-monitored/test_1.txt"
        file_2_name = "test-monitored/test_2.txt"

        file_1 = self.create_text_file(self.path(file_1_name))
        file_2 = self.create_text_file(self.path(file_2_name))
        # This creates an unrealistic situation, in which a text data-object
        # has audio content. Because nie:isStoredAs links it to the text file,
        # it should nevertheless be removed when the text file is deleted.
        ie_1 = self.create_extra_audio_content(
            file_1.urn, self.uri(file_1_name), "Test resource 1"
        )
        ie_2 = self.create_extra_audio_content(
            file_2.urn, self.uri(file_2_name), "Test resource 2"
        )

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, file_1.id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.unlink(self.path("test-monitored/test_1.txt"))

        self.assertResourceMissing(self.uri(file_1_name))
        # Ensure the logical resource is deleted when the relevant file is
        # removed.
        self.assertResourceMissing(ie_1.urn)

        self.assertResourceExists(self.uri(file_2_name))
        self.assertResourceExists(ie_2.urn)

    def test_02_stale_file(self):
        """
        Simulate the situation where the extractor will find a file that
        is already deleted, and ensure the situation will end up corrected.
        """

        file_1_name = "test-monitored/test_1.txt"
        file_1_uri = self.uri(file_1_name)
        file_2_name = "test-monitored/test_2.txt"
        file_2_uri = self.uri(file_2_name)
        file_3_name = "test-monitored/test_3.txt"
        file_3_uri = self.uri(file_3_name)

        # First create fake data for a file that would be picked up
        # by the extractor
        self.tracker.update(
            "INSERT DATA { GRAPH tracker:Audio { <test:content> a nie:InformationElement . <%s> a nfo:FileDataObject ; nie:interpretedAs <test:content> } }" % file_1_uri
        )

        self.assertResourceExists(file_1_uri)
        self.assertResourceExists("test:content")
        self.assertResourceMissing(file_2_uri)

        # Second create a second file that will be picked up by the extractor
        file_2 = self.create_text_file(file_2_name)

        # Third check that the first resource does not exist
        self.assertResourceMissing(file_1_uri)
        self.assertResourceMissing("test:content")
        self.assertResourceExists(file_2_uri)

        # Fourth, create another file to ensure the extractor is not stuck
        file_3 = self.create_text_file(file_3_name)
        self.assertResourceMissing(file_1_uri)
        self.assertResourceExists(file_2_uri)
        self.assertResourceExists(file_3_uri)

    # def test_02_removable_device_data (self):
    #    """
    #    Tracker does periodic cleanups of data on removable volumes that haven't
    #    been seen since 'removable-days-threshold', and will also remove all data
    #    from removable volumes if 'index-removable-devices' is disabled.
    #
    #    FIXME: not yet possible to test this - we need some way of mounting
    #    a fake removable volume: https://bugzilla.gnome.org/show_bug.cgi?id=659739
    #    """

    # dconf = DConfClient ()
    # dconf.write (cfg.DCONF_MINER_SCHEMA, 'index-removable-devices', 'true')

    # self.mount_test_removable_volume ()

    # self.add_test_resource ("urn:test:1", test_volume_urn)
    # self.add_test_resource ("urn:test:2", None)

    # Trigger removal of all resources from removable devices
    # dconf.write (cfg.DCONF_MINER_SCHEMA, 'index-removable-devices', 'false')

    # Check that only the data on the removable volume was deleted
    # self.await_updates (2)


if __name__ == "__main__":
    fixtures.tracker_test_main()
