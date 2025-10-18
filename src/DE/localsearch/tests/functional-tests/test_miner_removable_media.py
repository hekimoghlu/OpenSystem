# Copyright (C) 2021, Codethink Ltd
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
# Author: Sam Thursfield <sam@afuera.me.uk>


import logging
import pathlib

import fixtures
from fixtures import UnmountFlags

from gi.repository import GLib

log = logging.getLogger(__name__)


class MinerRemovableMediaTest(fixtures.TrackerMinerRemovableMediaTest):
    """Tests for tracker-miner-fs with index-removable-devices feature."""

    def setUp(self):
        super(MinerRemovableMediaTest, self).setUp()

        self.device_path = pathlib.Path(self.workdir).joinpath("removable-device-1")
        self.device_path.mkdir()

    def data_source_available(self, uri):
        """Check tracker:available set on the datasource containing `uri`"""
        result = self.tracker.query(
            """
          SELECT tracker:available(?folder) WHERE {
              <%s> nie:dataSource ?folder
          }"""
            % uri
        )
        return True if result[0][0] == "true" else False

    def create_test_data(self):
        files = ["file1.txt", "dir1/file2.txt"]

        for f in files:
            path = self.device_path.joinpath(f)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("This file exists.")

        return files

    def test_add_remove_device(self):
        """Device should be indexed by Tracker when connected."""

        files = self.create_test_data()

        self.add_removable_device(self.device_path)

        for f in files:
            path = self.device_path.joinpath(f)
            self.ensure_document_inserted(path)
            assert self.data_source_available(path.as_uri())

        with self.await_device_removed(self.device_path.as_uri()):
            self.remove_removable_device(self.device_path)

        for f in files:
            self.ensure_document_inserted(path)
            assert not self.data_source_available(path.as_uri()), (
                "Path %s should be marked unavailable" % path.as_uri()
            )

    def test_remove_busy_device(self):
        """Check that a busy device becomes indexed again."""

        files = self.create_test_data()

        self.add_removable_device(self.device_path)

        for f in files:
            path = self.device_path.joinpath(f)
            self.ensure_document_inserted(path)
            assert self.data_source_available(path.as_uri())

        with self.await_device_removed(self.device_path.as_uri()):
            self.remove_removable_device(self.device_path, UnmountFlags.EMULATE_BUSY)

        uri = self.device_path.as_uri();
        result = self.tracker.query(
            'SELECT DISTINCT tracker:id(?r) WHERE { ?r a nie:InformationElement; nie:isStoredAs "%s" }' % uri)
        assert len(result) == 1
        resource_id = int(result[0][0])

        with self.tracker.await_content_update(
            fixtures.FILESYSTEM_GRAPH,
            resource_id,
            f'tracker:available false',
            f'tracker:available true'
        ):
            # no-op
            resource_id = 0

class MinerRemovableMediaTestNoPreserve(MinerRemovableMediaTest):
    """Tests for tracker-miner-fs with index-removable-devices feature."""

    def __get_text_documents(self):
        return self.tracker.query(
            """
          SELECT DISTINCT ?url WHERE {
              ?u a nfo:TextDocument ;
                 nie:isStoredAs/nie:url ?url.
          }
          """
        )

    def test_add_remove_device(self):
        """Device should be indexed, then deleted."""

        files = self.create_test_data()

        self.add_removable_device(self.device_path)

        for f in files:
            path = self.device_path.joinpath(f)
            self.ensure_document_inserted(path)
            assert self.data_source_available(path.as_uri())

        uri = self.device_path.as_uri();
        result = self.tracker.query(
            'SELECT DISTINCT tracker:id(?r) WHERE { ?r a nie:InformationElement; nie:isStoredAs "%s" }' % uri)
        assert len(result) == 1
        id = int(result[0][0])

        with self.tracker.await_content_update(
                fixtures.FILESYSTEM_GRAPH, id,
                f'tracker:available true',
                f'tracker:available false'
        ):
            self.remove_removable_device(self.device_path)


if __name__ == "__main__":
    fixtures.tracker_test_main()
