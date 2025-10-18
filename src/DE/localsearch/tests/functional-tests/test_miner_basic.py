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
Monitor a test directory and copy/move/remove/update files and folders there.
Check the basic data of the files is updated accordingly in tracker.
"""

import itertools
import logging
import os
import shutil
import time
import unittest as ut

import configuration as cfg
import fixtures


log = logging.getLogger(__name__)

DEFAULT_TEXT = "Some stupid content, to have a test file"


class MinerCrawlTest(fixtures.TrackerMinerTest):
    """
    Tests crawling and monitoring of configured content locations.
    """

    def setUp(self):
        # We must create the test data before the miner does its
        # initial crawl, or it may miss some files due
        # https://gitlab.gnome.org/GNOME/tracker-miners/issues/79.
        monitored_files = self.create_test_data()

        try:
            # Start the miner.
            fixtures.TrackerMinerTest.setUp(self)

            for tf in monitored_files:
                url = self.uri(tf)
                self.tracker.ensure_resource(
                    fixtures.DOCUMENTS_GRAPH,
                    f"a nfo:Document ; nie:isStoredAs <{url}>",
                    timeout=cfg.AWAIT_TIMEOUT,
                )
        except Exception:
            cfg.remove_monitored_test_dir(self.workdir)
            raise

        logging.info("%s.setUp(): complete", self)

    def create_test_data(self):
        monitored_files = [
            "test-monitored/file1.txt",
            "test-monitored/dir1/file2.txt",
            "test-monitored/dir1/dir2/file3.txt",
        ]

        unmonitored_files = ["test-no-monitored/file0.txt"]

        for tf in itertools.chain(monitored_files, unmonitored_files):
            testfile = self.path(tf)
            os.makedirs(os.path.dirname(testfile), exist_ok=True)
            with open(testfile, "w") as f:
                f.write(DEFAULT_TEXT)

        return monitored_files

    def __get_text_documents(self):
        return self.tracker.query(
            """
          SELECT DISTINCT ?url WHERE {
              ?u a nfo:TextDocument ;
                 nie:isStoredAs/nie:url ?url.
          }
          """
        )

    def __get_parent_urn(self, filepath):
        result = self.tracker.query(
            """
          SELECT DISTINCT ?p WHERE {
              ?u a nfo:FileDataObject ;
                 nie:url \"%s\" ;
                 nfo:belongsToContainer ?p
          }
          """
            % (self.uri(filepath))
        )
        self.assertEqual(len(result), 1)
        return result[0][0]

    def __get_file_urn(self, filepath):
        result = self.tracker.query(
            """
          SELECT DISTINCT ?ia WHERE {
              ?u a nfo:FileDataObject ;
                 nie:interpretedAs ?ia ;
                 nie:url \"%s\" .
          }
          """
            % (self.uri(filepath))
        )
        self.assertEqual(len(result), 1)
        return result[0][0]

    """
    Boot the miner with the correct configuration and check everything is fine
    """

    def test_01_initial_crawling(self):
        """
        The precreated files and folders should be there
        """
        result = self.__get_text_documents()
        self.assertEqual(len(result), 3)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

        # We don't check (yet) folders, because Applications module is injecting results

    # class copy(TestUpdate):
    # FIXME all tests in one class because the miner-fs restarting takes some time (~5 sec)
    # Maybe we can move the miner-fs initialization to setUpModule and then move these
    # tests to different classes

    def test_02_copy_from_unmonitored_to_monitored(self):
        """
        Copy an file from unmonitored directory to monitored directory
        and verify if data base is updated accordingly
        """
        source = os.path.join(self.workdir, "test-no-monitored", "file0.txt")
        dest = os.path.join(self.workdir, "test-monitored", "file0.txt")

        with self.await_document_inserted(dest) as resource:
            shutil.copyfile(source, dest)
        dest_id = resource.id

        # Verify if miner indexed this file.
        result = self.__get_text_documents()
        self.assertEqual(len(result), 4)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/file0.txt"), unpacked_result)

        # Clean the new file so the test directory is as before
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, dest_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.remove(dest)

    def test_03_copy_from_monitored_to_unmonitored(self):
        """
        Copy an file from a monitored location to an unmonitored location
        Nothing should change
        """

        # Copy from monitored to unmonitored
        source = os.path.join(self.workdir, "test-monitored", "file1.txt")
        dest = os.path.join(self.workdir, "test-no-monitored", "file1.txt")
        shutil.copyfile(source, dest)

        time.sleep(1)
        # Nothing changed
        result = self.__get_text_documents()
        self.assertEqual(len(result), 3, "Results:" + str(result))
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

        # Clean the file
        os.remove(dest)

    def test_04_copy_from_monitored_to_monitored(self):
        """
        Copy a file between monitored directories
        """
        source = os.path.join(self.workdir, "test-monitored", "file1.txt")
        dest = os.path.join(
            self.workdir, "test-monitored", "dir1", "dir2", "file-test04.txt"
        )

        with self.await_document_inserted(dest) as resource:
            shutil.copyfile(source, dest)
        dest_id = resource.id

        result = self.__get_text_documents()
        self.assertEqual(len(result), 4)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertIn(
            self.uri("test-monitored/dir1/dir2/file-test04.txt"), unpacked_result
        )

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, dest_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.remove(dest)

        self.assertEqual(3, self.tracker.count_instances("nfo:TextDocument"))

    @ut.skip("https://gitlab.gnome.org/GNOME/tracker-miners/issues/56")
    def test_05_move_from_unmonitored_to_monitored(self):
        """
        Move a file from unmonitored to monitored directory
        """
        source = os.path.join(self.workdir, "test-no-monitored", "file0.txt")
        dest = os.path.join(self.workdir, "test-monitored", "dir1", "file-test05.txt")

        with self.await_document_inserted(dest) as resource:
            shutil.move(source, dest)
        dest_id = resource.id

        result = self.__get_text_documents()
        self.assertEqual(len(result), 4)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file-test05.txt"), unpacked_result)

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, dest_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.remove(dest)

        self.assertEqual(3, self.tracker.count_instances("nfo:TextDocument"))

    ## """ move operation and tracker-miner response test cases """
    # class move(TestUpdate):

    def test_06_move_from_monitored_to_unmonitored(self):
        """
        Move a file from monitored to unmonitored directory
        """
        source = self.path("test-monitored/dir1/file2.txt")
        dest = self.path("test-no-monitored/file2.txt")
        source_id = self.tracker.get_content_resource_id(self.uri(source))
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, source_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            shutil.move(source, dest)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 2)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

        with self.await_document_inserted(source):
            shutil.move(dest, source)
        self.assertEqual(3, self.tracker.count_instances("nfo:TextDocument"))

    def test_07_move_from_monitored_to_monitored(self):
        """
        Move a file between monitored directories
        """

        source = self.path("test-monitored/dir1/file2.txt")
        dest = self.path("test-monitored/file2.txt")

        source_dir_urn = self.__get_file_urn(os.path.dirname(source))
        parent_before = self.__get_parent_urn(source)
        self.assertEqual(source_dir_urn, parent_before)

        resource_id = self.tracker.get_content_resource_id(url=self.uri(source))
        with self.await_document_uri_change(resource_id, source, dest):
            shutil.move(source, dest)

        # Checking fix for NB#214413: After a move operation, nfo:belongsToContainer
        # should be changed to the new one
        dest_dir_urn = self.__get_file_urn(os.path.dirname(dest))
        parent_after = self.__get_parent_urn(dest)
        self.assertNotEqual(parent_before, parent_after)
        self.assertEqual(dest_dir_urn, parent_after)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 3)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

        # Restore the file
        with self.await_document_uri_change(resource_id, dest, source):
            shutil.move(dest, source)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 3)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)

    def test_08_deletion_single_file(self):
        """
        Delete one of the files
        """
        victim = self.path("test-monitored/dir1/file2.txt")
        victim_id = self.tracker.get_content_resource_id(self.uri(victim))
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, victim_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.remove(victim)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 2)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

        # Restore the file
        with self.await_document_inserted(victim):
            with open(victim, "w") as f:
                f.write("Don't panic, everything is fine")

    def test_09_deletion_directory(self):
        """
        Delete a directory
        """
        victim = self.path("test-monitored/dir1")

        file_inside_victim_url = self.uri(os.path.join(victim, "file2.txt"))
        file_inside_victim_id = self.tracker.get_content_resource_id(
            file_inside_victim_url
        )

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, file_inside_victim_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            shutil.rmtree(victim)

        # We have 2 text files inside the directory, but only look for one, assert that
        # the other file is gone as well.
        file_inside_victim_url2 = self.uri(os.path.join(victim, "dir2/file3.txt"))
        counter = 0
        while counter < 10 and self.tracker.ask(
            "ASK { <%s> a nfo:FileDataObject }" % (file_inside_victim_url2)
        ):
            counter += 1
            time.sleep(1)

        result = self.__get_text_documents()
        self.assertEqual(result, [[self.uri("test-monitored/file1.txt")]])

        # Restore the dirs
        os.makedirs(self.path("test-monitored/dir1"))
        os.makedirs(self.path("test-monitored/dir1/dir2"))
        for f in [
            "test-monitored/dir1/file2.txt",
            "test-monitored/dir1/dir2/file3.txt",
        ]:
            filename = self.path(f)
            with self.await_document_inserted(filename):
                with open(filename, "w") as f:
                    f.write("Don't panic, everything is fine")

        # Check everything is fine
        result = self.__get_text_documents()
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertEqual(len(result), 3)

    def test_10_folder_update(self):
        """
        Check that updating a folder updates nfo:belongsToContainer on its children
        """

        directory_uri = self.uri("test-monitored")
        directory = self.path("test-monitored")
        document = self.path("test-monitored/unrelated.txt")
        urn = self.__get_file_urn(directory)

        with self.await_document_inserted(document) as resource:
            # Force an update on the monitored folder, it needs both
            # a explicit request, and an attribute change (obtained
            # indirectly by the new file)
            with open(document, "w") as f:
                f.write(DEFAULT_TEXT)
            self.miner_fs.index_location(directory_uri, [], [])

        new_urn = self.__get_file_urn(directory)
        # Ensure that children remain consistent, old and new ones
        self.assertEqual(
            new_urn, self.__get_parent_urn(self.path("test-monitored/file1.txt"))
        )
        self.assertEqual(
            self.__get_parent_urn(document),
            self.__get_parent_urn(self.path("test-monitored/file1.txt")),
        )

    def test_11_move_from_visible_to_hidden(self):
        """
        Move a file from monitored to unmonitored directory
        """
        source = self.path("test-monitored/dir1/file2.txt")
        dest = self.path("test-monitored/dir1/.file2.txt")
        source_id = self.tracker.get_content_resource_id(self.uri(source))
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, source_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            shutil.move(source, dest)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 2)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

    def test_12_move_from_hidden_to_visible(self):
        """
        Move a file from monitored to unmonitored directory
        """
        source = self.path("test-monitored/dir1/.hidden.txt")
        dest = self.path("test-monitored/dir1/visible.txt")

        with open(source, "w") as f:
            f.write(DEFAULT_TEXT)

        with self.await_document_inserted(dest) as resource:
            shutil.move(source, dest)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 4)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/visible.txt"), unpacked_result)

    def test_13_move_directory_from_visible_to_hidden_and_back(self):
        """
        Move a directory from monitored to unmonitored
        """
        file = self.path("test-monitored/dir1/file2.txt")
        source = self.path("test-monitored/dir1")
        dest = self.path("test-monitored/.dir1")

        file_id = self.tracker.get_content_resource_id(self.uri(file))
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, file_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.rename(source, dest)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 1)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)

        """
        And move it back to the same place
        """
        with self.await_document_inserted(file) as resource:
            shutil.move(dest, source)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 3)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)

    def test_14_trigger_parent_hidden(self):
        """
        Insert a file triggering parent to be hidden
        """
        file = self.path("test-monitored/dir1/.nomedia")
        doc = self.path("test-monitored/dir1/file2.txt")

        doc_id = self.tracker.get_content_resource_id(self.uri(doc))
        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, doc_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            with open(file, "w") as f:
                f.write(DEFAULT_TEXT)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 1)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)

    def test_15_monitor_nofollow_symlinks(self):
        """
        Insert a file triggering parent to be hidden
        """
        source = self.path("test-no-monitored")
        dest = self.path("test-monitored/folder")
        deleted = self.path("test-monitored/file1.txt")
        deleted_id = self.tracker.get_content_resource_id(self.uri(deleted))

        with self.tracker.await_delete(
            fixtures.DOCUMENTS_GRAPH, deleted_id, timeout=cfg.AWAIT_TIMEOUT
        ):
            os.symlink(source, dest)
            os.remove(deleted)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 2)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)

        query = "ASK { <%s> a nfo:FileDataObject }" % self.uri(dest)
        result = self.tracker.query(query)
        self.assertEqual(result[0][0], "true")

        query = "ASK { ?u nie:isStoredAs <%s> }" % self.uri(dest)
        result = self.tracker.query(query)
        self.assertEqual(result[0][0], "false")

    def test_16_folder_update_offline(self):
        """
        Check that updating a folder deletes the old nfo:Folder
        """

        directory_uri = self.uri("test-monitored/dir1")
        document_uri = self.uri("test-monitored/dir1/unrelated.txt")
        dest_uri = self.uri("test-monitored/dir1/unrelated2.txt")
        directory = self.path("test-monitored/dir1")
        document = self.path("test-monitored/dir1/unrelated.txt")
        dest = self.path("test-monitored/dir1/unrelated2.txt")
        ie_urn = self.__get_file_urn(directory)

        self.assertResourceExists(ie_urn)

        # Create a dir to overwrite the original first, to guarantee
        # a different inode.
        os.makedirs(self.path("zzz"))

        self.sandbox.stop_daemon('org.freedesktop.LocalSearch3')

        # Replace the dir
        shutil.rmtree(directory)
        shutil.move(self.path("zzz"), directory)

        conn = self.miner_fs.get_sparql_connection()

        # Write document
        with self.await_document_inserted(document) as resource:
            with open(document, "w") as f:
                f.write(DEFAULT_TEXT)

        new_ie_urn = self.__get_file_urn(directory)
        self.assertNotEqual(new_ie_urn, ie_urn)

        self.assertResourceExists(new_ie_urn)
        self.assertResourceExists(document_uri)

        # Check the old URN is not a folder anymore
        query = "ASK { <%s> a nfo:Folder }"
        result = self.tracker.query(query)
        self.assertEqual(result, [['false']])

        query = "SELECT ?u { ?u nie:isStoredAs <%s> }" % self.uri(directory)
        result = self.tracker.query(query)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], new_ie_urn)

        query = "SELECT ?u { <%s> nie:interpretedAs ?u }" % self.uri(directory)
        result = self.tracker.query(query)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], new_ie_urn)

        # Move document
        resource_id = self.tracker.get_content_resource_id(document_uri)
        dest = self.path("test-monitored/dir1/unrelated2.txt")
        with self.await_document_uri_change(resource_id, document, dest):
            shutil.move(document, dest)

        self.assertResourceMissing(document_uri)
        self.assertResourceExists(dest_uri)

    def test_17_move_and_replace(self):
        """
        Move a file on top of another, and have another file replace the first
        """
        source = self.path("test-monitored/file1.txt");
        source_urn = self.__get_file_urn(source)
        dest = self.path("test-monitored/dir1/file2.txt");
        dest_urn = self.__get_file_urn(dest)

        source_id = self.tracker.get_content_resource_id(self.uri(source))

        # Move file overwriting another
        with self.await_document_uri_change(source_id, source, dest):
            shutil.move(source, dest)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 2)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)

        # Ensure the file replaced by source changed and matches
        new_dest_urn = self.__get_file_urn(dest)
        self.assertNotEqual(dest_urn, new_dest_urn)
        self.assertEqual(source_urn, new_dest_urn)

        # Write a new document in the former position
        with self.await_document_inserted(source) as resource:
            with open(source, "w") as f:
                f.write("Replacement")

        # Ensure the file created at source location is different
        new_source_urn = self.__get_file_urn(source)
        self.assertNotEqual(source_urn, new_source_urn)

    def test_18_move_and_replace_dir(self):
        """
        Move a directory, and create another directory replacing the first
        """
        source = self.path("test-monitored/dir1");
        source_urn = self.__get_file_urn(source)
        dest = self.path("test-monitored/moved");

        with self.await_insert_dir(dest):
            shutil.move(source, dest)

        with self.await_insert_dir(source):
            os.mkdir(source)

        new_source_urn = self.__get_file_urn(source)
        dest_urn = self.__get_file_urn(dest)

        self.assertEqual(dest_urn, source_urn)
        self.assertNotEqual(new_source_urn, source_urn)

    def test_19_move_and_replace_from_hidden(self):
        """
        Create a hidden file, and make it visible. Then repeat.
        """

        old_text = "I am old!"
        new_text = "I am new!"

        source = self.path("test-monitored/.hidden.txt");
        dest = self.path("test-monitored/nothidden.txt");

        with open(source, "w") as f:
            f.write(old_text)
        with self.await_document_inserted(dest):
            shutil.move(source, dest)

        urn = self.__get_file_urn(dest)

        with open(source, "w") as f:
            f.write(new_text)

        resource_id = self.tracker.get_content_resource_id(self.uri(dest))

        with self.tracker.await_content_update(
            fixtures.DOCUMENTS_GRAPH,
            resource_id,
            f'nie:plainTextContent "{old_text}"',
            f'nie:plainTextContent "{new_text}"',
            timeout=cfg.AWAIT_TIMEOUT,
        ):
            shutil.move(source, dest)

        new_urn = self.__get_file_urn(dest)
        self.assertNotEqual(urn, new_urn)

    def test_20_recover_corruption(self):
        """
        Trick a constraint corruption. See how it fares
        """

        # Insert a blank node with a nie:url
        path = self.path("test-monitored/test.txt")
        uri = self.uri("test-monitored/test.txt")
        self.tracker.update(
            "INSERT DATA { GRAPH tracker:FileSystem { _:u a nfo:FileDataObject ; nie:url '%s' } }" % uri)

        pid = self.sandbox.session_bus.get_connection_unix_process_id_sync(
            'org.freedesktop.LocalSearch3')

        # Create a file on that url, see it fail
        with open(path, "w") as f:
            f.write(DEFAULT_TEXT)

        # Ensure the process is gone
        pid_exists = True
        attempts = 0
        while pid_exists:
            if attempts == 10:
                raise RuntimeError("Took too long for indexer to die")

            try:
                time.sleep(1)
                program = os.readlink("/proc/" + str(pid) + "/exe")
                pid_exists = os.path.basename(program) == "localsearch-3"
                if pid_exists:
                    attempts += 1
            except:
                pid_exists = False

        # Restart the indexer
        with self.await_document_inserted(path) as resource:
            conn = self.miner_fs.get_sparql_connection()

        result = self.__get_text_documents()
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/test.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/dir2/file3.txt"), unpacked_result)
        self.assertEqual(len(result), 4)

class IndexedFolderTest(fixtures.TrackerMinerTest):
    """
    Tests handling of data across multiple data sources
    """

    def setUp(self):
        # We must create the test data before the miner does its
        # initial crawl, or it may miss some files due
        # https://gitlab.gnome.org/GNOME/tracker-miners/issues/79.
        monitored_files = self.create_test_data()

        try:
            # Start the miner.
            fixtures.TrackerMinerTest.setUp(self)

            for tf in monitored_files:
                url = self.uri(tf)
                self.tracker.ensure_resource(
                    fixtures.DOCUMENTS_GRAPH,
                    f"a nfo:Document ; nie:isStoredAs <{url}>",
                    timeout=cfg.AWAIT_TIMEOUT,
                )
        except Exception:
            cfg.remove_monitored_test_dir(self.workdir)
            raise

        logging.info("%s.setUp(): complete", self)

    def create_test_data(self):
        monitored_files = [
            "test-monitored/file1.txt",
            "test-monitored/dir1/file3.txt",
            "test-non-recursive/file2.txt",
        ]

        unmonitored_files = ["test-no-monitored/file0.txt"]

        for tf in monitored_files:
            testfile = self.path(tf)
            os.makedirs(os.path.dirname(testfile), exist_ok=True)
            with open(testfile, "w") as f:
                f.write(DEFAULT_TEXT)

        return monitored_files

    def __get_text_documents(self):
        return self.tracker.query(
            """
          SELECT DISTINCT ?url WHERE {
              ?u a nfo:TextDocument ;
                 nie:isStoredAs/nie:url ?url.
          }
          """
        )

    def __get_index_folder(self, filepath):
        result = self.tracker.query(
            """
          SELECT DISTINCT ?p WHERE {
              ?u a nfo:FileDataObject ;
                 nie:url \"%s\" ;
                 nie:dataSource/nie:isStoredAs ?p
          }
          """
            % (self.uri(filepath))
        )
        self.assertEqual(len(result), 1)
        return result[0][0]

    """
    Move a file between indexed folders and check the data source changed
    """
    def test_01_move_between_indexed_folders(self):
        """
        Move a file between indexed folders and check the data source changed
        """
        result = self.__get_text_documents()
        self.assertEqual(len(result), 3)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file3.txt"), unpacked_result)
        self.assertIn(self.uri("test-non-recursive/file2.txt"), unpacked_result)

        source = os.path.join(self.workdir, "test-non-recursive", "file2.txt")
        dest = os.path.join(self.workdir, "test-monitored", "file2.txt")

        datasource1 = self.__get_index_folder(source);
        self.assertEqual(datasource1, self.uri("test-non-recursive"))

        with self.await_document_inserted(dest) as resource:
            shutil.move(source, dest)

        result = self.__get_text_documents()
        self.assertEqual(len(result), 3)
        unpacked_result = [r[0] for r in result]
        self.assertIn(self.uri("test-monitored/file1.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/file2.txt"), unpacked_result)
        self.assertIn(self.uri("test-monitored/dir1/file3.txt"), unpacked_result)

        # The file was moved to test-monitored
        datasource2 = self.__get_index_folder(dest);
        self.assertEqual(datasource2, self.uri("test-monitored"))
        self.assertNotEqual(datasource1, datasource2)


if __name__ == "__main__":
    fixtures.tracker_test_main()
