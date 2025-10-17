/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "test.h"

/*
 * Exercise symlink recreation.
 */
DEFINE_TEST(test_write_disk_symlink)
{
	static const char data[]="abcdefghijklmnopqrstuvwxyz";
	struct archive *ad;
	struct archive_entry *ae;
	int r;

	if (!canSymlink()) {
		skipping("Symlinks not supported");
		return;
	}

	/* Write entries to disk. */
	assert((ad = archive_write_disk_new()) != NULL);

	/*
	 * First, create a regular file then a symlink to that file.
	 */

	/* Regular file: link1a */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link1a");
	archive_entry_set_mode(ae, AE_IFREG | 0755);
	archive_entry_set_size(ae, sizeof(data));
	assertEqualIntA(ad, 0, archive_write_header(ad, ae));
	assertEqualInt(sizeof(data),
	    archive_write_data(ad, data, sizeof(data)));
	assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic Link: link1b -> link1a */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link1b");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_set_size(ae, 0);
	archive_entry_copy_symlink(ae, "link1a");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/*
	 * We should be able to do this in the other order as well,
	 * of course.
	 */

	/* Symbolic link: link2b -> link2a */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link2b");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "link2a");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN) {
		assertEqualInt(ARCHIVE_WARN,
		    archive_write_data(ad, data, sizeof(data)));
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	}
	archive_entry_free(ae);

	/* File: link2a */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "link2a");
	archive_entry_set_mode(ae, AE_IFREG | 0755);
	archive_entry_set_size(ae, sizeof(data));
	assertEqualIntA(ad, 0, archive_write_header(ad, ae));
	assertEqualInt(sizeof(data),
	    archive_write_data(ad, data, sizeof(data)));
	assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: dot -> . */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dot");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, ".");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: dotdot -> .. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dotdot");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "..");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: slash -> / */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "slash");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "/");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: sldot -> /. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "sldot");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "/.");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: sldotdot -> /.. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "sldotdot");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "/..");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Dir: d1 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1");
	archive_entry_set_mode(ae, AE_IFDIR | 0777);
	archive_entry_unset_size(ae);
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: d1nosl -> d1 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1nosl");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "d1");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: d1slash -> d1/ */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1slash");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "d1/");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: d1sldot -> d1/. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1sldot");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "d1/.");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: d1slddot -> d1/.. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1slddot");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "d1/..");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: d1dir -> d1 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1dir");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_set_symlink_type(ae, AE_SYMLINK_TYPE_DIRECTORY);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "d1");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic link: d1file -> d1 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "d1file");
	archive_entry_set_mode(ae, AE_IFLNK | 0642);
	archive_entry_set_symlink_type(ae, AE_SYMLINK_TYPE_FILE);
	archive_entry_unset_size(ae);
	archive_entry_copy_symlink(ae, "d1");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	assertEqualInt(ARCHIVE_OK, archive_write_free(ad));

	/* Test the entries on disk. */

	/* Test #1 */
	assertIsReg("link1a", -1);
	assertFileSize("link1a", sizeof(data));
	assertFileNLinks("link1a", 1);
	assertIsSymlink("link1b", "link1a", 0);

	/* Test #2: Should produce identical results to test #1 */
	assertIsReg("link2a", -1);
	assertFileSize("link2a", sizeof(data));
	assertFileNLinks("link2a", 1);
	assertIsSymlink("link2b", "link2a", 0);

	/* Test #3: Special symlinks */
	assertIsSymlink("dot", ".", 1);
	assertIsSymlink("dotdot", "..", 1);
	assertIsSymlink("slash", "/", 1);
	assertIsSymlink("sldot", "/.", 1);
	assertIsSymlink("sldotdot", "/..", 1);

	/* Test #4: Directory symlink mixed with . and .. */
	assertIsDir("d1", -1);
	/* On Windows, d1nosl should be a file symlink */
	assertIsSymlink("d1nosl", "d1", 0);
	assertIsSymlink("d1slash", "d1/", 1);
	assertIsSymlink("d1sldot", "d1/.", 1);
	assertIsSymlink("d1slddot", "d1/..", 1);

	/* Test #5: symlink_type is set */
	assertIsSymlink("d1dir", "d1", 1);
	assertIsSymlink("d1file", "d1", 0);
}
