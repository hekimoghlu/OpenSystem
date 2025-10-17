/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
 * Test fixup entries don't follow symlinks
 */
DEFINE_TEST(test_write_disk_fixup)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	skipping("Skipping test on Windows");
#else
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
	 * Create a file
	 */
	assertMakeFile("file", 0600, "a");

	/*
	 * Create a directory
	 */
	assertMakeDir("dir", 0700);

	/*
	 * Create a directory and a symlink with the same name
	 */

	/* Directory: dir1 */
        assert((ae = archive_entry_new()) != NULL);
        archive_entry_copy_pathname(ae, "dir1/");
        archive_entry_set_mode(ae, AE_IFDIR | 0555);
	assertEqualIntA(ad, 0, archive_write_header(ad, ae));
	assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
        archive_entry_free(ae);

	/* Directory: dir2 */
        assert((ae = archive_entry_new()) != NULL);
        archive_entry_copy_pathname(ae, "dir2/");
        archive_entry_set_mode(ae, AE_IFDIR | 0555);
	assertEqualIntA(ad, 0, archive_write_header(ad, ae));
	assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
        archive_entry_free(ae);

	/* Symbolic Link: dir1 -> dir */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dir1");
	archive_entry_set_mode(ae, AE_IFLNK | 0777);
	archive_entry_set_size(ae, 0);
	archive_entry_copy_symlink(ae, "dir");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	/* Symbolic Link: dir2 -> file */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dir2");
	archive_entry_set_mode(ae, AE_IFLNK | 0777);
	archive_entry_set_size(ae, 0);
	archive_entry_copy_symlink(ae, "file");
	assertEqualIntA(ad, 0, r = archive_write_header(ad, ae));
	if (r >= ARCHIVE_WARN)
		assertEqualIntA(ad, 0, archive_write_finish_entry(ad));
	archive_entry_free(ae);

	assertEqualInt(ARCHIVE_OK, archive_write_free(ad));

	/* Test the entries on disk. */
	assertIsSymlink("dir1", "dir", 0);
	assertIsSymlink("dir2", "file", 0);
	assertFileMode("dir", 0700);
	assertFileMode("file", 0600);
#endif
}
