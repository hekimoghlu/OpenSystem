/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

static char buff2[64];

/* Some names 1026 characters long */
static const char *longfilename = "abcdefghijklmnopqrstuvwxyz"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890";

static const char *longlinkname = "Xabcdefghijklmnopqrstuvwxyz"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890";

static const char *longhardlinkname = "Yabcdefghijklmnopqrstuvwxyz"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890"
    "12345678901234567890123456789012345678901234567890";


DEFINE_TEST(test_write_format_gnutar)
{
	size_t buffsize = 1000000;
	char *buff;
	struct archive_entry *ae;
	struct archive *a;
	size_t used;

	buff = malloc(buffsize); /* million bytes of work area */
	assert(buff != NULL);

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_gnutar(a));
	assertA(0 == archive_write_add_filter_none(a));
	assertA(0 == archive_write_open_memory(a, buff, buffsize, &used));

	/*
	 * "file" has a bunch of attributes and 8 bytes of data.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 20);
	archive_entry_set_birthtime(ae, 3, 30);
	archive_entry_set_ctime(ae, 4, 40);
	archive_entry_set_mtime(ae, 5, 50);
	archive_entry_copy_pathname(ae, "file");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 8);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 8, archive_write_data(a, "12345678", 9));

	/*
	 * A file with a very long name
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, longfilename);
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 8);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 8, archive_write_data(a, "abcdefgh", 9));

	/*
	 * A hardlink to the above file.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, longhardlinkname);
	archive_entry_copy_hardlink(ae, longfilename);
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 8);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/*
	 * A symlink to the above file.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, longlinkname);
	archive_entry_copy_symlink(ae, longfilename);
	archive_entry_set_mode(ae, AE_IFLNK | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/*
	 * A file with large UID/GID that overflow octal encoding.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "large_uid_gid");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 8);
	archive_entry_set_uid(ae, 123456789);
	archive_entry_set_gid(ae, 987654321);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 8, archive_write_data(a, "abcdefgh", 9));

	/* TODO: support GNU tar sparse format and test it here. */
	/* See test_write_format_pax for an example of testing sparse files. */

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));

	/*
	 * Some basic verification of the low-level format.
	 */

	/* Verify GNU tar magic/version fields */
	assertEqualMem(buff + 257, "ustar  \0", 8);

	assertEqualInt(15360, used);

	/*
	 *
	 * Now, read the data back.
	 *
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, 0, archive_read_support_format_all(a));
	assertEqualIntA(a, 0, archive_read_support_filter_all(a));
	assertEqualIntA(a, 0, archive_read_open_memory(a, buff, used));

	/*
	 * Read "file"
	 */
	assertEqualIntA(a, 0, archive_read_next_header(a, &ae));
	assert(!archive_entry_atime_is_set(ae));
	assert(!archive_entry_birthtime_is_set(ae));
	assert(!archive_entry_ctime_is_set(ae));
	assertEqualInt(5, archive_entry_mtime(ae));
	assertEqualInt(0, archive_entry_mtime_nsec(ae));
	assertEqualString("file", archive_entry_pathname(ae));
	assertEqualInt(S_IFREG | 0755, archive_entry_mode(ae));
	assertEqualInt(8, archive_entry_size(ae));
	assertEqualIntA(a, 8, archive_read_data(a, buff2, 10));
	assertEqualMem(buff2, "12345678", 8);

	/*
	 * Read file with very long name.
	 */
	assertEqualIntA(a, 0, archive_read_next_header(a, &ae));
	assertEqualString(longfilename, archive_entry_pathname(ae));
	assertEqualInt(S_IFREG | 0755, archive_entry_mode(ae));
	assertEqualInt(8, archive_entry_size(ae));
	assertEqualIntA(a, 8, archive_read_data(a, buff2, 10));
	assertEqualMem(buff2, "abcdefgh", 8);

	/*
	 * Read hardlink.
	 */
	assertEqualIntA(a, 0, archive_read_next_header(a, &ae));
	assertEqualString(longhardlinkname, archive_entry_pathname(ae));
	assertEqualString(longfilename, archive_entry_hardlink(ae));

	/*
	 * Read symlink.
	 */
	assertEqualIntA(a, 0, archive_read_next_header(a, &ae));
	assertEqualString(longlinkname, archive_entry_pathname(ae));
	assertEqualString(longfilename, archive_entry_symlink(ae));
	assertEqualInt(AE_IFLNK | 0755, archive_entry_mode(ae));

	/*
	 * Read file with large UID/GID.
	 */
	assertEqualIntA(a, 0, archive_read_next_header(a, &ae));
	assertEqualInt(123456789, archive_entry_uid(ae));
	assertEqualInt(987654321, archive_entry_gid(ae));
	assertEqualString("large_uid_gid", archive_entry_pathname(ae));
	assertEqualInt(S_IFREG | 0755, archive_entry_mode(ae));

	/*
	 * Verify the end of the archive.
	 */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_free(a));

	free(buff);
}
