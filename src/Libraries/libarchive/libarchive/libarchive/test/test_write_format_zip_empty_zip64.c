/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
/*
 * Development supported by Google Summer of Code 2008.
 */

#include "test.h"

DEFINE_TEST(test_write_format_zip_empty_zip64)
{
	struct archive *a;
	struct archive_entry *ae;
	char buff[256];
	size_t used;

	/* Zip format: Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_bytes_per_block(a, 1));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_bytes_in_last_block(a, 1));
	/* Force zip writer to use Zip64 extensions. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_option(a, "zip", "zip64", "1"));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_open_memory(a, buff, sizeof(buff), &used));

	/* Close out the archive without writing anything. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Verify the correct format for an empty Zip archive with Zip64 extensions forced. */
	assertEqualInt(used, 98);
	assertEqualMem(buff,
	    "PK\006\006" /* Zip64 end-of-central-directory record */
	    "\x2c\0\0\0\0\0\0\0"  /* 44 bytes long */
	    "\x2d\0" /* Created by Zip 4.5 */
	    "\x2d\0" /* Extract with Zip 4.5 or later */
	    "\0\0\0\0" /* This is disk #0 */
	    "\0\0\0\0" /* Central dir starts on disk #0 */
	    "\0\0\0\0\0\0\0\0" /* There are 0 entries on this disk ... */
	    "\0\0\0\0\0\0\0\0" /* ... out of 0 entries total ... */
	    "\0\0\0\0\0\0\0\0" /* ... requiring a total of 0 bytes. */
	    "\0\0\0\0\0\0\0\0" /* Directory starts at offset 0 */

	    "PK\006\007" /* Zip64 end-of-central-directory locator */
	    "\0\0\0\0" /* Zip64 EOCD record is on disk #0 .. */
	    "\0\0\0\0\0\0\0\0" /* .. at offset 0 .. */
	    "\1\0\0\0"  /* .. of 1 total disks. */

	    "PK\005\006" /* Regular Zip end-of-central-directory record */
	    "\0\0" /* This is disk #0 */
	    "\0\0" /* Central dir is on disk #0 */
	    "\0\0" /* There are 0 entries on this disk ... */
	    "\0\0" /* ... out of 0 total entries ... */
	    "\0\0\0\0" /* ... requiring a total of 0 bytes. */
	    "\0\0\0\0" /* Directory starts at offset 0. */
	    "\0\0" /* File comment is zero bytes long. */,
	    98);

	/* Verify that we read this kind of empty archive correctly. */
	/* Try with the standard memory reader, and with the test
	   memory reader with and without seek support. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, 98));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory(a, buff, 98, 1));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, buff, 98, 98));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));
}
