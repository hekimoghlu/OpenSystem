/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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

DEFINE_TEST(test_empty_write)
{
	char buff[32768];
	struct archive_entry *ae;
	struct archive *a;
	size_t used;
	int r;

	/*
	 * Exercise a zero-byte write to a gzip-compressed archive.
	 */

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_ustar(a));
	r = archive_write_add_filter_gzip(a);
	if (r != ARCHIVE_OK && !canGzip()) {
		skipping("Empty write to gzip-compressed archive");
	} else {
		if (r != ARCHIVE_OK && canGzip())
			assertEqualIntA(a, ARCHIVE_WARN, r);
		else
			assertEqualIntA(a, ARCHIVE_OK, r);
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_write_open_memory(a, buff, sizeof(buff), &used));
		/* Write a file to it. */
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, "file");
		archive_entry_set_mode(ae, S_IFREG | 0755);
		archive_entry_set_size(ae, 0);
		assertA(0 == archive_write_header(a, ae));
		archive_entry_free(ae);

		/* THE TEST: write zero bytes to this entry. */
		/* This used to crash. */
		assertEqualIntA(a, 0, archive_write_data(a, "", 0));

		/* Close out the archive. */
		assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
		assertEqualInt(ARCHIVE_OK, archive_write_free(a));
	}

	/*
	 * Again, with bzip2 compression.
	 */

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_ustar(a));
	r = archive_write_add_filter_bzip2(a);
	if (r != ARCHIVE_OK && !canBzip2()) {
		skipping("Empty write to bzip2-compressed archive");
	} else {
		if (r != ARCHIVE_OK && canBzip2())
			assertEqualIntA(a, ARCHIVE_WARN, r);
		else
			assertEqualIntA(a, ARCHIVE_OK, r);
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_write_open_memory(a, buff, sizeof(buff), &used));
		/* Write a file to it. */
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, "file");
		archive_entry_set_mode(ae, S_IFREG | 0755);
		archive_entry_set_size(ae, 0);
		assertA(0 == archive_write_header(a, ae));
		archive_entry_free(ae);

		/* THE TEST: write zero bytes to this entry. */
		assertEqualIntA(a, 0, archive_write_data(a, "", 0));

		/* Close out the archive. */
		assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
		assertEqualInt(ARCHIVE_OK, archive_write_free(a));
	}

	/*
	 * For good measure, one more time with no compression.
	 */

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_ustar(a));
	assertA(0 == archive_write_add_filter_none(a));
	assertA(0 == archive_write_open_memory(a, buff, sizeof(buff), &used));
	/* Write a file to it. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "file");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	archive_entry_set_size(ae, 0);
	assertA(0 == archive_write_header(a, ae));
	archive_entry_free(ae);

	/* THE TEST: write zero bytes to this entry. */
	assertEqualIntA(a, 0, archive_write_data(a, "", 0));

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));
}
