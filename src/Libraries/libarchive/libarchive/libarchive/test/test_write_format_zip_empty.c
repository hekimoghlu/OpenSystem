/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

DEFINE_TEST(test_write_format_zip_empty)
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
	assertEqualIntA(a, ARCHIVE_OK, archive_write_open_memory(a, buff, sizeof(buff), &used));

	/* Close out the archive without writing anything. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Verify the correct format for an empty Zip archive. */
	assertEqualInt(used, 22);
	assertEqualMem(buff,
	    "PK\005\006\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
	    22);

	/* Verify that we read this kind of empty archive correctly. */
	/* Try with the standard memory reader, and with the test
	   memory reader with and without seek support. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, 22));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory(a, buff, 22, 1));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, buff, 22, 22));
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));
}
