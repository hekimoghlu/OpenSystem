/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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

static char buff[1000000];
static char buff2[100000];

DEFINE_TEST(test_read_truncated)
{
	struct archive_entry *ae;
	struct archive *a;
	unsigned int i;
	size_t used;

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_open_memory(a, buff, sizeof(buff), &used));

	/*
	 * Write a file to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "file");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	fill_with_pseudorandom_data(buff2, sizeof(buff2));
	archive_entry_set_size(ae, sizeof(buff2));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, sizeof(buff2), archive_write_data(a, buff2, sizeof(buff2)));

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Now, read back a truncated version of the archive and
	 * verify that we get an appropriate error. */
	for (i = 1; i < used + 100; i += 100) {
		assert((a = archive_read_new()) != NULL);
		assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
		assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
		if (i < 512) {
			assertEqualIntA(a, ARCHIVE_FATAL, archive_read_open_memory(a, buff, i));
			goto wrap_up;
		} else {
			assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, i));
		}
		assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));

		if (i < 512 + sizeof(buff2)) {
			assertEqualIntA(a, ARCHIVE_FATAL, archive_read_data(a, buff2, sizeof(buff2)));
			goto wrap_up;
		} else {
			assertEqualIntA(a, sizeof(buff2), archive_read_data(a, buff2, sizeof(buff2)));
		}

		/* Verify the end of the archive. */
		/* Archive must be long enough to capture a 512-byte
		 * block of zeroes after the entry.  (POSIX requires a
		 * second block of zeros to be written but libarchive
		 * does not return an error if it can't consume
		 * it.) */
		if (i < 512 + 512*((sizeof(buff2) + 511)/512) + 512) {
			assertEqualIntA(a, ARCHIVE_FATAL, archive_read_next_header(a, &ae));
		} else {
			assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
		}
	wrap_up:
		assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	}



	/* Same as above, except skip the body instead of reading it. */
	for (i = 1; i < used + 100; i += 100) {
		assert((a = archive_read_new()) != NULL);
		assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
		assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
		if (i < 512) {
			assertEqualIntA(a, ARCHIVE_FATAL, archive_read_open_memory(a, buff, i));
			goto wrap_up2;
		} else {
			assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, i));
		}
		assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));

		if (i < 512 + 512*((sizeof(buff2)+511)/512)) {
			assertEqualIntA(a, ARCHIVE_FATAL, archive_read_data_skip(a));
			goto wrap_up2;
		} else {
			assertEqualIntA(a, ARCHIVE_OK, archive_read_data_skip(a));
		}

		/* Verify the end of the archive. */
		/* Archive must be long enough to capture a 512-byte
		 * block of zeroes after the entry.  (POSIX requires a
		 * second block of zeros to be written but libarchive
		 * does not return an error if it can't consume
		 * it.) */
		if (i < 512 + 512*((sizeof(buff2) + 511)/512) + 512) {
			assertEqualIntA(a, ARCHIVE_FATAL, archive_read_next_header(a, &ae));
		} else {
			assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
		}
	wrap_up2:
		assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	}
}
