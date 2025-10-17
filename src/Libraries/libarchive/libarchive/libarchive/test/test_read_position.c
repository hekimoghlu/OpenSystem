/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

static unsigned char nulls[1000];
static unsigned char tmp[1000];
static unsigned char  buff[10000];
size_t data_sizes[] = {0, 5, 511, 512, 513};

static void verify_read_positions(struct archive *a);

static void
verify_read_positions(struct archive *a)
{
	struct archive_entry *ae;
	intmax_t read_position = 0;
	size_t j;

	/* Initial header position is zero. */
	assert(read_position == (intmax_t)archive_read_header_position(a));
	for (j = 0; j < sizeof(data_sizes)/sizeof(data_sizes[0]); ++j) {
		assertA(0 == archive_read_next_header(a, &ae));
		assertEqualInt(read_position,
		    (intmax_t)archive_read_header_position(a));
		/* Every other entry: read, then skip */
		if (j & 1)
			assertEqualInt(1,
			    archive_read_data(a, tmp, 1));
		assertA(0 == archive_read_data_skip(a));
		/* read_data_skip() doesn't change header_position */
		assertEqualInt(read_position,
		    (intmax_t)archive_read_header_position(a));

		read_position += 512; /* Size of header. */
		read_position += (data_sizes[j] + 511) & ~511;
	}

	assertA(1 == archive_read_next_header(a, &ae));
	assertEqualInt(read_position, (intmax_t)archive_read_header_position(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(read_position, (intmax_t)archive_read_header_position(a));
}

/* Check that header_position tracks correctly on read. */
DEFINE_TEST(test_read_position)
{
	struct archive *a;
	struct archive_entry *ae;
	size_t write_pos;
	size_t i;

	/* Sanity test */
	assert(sizeof(nulls) + 512 + 1024 <= sizeof(buff));

	/* Create an archive. */
	assert(NULL != (a = archive_write_new()));
	assertA(0 == archive_write_set_format_pax_restricted(a));
	assertA(0 == archive_write_set_bytes_per_block(a, 512));
	assertA(0 == archive_write_open_memory(a, buff, sizeof(buff), &write_pos));

	for (i = 0; i < sizeof(data_sizes)/sizeof(data_sizes[0]); ++i) {
		/* Create a simple archive_entry. */
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_set_pathname(ae, "testfile");
		archive_entry_set_mode(ae, S_IFREG);
		archive_entry_set_size(ae, data_sizes[i]);
		assertA(0 == archive_write_header(a, ae));
		archive_entry_free(ae);
		assertA(data_sizes[i]
		    == (size_t)archive_write_data(a, nulls, sizeof(nulls)));
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Read the archive back with a skip function. */
	assert(NULL != (a = archive_read_new()));
	assertA(0 == archive_read_support_format_tar(a));
	assertA(0 == read_open_memory(a, buff, sizeof(buff), 512));
	verify_read_positions(a);
	archive_read_free(a);

	/* Read the archive back without a skip function. */
	assert(NULL != (a = archive_read_new()));
	assertA(0 == archive_read_support_format_tar(a));
	assertA(0 == read_open_memory_minimal(a, buff, sizeof(buff), 512));
	verify_read_positions(a);
	archive_read_free(a);

}
