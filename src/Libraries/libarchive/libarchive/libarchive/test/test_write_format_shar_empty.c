/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
 * Check that an "empty" shar archive is correctly created as an empty file.
 */

DEFINE_TEST(test_write_format_shar_empty)
{
	struct archive *a;
	char buff[2048];
	size_t used;

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_shar(a));
	assertA(0 == archive_write_add_filter_none(a));
	/* 1-byte block size ensures we see only the required bytes. */
	/* We're not testing the padding here. */
	assertA(0 == archive_write_set_bytes_per_block(a, 1));
	assertA(0 == archive_write_set_bytes_in_last_block(a, 1));
	assertA(0 == archive_write_open_memory(a, buff, sizeof(buff), &used));

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	failure("Empty shar archive should be exactly 0 bytes, was %zu.", used);
	assert(used == 0);
}
