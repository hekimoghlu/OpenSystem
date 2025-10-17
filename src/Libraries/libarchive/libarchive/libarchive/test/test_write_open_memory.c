/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

/* Try to force archive_write_open_memory.c to write past the end of an array. */
static unsigned char buff[16384];

DEFINE_TEST(test_write_open_memory)
{
	unsigned int i;
	struct archive *a;
	struct archive_entry *ae;
	const char *name="/tmp/test";

	/* Create a simple archive_entry. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, name);
	archive_entry_set_mode(ae, S_IFREG);
	assertEqualString(archive_entry_pathname(ae), name);

	/* Try writing with different buffer sizes. */
	/* Make sure that we get failure on too-small buffers, success on
	 * large enough ones. */
	for (i = 100; i < 1600; i++) {
		size_t used;
		size_t blocksize = 94;
		assert((a = archive_write_new()) != NULL);
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_write_set_format_ustar(a));
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_write_set_bytes_in_last_block(a, 1));
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_write_set_bytes_per_block(a, (int)blocksize));
		buff[i] = 0xAE;
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_write_open_memory(a, buff, i, &used));
		/* If buffer is smaller than a tar header, this should fail. */
		if (i < (511/blocksize)*blocksize)
			assertEqualIntA(a, ARCHIVE_FATAL,
			    archive_write_header(a,ae));
		else
			assertEqualIntA(a, ARCHIVE_OK,
			    archive_write_header(a, ae));
		/* If buffer is smaller than a tar header plus 1024 byte
		 * end-of-archive marker, then this should fail. */
		failure("buffer size=%d\n", (int)i);
		if (i < 1536)
			assertEqualIntA(a, ARCHIVE_FATAL,
			    archive_write_close(a));
		else {
			assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
			assertEqualInt(used, archive_filter_bytes(a, -1));
			assertEqualInt(archive_filter_bytes(a, -1),
			    archive_filter_bytes(a, 0));
		}
		assertEqualInt(ARCHIVE_OK, archive_write_free(a));
		assertEqualInt(buff[i], 0xAE);
		assert(used <= i);
	}
	archive_entry_free(ae);
}
