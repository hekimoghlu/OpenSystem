/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
 * Test read/write of a 10M block of data in a single operation.
 * Uses an in-memory archive with a single 10M entry.  Exercises
 * archive_read_data() to ensure it can handle large blocks like
 * this and also exercises archive_read_data_into_fd() (which
 * had a bug relating to this, fixed in Nov 2006).
 */

#if defined(_WIN32) && !defined(__CYGWIN__)
#define open _open
#define close _close
#endif

static char buff1[11000000];
static char buff2[10000000];
static char buff3[10000000];

DEFINE_TEST(test_read_data_large)
{
	struct archive_entry *ae;
	struct archive *a;
	char tmpfilename[] = "largefile";
	int tmpfilefd;
	FILE *f;
	size_t used;

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertA(0 == archive_write_set_format_ustar(a));
	assertA(0 == archive_write_add_filter_none(a));
	assertA(0 == archive_write_open_memory(a, buff1, sizeof(buff1), &used));

	/*
	 * Write a file (with random contents) to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "file");
	archive_entry_set_mode(ae, S_IFREG | 0755);
	fill_with_pseudorandom_data(buff2, sizeof(buff2));
	archive_entry_set_size(ae, sizeof(buff2));
	assertA(0 == archive_write_header(a, ae));
	archive_entry_free(ae);
	assertA((int)sizeof(buff2) == archive_write_data(a, buff2, sizeof(buff2)));

	/* Close out the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Check that archive_read_data can handle 10*10^6 at a pop. */
	assert((a = archive_read_new()) != NULL);
	assertA(0 == archive_read_support_format_all(a));
	assertA(0 == archive_read_support_filter_all(a));
	assertA(0 == archive_read_open_memory(a, buff1, sizeof(buff1)));
	assertA(0 == archive_read_next_header(a, &ae));
	failure("Wrote 10MB, but didn't read the same amount");
	assertEqualIntA(a, sizeof(buff2),archive_read_data(a, buff3, sizeof(buff3)));
	failure("Read expected 10MB, but data read didn't match what was written");
	assertEqualMem(buff2, buff3, sizeof(buff3));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/* Check archive_read_data_into_fd */
	assert((a = archive_read_new()) != NULL);
	assertA(0 == archive_read_support_format_all(a));
	assertA(0 == archive_read_support_filter_all(a));
	assertA(0 == archive_read_open_memory(a, buff1, sizeof(buff1)));
	assertA(0 == archive_read_next_header(a, &ae));
#if defined(__BORLANDC__)
	tmpfilefd = open(tmpfilename, O_WRONLY | O_CREAT | O_BINARY);
#else
	tmpfilefd = open(tmpfilename, O_WRONLY | O_CREAT | O_BINARY, 0777);
#endif
	assert(tmpfilefd != 0);
	assertEqualIntA(a, 0, archive_read_data_into_fd(a, tmpfilefd));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	close(tmpfilefd);

	f = fopen(tmpfilename, "rb");
	assert(f != NULL);
	assertEqualInt(sizeof(buff3), fread(buff3, 1, sizeof(buff3), f));
	fclose(f);
	assertEqualMem(buff2, buff3, sizeof(buff3));
}
