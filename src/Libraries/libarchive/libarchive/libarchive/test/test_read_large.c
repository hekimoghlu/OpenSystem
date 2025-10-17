/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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

static unsigned char testdata[10 * 1024 * 1024];
static unsigned char testdatacopy[10 * 1024 * 1024];
static unsigned char buff[11 * 1024 * 1024];

#if defined(_WIN32) && !defined(__CYGWIN__)
#define open _open
#define close _close
#endif

/* Check correct behavior on large reads. */
DEFINE_TEST(test_read_large)
{
	int tmpfilefd;
	char tmpfilename[] = "test-read_large.XXXXXX";
	size_t used;
	struct archive *a;
	struct archive_entry *entry;
	FILE *f;

	fill_with_pseudorandom_data(testdata, sizeof(testdata));

	assert(NULL != (a = archive_write_new()));
	assertA(0 == archive_write_set_format_ustar(a));
	assertA(0 == archive_write_open_memory(a, buff, sizeof(buff), &used));
	assert(NULL != (entry = archive_entry_new()));
	archive_entry_set_size(entry, sizeof(testdata));
	archive_entry_set_mode(entry, S_IFREG | 0777);
	archive_entry_set_pathname(entry, "test");
	assertA(0 == archive_write_header(a, entry));
	archive_entry_free(entry);
	assertA((int)sizeof(testdata) == archive_write_data(a, testdata, sizeof(testdata)));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert(NULL != (a = archive_read_new()));
	assertA(0 == archive_read_support_format_all(a));
	assertA(0 == archive_read_support_filter_all(a));
	assertA(0 == archive_read_open_memory(a, buff, sizeof(buff)));
	assertA(0 == archive_read_next_header(a, &entry));
	assertEqualIntA(a, sizeof(testdatacopy),
	    archive_read_data(a, testdatacopy, sizeof(testdatacopy)));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	assertEqualMem(testdata, testdatacopy, sizeof(testdata));


	assert(NULL != (a = archive_read_new()));
	assertA(0 == archive_read_support_format_all(a));
	assertA(0 == archive_read_support_filter_all(a));
	assertA(0 == archive_read_open_memory(a, buff, sizeof(buff)));
	assertA(0 == archive_read_next_header(a, &entry));
#if defined(__BORLANDC__)
	tmpfilefd = open(tmpfilename, O_WRONLY | O_CREAT | O_BINARY);
#else
	tmpfilefd = open(tmpfilename, O_WRONLY | O_CREAT | O_BINARY, 0755);
#endif
	assert(0 < tmpfilefd);
	assertA(0 == archive_read_data_into_fd(a, tmpfilefd));
	close(tmpfilefd);
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	f = fopen(tmpfilename, "rb");
	assertEqualInt(sizeof(testdatacopy),
	    fread(testdatacopy, 1, sizeof(testdatacopy), f));
	fclose(f);
	assertEqualMem(testdata, testdatacopy, sizeof(testdata));
}
