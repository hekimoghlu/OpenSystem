/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

static void test_read(struct archive *a, char *buff, size_t used, char *filedata)
{
	struct archive_entry *ae;
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, used));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualIntA(a, 9, archive_read_data(a, filedata, 10));
	assertEqualMem(filedata, "12345678", 9);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_write_format_warc)
{
	char filedata[64];
	struct archive *a;
	struct archive_entry *ae;
	const size_t bsiz = 2048U;
	char *buff;
	size_t used;

	buff = malloc(bsiz);

	/* Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_warc(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, bsiz, &used));

	/*
	 * Write a file to it.
	 */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, "test");
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_size(ae, 9);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 9, archive_write_data(a, "12345678", 9));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Read from it.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_warc(a));
	test_read(a, buff, used, filedata);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_by_code(a, ARCHIVE_FORMAT_WARC));
	test_read(a, buff, used, filedata);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_set_format(a, ARCHIVE_FORMAT_WARC));
	test_read(a, buff, used, filedata);

	/* Create a new archive */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_warc(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_open_memory(a, buff, bsiz, &used));

	/* write first file: that should succeed */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, "test");
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_size(ae, 9);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 9, archive_write_data(a, "12345678", 9));

	/* write second file: should succeed as well */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, "test2");
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_size(ae, 9);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);
	assertEqualIntA(a, 9, archive_write_data(a, "12345678", 9));

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Create a new archive */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_warc(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_open_memory(a, buff, bsiz, &used));

	/* write a directory: this should fail */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "dir");
	archive_entry_set_filetype(ae, AE_IFDIR);
	archive_entry_set_size(ae, 512);
	assertEqualIntA(a, ARCHIVE_FAILED, archive_write_header(a, ae));
	archive_entry_free(ae);

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* test whether last archive is indeed empty */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, used));

	/* Test EOF */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	free(buff);
}
