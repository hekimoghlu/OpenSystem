/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
 * A basic exercise of lrzip reading and writing.
 */

DEFINE_TEST(test_write_filter_lrzip)
{
	struct archive_entry *ae;
	struct archive* a;
	char *buff, *data;
	size_t buffsize, datasize;
	char path[16];
	size_t used1, used2;
	int i;

	if (!canLrzip()) {
		skipping("lrzip command-line program not found");
		return;
	}

	buffsize = 2000000;
	assert(NULL != (buff = (char *)malloc(buffsize)));

	datasize = 10000;
	assert(NULL != (data = (char *)malloc(datasize)));
	memset(data, 0, datasize);

	/*
	 * Write 100 files and read them all back.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_gnutar(a));
	assertEqualIntA(a, ARCHIVE_WARN, archive_write_add_filter_lrzip(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_set_bytes_per_block(a, 10));
	assertEqualInt(ARCHIVE_FILTER_LRZIP, archive_filter_code(a, 0));
	assertEqualString("lrzip", archive_filter_name(a, 0));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_open_memory(a, buff, buffsize, &used1));
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_filetype(ae, AE_IFREG);
	archive_entry_set_size(ae, datasize);
	for (i = 0; i < 100; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		archive_entry_copy_pathname(ae, path);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		assertA(datasize
		    == (size_t)archive_write_data(a, data, datasize));
	}
	archive_entry_free(ae);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_WARN, archive_read_support_filter_lrzip(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_memory(a, buff, used1));
	for (i = 0; i < 100; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		if (!assertEqualInt(ARCHIVE_OK,
			archive_read_next_header(a, &ae)))
			break;
		assertEqualString(path, archive_entry_pathname(ae));
		assertEqualInt((int)datasize, archive_entry_size(ae));
	}
	assertEqualInt(ARCHIVE_FILTER_LRZIP, archive_filter_code(a, 0));
	assertEqualString("lrzip", archive_filter_name(a, 0));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Test various premature shutdown scenarios to make sure we
	 * don't crash or leak memory.
	 */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_WARN, archive_write_add_filter_lrzip(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_WARN, archive_write_add_filter_lrzip(a));
	assertEqualInt(ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_WARN, archive_write_add_filter_lrzip(a));
	assertEqualInt(ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_WARN, archive_write_add_filter_lrzip(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_open_memory(a, buff, buffsize, &used2));
	assertEqualInt(ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Clean up.
	 */
	free(data);
	free(buff);
}
