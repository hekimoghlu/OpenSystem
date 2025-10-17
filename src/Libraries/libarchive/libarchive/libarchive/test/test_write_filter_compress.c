/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
 * A basic exercise of compress reading and writing.
 *
 * TODO: Add a reference file and make sure we can decompress that.
 */

DEFINE_TEST(test_write_filter_compress)
{
	struct archive_entry *ae;
	struct archive* a;
	char *buff, *data;
	size_t buffsize, datasize;
	char path[16];
	size_t used;
	int i;

	buffsize = 1000000;
	assert(NULL != (buff = (char *)malloc(buffsize)));

	datasize = 10000;
	assert(NULL != (data = (char *)malloc(datasize)));
	memset(data, 0, datasize);

	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_format_ustar(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_add_filter_compress(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, buffsize, &used));

	for (i = 0; i < 100; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, path);
		archive_entry_set_size(ae, datasize);
		archive_entry_set_filetype(ae, AE_IFREG);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		assertEqualInt(datasize,
		    archive_write_data(a, data, datasize));
		archive_entry_free(ae);
	}

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Now, read the data back.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, used));


	for (i = 0; i < 100; i++) {
		snprintf(path, sizeof(path), "file%03d", i);
		if (!assertEqualInt(0, archive_read_next_header(a, &ae)))
			break;
		assertEqualString(path, archive_entry_pathname(ae));
		assertEqualInt((int)datasize, archive_entry_size(ae));
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	free(data);
	free(buff);
}
