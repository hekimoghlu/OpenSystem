/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
 * Verify our ability to read sample files compatibly with lzop.
 *
 * In particular:
 *  * lzop will stop at the end of a stream if the following data
 *    doesn't start with a lzop signature.
 *
 */

DEFINE_TEST(test_compat_lzop)
{
	const char *sample1 = "test_compat_lzop_1.tar.lzo";
	const char *sample2 = "test_compat_lzop_2.tar.lzo";
	const char *sample3 = "test_compat_lzop_3.tar.lzo";
	struct archive_entry *ae;
	struct archive *a;
	int r;

	/*
	 * Test1: tar ball compressed in one block.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	r = archive_read_support_filter_lzop(a);
	if (r == ARCHIVE_WARN) {
		skipping("lzop reading not fully supported on this platform");
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	extract_reference_file(sample1);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, sample1, 2));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("INSTALL", archive_entry_pathname(ae));

	/* Verify the end-of-archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify that the format detection worked. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_LZOP);
	assertEqualString(archive_filter_name(a, 0), "lzop");
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Test2: tar ball compressed in multi blocks.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	extract_reference_file(sample2);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, sample2, 2));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("file1", archive_entry_pathname(ae));
	assertEqualInt(19, archive_entry_size(ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("file2", archive_entry_pathname(ae));
	assertEqualInt(262144, archive_entry_size(ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("file3", archive_entry_pathname(ae));
	assertEqualInt(19, archive_entry_size(ae));

	/* Verify the end-of-archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify that the format detection worked. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_LZOP);
	assertEqualString(archive_filter_name(a, 0), "lzop");
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Test3: tar ball compressed in one block with junk data.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	extract_reference_file(sample3);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, sample3, 2));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("INSTALL", archive_entry_pathname(ae));

	/* Verify the end-of-archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify that the format detection worked. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_LZOP);
	assertEqualString(archive_filter_name(a, 0), "lzop");
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
