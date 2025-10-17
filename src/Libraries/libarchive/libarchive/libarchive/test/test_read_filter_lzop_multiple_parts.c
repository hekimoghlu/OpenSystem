/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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

DEFINE_TEST(test_read_filter_lzop_multiple_parts)
{
	const char *reference = "test_read_filter_lzop_multiple_parts.tar.lzo";
	struct archive_entry *ae;
	struct archive *a;
	int r;

	extract_reference_file(reference);
	assert((a = archive_read_new()) != NULL);
	r = archive_read_support_filter_lzop(a);
	if (r != ARCHIVE_OK) {
		if (!canLzop()) {
			skipping("lzop compression is not supported "
			    "on this platform");
		} else if (r == ARCHIVE_WARN) {
			skipping("lzop multiple parts decoding is not "
			    "supported via external program");

		} else
			assertEqualIntA(a, ARCHIVE_OK, r);
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_read_open_filename(a, reference, 10240));

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
	assertEqualInt(archive_filter_count(a), 2);
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_LZOP);
	assertEqualString(archive_filter_name(a, 0), "lzop");
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
