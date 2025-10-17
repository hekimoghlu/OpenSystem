/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

DEFINE_TEST(test_read_filter_lzop)
{
	const char *reference = "test_read_filter_lzop.tar.lzo";
	/* lrzip tracks directories as files, ensure that we list everything */
	const char *n[] = {
		"d1/", "d1/f2", "d1/f3", "d1/f1", "f1", "f2", "f3", NULL };
	struct archive_entry *ae;
	struct archive *a;
	int i, r;

	extract_reference_file(reference);
	assert((a = archive_read_new()) != NULL);
	r = archive_read_support_filter_lzop(a);
	if (r != ARCHIVE_OK) {
		if (!canLzop()) {
			assertEqualInt(ARCHIVE_OK, archive_read_free(a));
			skipping("lzop compression is not supported "
			    "on this platform");
			return;
		} else if (r != ARCHIVE_WARN) {
			assertEqualIntA(a, ARCHIVE_OK, r);
			assertEqualInt(ARCHIVE_OK, archive_read_free(a));
			return;
		}
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_read_open_filename(a, reference, 10240));

	/* Read entries, match up names with list above. */
	for (i = 0; n[i] != NULL; ++i) {
		failure("Could not read file %d (%s) from %s",
			i, n[i], reference);
		assertEqualIntA(a, ARCHIVE_OK,
			archive_read_next_header(a, &ae));
		assertEqualString(n[i], archive_entry_pathname(ae));
	}

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
