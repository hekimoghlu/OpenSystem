/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

DEFINE_TEST(test_read_filter_lrzip)
{
	const char *name = "test_read_filter_lrzip.tar.lrz";
	/* lrzip tracks directories as files, ensure that we list everything */
	const char *n[] = {
		"d1/", "d1/f1", "d1/f2", "d1/f3", "f1", "f2", "f3", NULL };
	struct archive_entry *ae;
	struct archive *a;
	int i;

	if (!canLrzip()) {
		skipping("lrzip command-line program not found");
		return;
	}

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_WARN, archive_read_support_filter_lrzip(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	extract_reference_file(name);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, name, 200));

	/* Read entries, match up names with list above. */
	for (i = 0; i < 7; ++i) {
		failure("Could not read file %d (%s) from %s", i, n[i], name);
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_next_header(a, &ae));
		assertEqualString(n[i], archive_entry_pathname(ae));
	}

	/* Verify the end-of-archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify that the format detection worked. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_LRZIP);
	assertEqualString(archive_filter_name(a, 0), "lrzip");
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_GNUTAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
