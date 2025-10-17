/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
 * Issue 332:  Some epub files (which are really Zip archives) have
 * nonsense in the "external file attributes" field.
 */

DEFINE_TEST(test_read_format_zip_nofiletype)
{
	const char *refname = "test_read_format_zip_nofiletype.zip";
	char *p;
	size_t s;
	struct archive *a;
	struct archive_entry *ae;
	char data[16];

	extract_reference_file(refname);
	p = slurpfile(&s, "%s", refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip_seekable(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, p, s, 1));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("file1", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0644, archive_entry_mode(ae));
	assertEqualInt(6, archive_entry_size(ae));
	assertEqualIntA(a, 6, archive_read_data(a, data, 16));
	assertEqualMem(data, "file1\x0a", 6);

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("dir2/", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR | 0755, archive_entry_mode(ae));
	assertEqualInt(0, archive_entry_size(ae));
	assertEqualIntA(a, 0, archive_read_data(a, data, 16));

	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_free(a));
	free(p);
}
