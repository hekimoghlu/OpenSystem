/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
 * Background:  Original tar file format did not use its linkflag to
 * specify directories.  Instead regular files simply have a slash
 * appended to their names.  No data blocks follow directories in
 * archives.  This means that a possibly specified file size must not
 * be used to determine the amount of data blocks to skip.
 */

static void
test_compat_tar_directory_1(void)
{
	char name[] = "test_compat_tar_directory_1.tar";
	struct archive_entry *ae;
	struct archive *a;

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	extract_reference_file(name);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, name, 10240));

	/* Read first entry, which is a directory in a regular file header. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("directory1/", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
	assertEqualInt(1, archive_entry_size(ae));

	/* Read second entry, which is a ustar directory entry. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("directory2/", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
	assertEqualInt(0, archive_entry_size(ae));

	/* Verify the end-of-archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify that the format detection worked. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_NONE);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR);

	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_compat_tar_directory)
{
	test_compat_tar_directory_1();
}


