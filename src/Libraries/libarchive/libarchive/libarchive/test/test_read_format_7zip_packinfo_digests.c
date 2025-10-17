/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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

/* Read archive with digests in PackInfo */
DEFINE_TEST(test_read_format_7zip_packinfo_digests)
{
	struct archive_entry *ae;
	struct archive *a;
	char buff[4];
	const char *refname = "test_read_format_7zip_packinfo_digests.7z";

	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	if (ARCHIVE_OK != archive_read_support_filter_xz(a)) {
		skipping("7zip:lzma decoding is not supported on this "
		"platform");
	} else {
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_support_filter_all(a));
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_support_format_all(a));
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_open_filename(a, refname, 10240));

		/* Verify regular file1. */
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_next_header(a, &ae));
		assertEqualInt((AE_IFREG | 0644), archive_entry_mode(ae));
		assertEqualString("a.txt", archive_entry_pathname(ae));
		assertEqualInt(1576808819, archive_entry_mtime(ae));
		assertEqualInt(4, archive_entry_size(ae));
		assertEqualInt(archive_entry_is_encrypted(ae), 0);
		assertEqualIntA(a, archive_read_has_encrypted_entries(a), 0);
		assertEqualInt(4, archive_read_data(a, buff, sizeof(buff)));
		assertEqualMem(buff, "aaa\n", 4);

		/* Verify regular file2. */
		assertEqualIntA(a, ARCHIVE_OK,
		    archive_read_next_header(a, &ae));
		assertEqualInt((AE_IFREG | 0644), archive_entry_mode(ae));
		assertEqualString("b.txt", archive_entry_pathname(ae));
		assertEqualInt(1576808819, archive_entry_mtime(ae));
		assertEqualInt(4, archive_entry_size(ae));
		assertEqualInt(archive_entry_is_encrypted(ae), 0);
		assertEqualIntA(a, archive_read_has_encrypted_entries(a), 0);
		assertEqualInt(4, archive_read_data(a, buff, sizeof(buff)));
		assertEqualMem(buff, "bbb\n", 4);

		assertEqualInt(2, archive_file_count(a));

		/* End of archive. */
		assertEqualIntA(a, ARCHIVE_EOF,
		    archive_read_next_header(a, &ae));

		/* Verify archive format. */
		assertEqualIntA(a, ARCHIVE_FILTER_NONE,
		    archive_filter_code(a, 0));
		assertEqualIntA(a, ARCHIVE_FORMAT_7ZIP,
		    archive_format(a));

		/* Close the archive. */
		assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	}
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
