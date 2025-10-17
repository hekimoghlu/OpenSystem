/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
 * The data for this testcase was provided by Mike Qin <mikeandmore@gmail.com>
 * and created with Nero.
 */

DEFINE_TEST(test_read_format_isojoliet_versioned)
{
	const char *refname = "test_read_format_iso_joliet_by_nero.iso.Z";
	struct archive_entry *ae;
	struct archive *a;

	extract_reference_file(refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualInt(0, archive_read_support_filter_all(a));
	assertEqualInt(0, archive_read_support_format_all(a));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_option(a, "iso9660", "rockridge", NULL));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* First entry is '.' root directory. */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString(".", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));

	/* A directory. */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString("test", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR, archive_entry_filetype(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* A regular file which is called test.txt and has
	 * ;1 appended to it because apparently Nero always
	 * appends versions to all files in the joliet extension.
	 *
	 * We test to make sure the version has been stripped.
	 */
	assertEqualInt(0, archive_read_next_header(a, &ae));
	assertEqualString("test/test.txt",
	    archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* End of archive. */
	assertEqualInt(ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_COMPRESS);

	/* Close the archive. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

