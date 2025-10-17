/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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

#include <locale.h>

static void
test_read_format_cab_filename_CP932_eucJP(const char *refname)
{
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read CAB filename in ja_JP.eucJP with "hdrcharset=CP932" option.
	 */
	if (NULL == setlocale(LC_ALL, "ja_JP.eucJP")) {
		skipping("ja_JP.eucJP locale not available on this system.");
		return;
	}

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP932")) {
		skipping("This system cannot convert character-set"
		    " from CP932 to eucJP.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString(
	    "\xc9\xbd\xa4\xc0\xa4\xe8\x2f\xb4\xc1\xbb\xfa\x2e\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(5, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString(
	    "\xc9\xbd\xa4\xc0\xa4\xe8\x2f\xb0\xec\xcd\xf7\xc9\xbd\x2e\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(5, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CAB, archive_format(a));

	/* Close the archive. */
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

static void
test_read_format_cab_filename_CP932_UTF8(const char *refname)
{
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read CAB filename in en_US.UTF-8 with "hdrcharset=CP932" option.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP932")) {
		skipping("This system cannot convert character-set"
		    " from CP932 to UTF-8.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
#if defined(__APPLE__)
	/* Compare NFD string. */
	assertEqualUTF8String(
	    "\xe8\xa1\xa8\xe3\x81\x9f\xe3\x82\x99\xe3\x82\x88\x2f"
	    "\xe6\xbc\xa2\xe5\xad\x97\x2e\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(5, archive_entry_size(ae));
#else
	/* Compare NFC string. */
	assertEqualUTF8String(
	    "\xe8\xa1\xa8\xe3\x81\xa0\xe3\x82\x88\x2f"
	    "\xe6\xbc\xa2\xe5\xad\x97\x2e\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(5, archive_entry_size(ae));
#endif
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
#if defined(__APPLE__)
	/* Compare NFD string. */
	assertEqualUTF8String(
	    "\xe8\xa1\xa8\xe3\x81\x9f\xe3\x82\x99\xe3\x82\x88\x2f"
	    "\xe4\xb8\x80\xe8\xa6\xa7\xe8\xa1\xa8\x2e\x74\x78\x74",
	    archive_entry_pathname(ae));
#else
	/* Compare NFC string. */
	assertEqualUTF8String(
	    "\xe8\xa1\xa8\xe3\x81\xa0\xe3\x82\x88\x2f"
	    "\xe4\xb8\x80\xe8\xa6\xa7\xe8\xa1\xa8\x2e\x74\x78\x74",
	    archive_entry_pathname(ae));
#endif
	assertEqualInt(5, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CAB, archive_format(a));

	/* Close the archive. */
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cab_filename)
{
	const char *refname = "test_read_format_cab_filename_cp932.cab";

	extract_reference_file(refname);
	test_read_format_cab_filename_CP932_eucJP(refname);
	test_read_format_cab_filename_CP932_UTF8(refname);
}
