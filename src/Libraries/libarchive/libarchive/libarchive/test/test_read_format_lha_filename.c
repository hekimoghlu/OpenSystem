/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
test_read_format_lha_filename_CP932_eucJP(const char *refname)
{
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read LHA filename in ja_JP.eucJP.
	 */
	if (NULL == setlocale(LC_ALL, "ja_JP.eucJP")) {
		skipping("ja_JP.eucJP locale not available on this system.");
		return;
	}

	/*
	 * Create a read object only for a test that platform support
	 * a character-set conversion because we can read a character-set
	 * of filenames from the header of an lha archive file and so we
	 * want to test that it works well. 
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP932")) {
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		skipping("This system cannot convert character-set"
		    " from CP932 to eucJP.");
		return;
	}
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xB4\xC1\xBB\xFA\x2E\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xC9\xBD\x2E\x74\x78\x74", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_LHA, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

static void
test_read_format_lha_filename_CP932_UTF8(const char *refname)
{
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read LHA filename in en_US.UTF-8.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
	/*
	 * Create a read object only for a test that platform support
	 * a character-set conversion because we can read a character-set
	 * of filenames from the header of an lha archive file and so we
	 * want to test that it works well. 
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP932")) {
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		skipping("This system cannot convert character-set"
		    " from CP932 to UTF-8.");
		return;
	}
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xE6\xBC\xA2\xE5\xAD\x97\x2E\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xE8\xA1\xA8\x2E\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_LHA, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

#if defined(_WIN32) && !defined(__CYGWIN__)
static void
test_read_format_lha_filename_CP932_Windows(const char *refname)
{
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read LHA filename in jpn on Windows.
	 */
	if (NULL == setlocale(LC_ALL, "jpn")) {
		skipping("jpn locale not available on this system.");
		return;
	}
	/*
	 * Create a read object only for a test that platform support
	 * a character-set conversion because we can read a character-set
	 * of filenames from the header of an lha archive file and so we
	 * want to test that it works well. 
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\x8A\xBF\x8E\x9A\x2E\x74\x78\x74",
	    archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\x95\x5C\x2E\x74\x78\x74", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_LHA, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
#else
/* Stub */
static void
test_read_format_lha_filename_CP932_Windows(const char *refname)
{
	(void)refname; /* UNUSED */
}
#endif

DEFINE_TEST(test_read_format_lha_filename)
{
	/* A sample file was created with LHA32.EXE through UNLHA.DLL. */
	const char *refname = "test_read_format_lha_filename_cp932.lzh";

	extract_reference_file(refname);

	test_read_format_lha_filename_CP932_eucJP(refname);
	test_read_format_lha_filename_CP932_UTF8(refname);
	test_read_format_lha_filename_CP932_Windows(refname);
}
