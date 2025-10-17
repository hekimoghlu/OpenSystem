/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

DEFINE_TEST(test_read_format_cpio_filename_eucJP_UTF8)
{
	const char *refname = "test_read_format_cpio_filename_eucjp.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read eucJP filename in en_US.UTF-8 with "hdrcharset=eucJP" option.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=eucJP")) {
		skipping("This system cannot convert character-set"
		    " from eucJP to UTF-8.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xe6\xbc\xa2\xe5\xad\x97.txt",
	    archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xe8\xa1\xa8.txt", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_eucJP)
{
	const char *refname = "test_read_format_cpio_filename_utf8_jp.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read UTF-8 filename in ja_JP.eucJP with "hdrcharset=UTF-8" option.
	 */
	if (NULL == setlocale(LC_ALL, "ja_JP.eucJP")) {
		skipping("ja_JP.eucJP locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=UTF-8")) {
		skipping("This system cannot convert character-set"
		    " from UTF-8 to eucJP.");
		goto cleanup;
	}

	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("\xb4\xc1\xbb\xfa.txt", archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("\xc9\xbd.txt", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_UTF8_jp)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	/* Since we explicitly DON'T set hdrcharset=UTF-8 below */
	skipping("Windows defaults to OEMCP, not UTF-8");
	return;
#else
	const char *refname = "test_read_format_cpio_filename_utf8_jp.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read UTF-8 filename in en_US.UTF-8 without "hdrcharset=UTF-8" option.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("\xe6\xbc\xa2\xe5\xad\x97.txt",
	    archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("\xe8\xa1\xa8.txt", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
#endif
}

DEFINE_TEST(test_read_format_cpio_filename_CP866_KOI8R)
{
	const char *refname = "test_read_format_cpio_filename_cp866.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read CP866 filename in ru_RU.KOI8-R with "hdrcharset=CP866" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia.20866") &&
	    NULL == setlocale(LC_ALL, "ru_RU.KOI8-R")) {
		skipping("ru_RU.KOI8-R locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP866")) {
		skipping("This system cannot convert character-set"
		    " from CP866 to KOI8-R.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xf0\xf2\xe9\xf7\xe5\xf4",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\xd2\xc9\xd7\xc5\xd4",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_CP866_UTF8)
{
	const char *refname = "test_read_format_cpio_filename_cp866.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read CP866 filename in en_US.UTF-8 with "hdrcharset=CP866" option.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP866")) {
		skipping("This system cannot convert character-set"
		    " from CP866 to UTF-8.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\x9f\xd0\xa0\xd0\x98\xd0\x92\xd0\x95\xd0\xa2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_KOI8R_CP866)
{
	const char *refname = "test_read_format_cpio_filename_koi8r.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read KOI8-R filename in ru_RU.CP866 with "hdrcharset=KOI8-R" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia.866") &&
	    NULL == setlocale(LC_ALL, "ru_RU.CP866")) {
		skipping("ru_RU.CP866 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=KOI8-R")) {
		skipping("This system cannot convert character-set"
		    " from KOI8-R to CP866.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\x8f\x90\x88\x82\x85\x92",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xaf\xe0\xa8\xa2\xa5\xe2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_KOI8R_UTF8)
{
	const char *refname = "test_read_format_cpio_filename_koi8r.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read KOI8-R filename in en_US.UTF-8 with "hdrcharset=KOI8-R" option.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=KOI8-R")) {
		skipping("This system cannot convert character-set"
		    " from KOI8-R to UTF-8.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\x9f\xd0\xa0\xd0\x98\xd0\x92\xd0\x95\xd0\xa2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_KOI8R)
{
	const char *refname = "test_read_format_cpio_filename_utf8_ru.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read UTF-8 filename in ru_RU.KOI8-R with "hdrcharset=UTF-8" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia.20866") &&
	    NULL == setlocale(LC_ALL, "ru_RU.KOI8-R")) {
		skipping("ru_RU.KOI8-R locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=UTF-8")) {
		skipping("This system cannot convert character-set"
		    " from UTF-8 to KOI8-R.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xf0\xf2\xe9\xf7\xe5\xf4",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\xd2\xc9\xd7\xc5\xd4",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_CP866)
{
	const char *refname = "test_read_format_cpio_filename_utf8_ru.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read UTF-8 filename in ru_RU.CP866 with "hdrcharset=UTF-8" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia.866") &&
		NULL == setlocale(LC_ALL, "ru_RU.CP866")) {
		skipping("ru_RU.CP866 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=UTF-8")) {
		skipping("This system cannot convert character-set"
		    " from UTF-8 to CP866.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\x8f\x90\x88\x82\x85\x92",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xaf\xe0\xa8\xa2\xa5\xe2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_UTF8_ru)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	/* Since we explicitly DON'T set hdrcharset=UTF-8 below */
	skipping("Windows defaults to OEMCP, not UTF-8");
	return;
#else
	const char *refname = "test_read_format_cpio_filename_utf8_ru.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read UTF-8 filename in en_US.UTF-8 without "hdrcharset=UTF-8" option.
	 */
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\x9f\xd0\xa0\xd0\x98\xd0\x92\xd0\x95\xd0\xa2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
#endif
}

DEFINE_TEST(test_read_format_cpio_filename_eucJP_CP932)
{
	const char *refname = "test_read_format_cpio_filename_eucjp.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read eucJP filename in CP932/SJIS with "hdrcharset=eucJP" option.
	 */
	if (NULL == setlocale(LC_ALL, "Japanese_Japan") &&
	    NULL == setlocale(LC_ALL, "ja_JP.SJIS")) {
		skipping("CP932 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=eucJP")) {
		skipping("This system cannot convert character-set"
		    " from eucJP.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\x8a\xbf\x8e\x9a.txt", archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\x95\x5c.txt", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_CP932)
{
	const char *refname = "test_read_format_cpio_filename_utf8_jp.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read UTF-8 filename in CP932/SJIS with "hdrcharset=UTF-8" option.
	 */
	if (NULL == setlocale(LC_ALL, "Japanese_Japan") &&
	    NULL == setlocale(LC_ALL, "ja_JP.SJIS")) {
		skipping("CP932 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=UTF-8")) {
		skipping("This system cannot convert character-set"
		    " from UTF-8 to CP932.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("\x8a\xbf\x8e\x9a.txt", archive_entry_pathname(ae));
	assertEqualInt(8, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG, archive_entry_filetype(ae));
	assertEqualString("\x95\x5c.txt", archive_entry_pathname(ae));
	assertEqualInt(4, archive_entry_size(ae));

	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_CP866_CP1251)
{
	const char *refname = "test_read_format_cpio_filename_cp866.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read CP866 filename in CP1251 with "hdrcharset=CP866" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia") &&
	    NULL == setlocale(LC_ALL, "ru_RU.CP1251")) {
		skipping("CP1251 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=CP866")) {
		skipping("This system cannot convert character-set"
		    " from CP866 to CP1251.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xcf\xd0\xc8\xc2\xc5\xd2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xef\xf0\xe8\xe2\xe5\xf2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

/*
 * This test only for Windows platform because other archiver
 * applications on Windows translate CP1251 filenames into CP866
 * filenames and store it in the cpio file and so we should read
 * it by default on Windows.
 */
DEFINE_TEST(test_read_format_cpio_filename_CP866_CP1251_win)
{
	const char *refname = "test_read_format_cpio_filename_cp866.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read CP866 filename in CP1251 without "hdrcharset=CP866" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia")) {
		skipping("Russian_Russia locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xcf\xd0\xc8\xc2\xc5\xd2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xef\xf0\xe8\xe2\xe5\xf2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_KOI8R_CP1251)
{
	const char *refname = "test_read_format_cpio_filename_koi8r.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read KOI8-R filename in CP1251 with "hdrcharset=KOI8-R" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia") &&
	    NULL == setlocale(LC_ALL, "ru_RU.CP1251")) {
		skipping("CP1251 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=KOI8-R")) {
		skipping("This system cannot convert character-set"
		    " from KOI8-R to CP1251.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xcf\xd0\xc8\xc2\xc5\xd2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xef\xf0\xe8\xe2\xe5\xf2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_cpio_filename_UTF8_CP1251)
{
	const char *refname = "test_read_format_cpio_filename_utf8_ru.cpio";
	struct archive *a;
	struct archive_entry *ae;

	/*
	 * Read KOI8-R filename in CP1251 with "hdrcharset=KOI8-R" option.
	 */
	if (NULL == setlocale(LC_ALL, "Russian_Russia") &&
	    NULL == setlocale(LC_ALL, "ru_RU.CP1251")) {
		skipping("CP1251 locale not available on this system.");
		return;
	}
	extract_reference_file(refname);

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	if (ARCHIVE_OK != archive_read_set_options(a, "hdrcharset=UTF-8")) {
		skipping("This system cannot convert character-set"
		    " from UTF-8 to CP1251.");
		goto cleanup;
	}
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, refname, 10240));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xcf\xd0\xc8\xc2\xc5\xd2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));

	/* Verify regular file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("\xef\xf0\xe8\xe2\xe5\xf2",
	    archive_entry_pathname(ae));
	assertEqualInt(6, archive_entry_size(ae));


	/* End of archive. */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));

	/* Verify archive format. */
	assertEqualIntA(a, ARCHIVE_FILTER_NONE, archive_filter_code(a, 0));
	assertEqualIntA(a, ARCHIVE_FORMAT_CPIO_POSIX, archive_format(a));

	/* Close the archive. */
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
cleanup:
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

