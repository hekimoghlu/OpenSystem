/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

DEFINE_TEST(test_read_format_raw)
{
	char buff[512];
	struct archive_entry *ae;
	struct archive *a;
	const char *reffile1 = "test_read_format_raw.data";
	const char *reffile2 = "test_read_format_raw.data.Z";
	const char *reffile3 = "test_read_format_raw.bufr";
#ifdef HAVE_ZLIB_H
	const char *reffile4 = "test_read_format_raw.data.gz";
#endif

	/* First, try pulling data out of an uninterpretable file. */
	extract_reference_file(reffile1);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_raw(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, reffile1, 512));

	/* First (and only!) Entry */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("data", archive_entry_pathname(ae));
	/* Most fields should be unset (unknown) */
	assert(!archive_entry_size_is_set(ae));
	assert(!archive_entry_atime_is_set(ae));
	assert(!archive_entry_ctime_is_set(ae));
	assert(!archive_entry_mtime_is_set(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(4, archive_read_data(a, buff, 32));
	assertEqualMem(buff, "foo\n", 4);

	/* Test EOF */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));


	/* Second, try the same with a compressed file. */
	extract_reference_file(reffile2);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_by_code(a, ARCHIVE_FORMAT_RAW));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, reffile2, 1));

	/* First (and only!) Entry */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("data", archive_entry_pathname(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	/* Most fields should be unset (unknown) */
	assert(!archive_entry_size_is_set(ae));
	assert(!archive_entry_atime_is_set(ae));
	assert(!archive_entry_ctime_is_set(ae));
	assert(!archive_entry_mtime_is_set(ae));
	assertEqualInt(4, archive_read_data(a, buff, 32));
	assertEqualMem(buff, "foo\n", 4);

	/* Test EOF */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));


	/* Third, try with file which fooled us in past - appeared to be tar. */
	extract_reference_file(reffile3);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_set_format(a, ARCHIVE_FORMAT_RAW));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, reffile3, 1));

	/* First (and only!) Entry */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("data", archive_entry_pathname(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	/* Most fields should be unset (unknown) */
	assert(!archive_entry_size_is_set(ae));
	assert(!archive_entry_atime_is_set(ae));
	assert(!archive_entry_ctime_is_set(ae));
	assert(!archive_entry_mtime_is_set(ae));

	/* Test EOF */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

#ifdef HAVE_ZLIB_H
	/* Fourth, try with gzip which has metadata. */
	extract_reference_file(reffile4);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_raw(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_filename(a, reffile4, 1));

	/* First (and only!) Entry */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("test-file-name.data", archive_entry_pathname(ae));
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assert(archive_entry_mtime_is_set(ae));
	assertEqualIntA(a, archive_entry_mtime(ae), 0x5cbafd25);
	/* Most fields should be unset (unknown) */
	assert(!archive_entry_size_is_set(ae));
	assert(!archive_entry_atime_is_set(ae));
	assert(!archive_entry_ctime_is_set(ae));

	/* Test EOF */
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
#endif
}
