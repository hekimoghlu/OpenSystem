/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
 * Sample file was created with:
 *    echo file0 | zip
 * Info-Zip uses Zip64 extensions in this case.
 */
static void
verify_file0_seek(struct archive *a)
{
	struct archive_entry *ae;

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("-", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0660, archive_entry_mode(ae));
	assertEqualInt(6, archive_entry_size(ae));
#ifdef HAVE_ZLIB_H
	{
		char data[16];
		assertEqualIntA(a, 6, archive_read_data(a, data, 16));
		assertEqualMem(data, "file0\x0a", 6);
	}
#endif
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_free(a));
}

static void
verify_file0_stream(struct archive *a, int size_known)
{
	struct archive_entry *ae;

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("-", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0664, archive_entry_mode(ae));
	if (size_known) {
		// zip64b has the uncompressed size at the beginning,
		// plus CRC and compressed size using length-at-end.
		assert(archive_entry_size_is_set(ae));
		assertEqualInt(6, archive_entry_size(ae));
	} else {
		// zip64a does not have a size at the beginning at all.
		assert(!archive_entry_size_is_set(ae));
	}
#ifdef HAVE_ZLIB_H
	{
		char data[16];
		assertEqualIntA(a, 6, archive_read_data(a, data, 16));
		assertEqualMem(data, "file0\x0a", 6);
	}
#endif
	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_free(a));
}

DEFINE_TEST(test_read_format_zip_zip64a)
{
	const char *refname = "test_read_format_zip_zip64a.zip";
	struct archive *a;
	char *p;
	size_t s;

	extract_reference_file(refname);
	p = slurpfile(&s, "%s", refname);

	/* First read with seeking. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip_seekable(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, p, s, 1));
	verify_file0_seek(a);

	/* Then read streaming. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip_streamable(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, p, s, 1));
	verify_file0_stream(a, 0);
	free(p);
}

DEFINE_TEST(test_read_format_zip_zip64b)
{
	const char *refname = "test_read_format_zip_zip64b.zip";
	struct archive *a;
	char *p;
	size_t s;

	extract_reference_file(refname);
	p = slurpfile(&s, "%s", refname);

	/* First read with seeking. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip_seekable(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, p, s, 1));
	verify_file0_seek(a);

	/* Then read streaming. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_zip_streamable(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory_seek(a, p, s, 1));
	verify_file0_stream(a, 1);
	free(p);
}

