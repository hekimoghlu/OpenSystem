/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
 * Test archive contains the following entries with only MSDOS attributes:
 *   'abc' -- zero-length file
 *   'def' -- directory without trailing slash and without streaming extension
 *   'def/foo' -- file in def
 *   'ghi/' -- directory with trailing slash and without streaming extension
 *   'jkl'  -- directory without trailing slash and with streaming extension
 *   'mno/' -- directory with trailing slash and streaming extension
 *
 * Seeking reader should identify all of these correctly using the
 * central directory information.
 * Streaming reader should correctly identify everything except 'def';
 * since the standard Zip local file header does not include any file
 * type information, it will be mis-identified as a zero-length file.
 */

static void verify(struct archive *a, int streaming) {
	struct archive_entry *ae;

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("abc", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0664, archive_entry_mode(ae));

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	if (streaming) {
		/* Streaming reader has no basis for making this a dir */
		assertEqualString("def", archive_entry_pathname(ae));
		assertEqualInt(AE_IFREG | 0664, archive_entry_mode(ae));
	} else {
		/* Since 'def' is a dir, '/' should be added */
		assertEqualString("def/", archive_entry_pathname(ae));
		assertEqualInt(AE_IFDIR | 0775, archive_entry_mode(ae));
	}

	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("def/foo", archive_entry_pathname(ae));
	assertEqualInt(AE_IFREG | 0664, archive_entry_mode(ae));

	/* Streaming reader can tell this is a dir because it ends in '/' */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("ghi/", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR | 0775, archive_entry_mode(ae));

	/* Streaming reader can tell this is a dir because it has xl
	 * extension */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	/* '/' gets added because this is a dir */
	assertEqualString("jkl/", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR | 0775, archive_entry_mode(ae));

	/* Streaming reader can tell this is a dir because it ends in
	 * '/' and has xl extension */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualString("mno/", archive_entry_pathname(ae));
	assertEqualInt(AE_IFDIR | 0775, archive_entry_mode(ae));

	assertEqualIntA(a, ARCHIVE_EOF, archive_read_next_header(a, &ae));
}

DEFINE_TEST(test_read_format_zip_msdos)
{
	const char *refname = "test_read_format_zip_msdos.zip";
	struct archive *a;
	char *p;
	size_t s;

	extract_reference_file(refname);

	/* Verify with seeking reader. */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, refname, 17));
	verify(a, 0);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/* Verify with streaming reader. */
	p = slurpfile(&s, "%s", refname);
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, read_open_memory(a, p, s, 31));
	verify(a, 1);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
	
	free(p);
}
