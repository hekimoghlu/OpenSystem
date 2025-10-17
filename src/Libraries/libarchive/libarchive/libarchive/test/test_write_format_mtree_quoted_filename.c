/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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

static char buff[4096];

static const char image [] = {
"#mtree\n"
"./a\\040!$\\043&\\075_^z\\177~ mode=644 type=file\n"
};


DEFINE_TEST(test_write_format_mtree_quoted_filename)
{
	struct archive_entry *ae;
	struct archive* a;
	size_t used;

	/* Create a mtree format archive. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_mtree(a));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_set_format_option(a, NULL, "all", NULL));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_set_format_option(a, NULL, "type", "1"));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_set_format_option(a, NULL, "mode", "1"));
	assertEqualIntA(a, ARCHIVE_OK,
		archive_write_open_memory(a, buff, sizeof(buff)-1, &used));

	/* Write entry which has #, = , \ and DEL(0177) in the filename. */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_mode(ae, AE_IFREG | 0644);
	assertEqualInt(AE_IFREG | 0644, archive_entry_mode(ae));
	archive_entry_copy_pathname(ae, "./a !$#&=_^z\177~");
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
        assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	buff[used] = '\0';
	failure("#, = and \\ in the filename should be quoted");
	assertEqualString(buff, image);

	/*
	 * Read the data and check it.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, used));

	/* Read entry */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(AE_IFREG | 0644, archive_entry_mode(ae));
	assertEqualString("./a !$#&=_^z\177~", archive_entry_pathname(ae));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

