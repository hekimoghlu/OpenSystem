/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

DEFINE_TEST(test_write_format_mtree_absolute_path)
{
	struct archive_entry *ae;
	struct archive* a;
	size_t used;

	/* Create a mtree format archive. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_mtree(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, sizeof(buff)-1, &used));

	/* Write "." file.  */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, ".");
	archive_entry_set_mode(ae, AE_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Write "/file" file.  */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "/file");
	archive_entry_set_size(ae, 0);
	archive_entry_set_mode(ae, AE_IFREG | 0644);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Write "/dir" directory.  */
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_copy_pathname(ae, "/dir");
	archive_entry_set_mode(ae, AE_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
        assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/*
	 * Read the data and check it.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_open_memory(a, buff, used));

	/* Read "." file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	failure("The path should be just \".\"");
	assertEqualString(archive_entry_pathname(ae), ".");
	assertEqualInt(archive_entry_mode(ae), AE_IFDIR | 0755);

	/* Read "/file" file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	failure("The path should have \"./\" prefix");
	assertEqualString(archive_entry_pathname(ae), "./file");
	assertEqualInt(archive_entry_size(ae), 0);
	assertEqualInt(archive_entry_mode(ae), AE_IFREG | 0644);

	/* Read "/dir" file. */
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	failure("The path should have \"./\" prefix");
	assertEqualString(archive_entry_pathname(ae), "./dir");
	assertEqualInt(archive_entry_mode(ae), AE_IFDIR | 0755);

	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

