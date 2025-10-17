/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
/*
 * Development supported by Google Summer of Code 2008.
 */

#include "test.h"

DEFINE_TEST(test_write_format_xar_empty)
{
	struct archive *a;
	struct archive_entry *ae;
	char buff[256];
	size_t used;

	/* Xar format: Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	if (archive_write_set_format_xar(a) != ARCHIVE_OK) {
		skipping("xar is not supported on this platform");
		assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_write_add_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_bytes_per_block(a, 1));
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_bytes_in_last_block(a, 1));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, sizeof(buff), &used));

	/* Add "." entry which must be ignored. */ 
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 0);
	archive_entry_set_ctime(ae, 4, 0);
	archive_entry_set_mtime(ae, 5, 0);
	archive_entry_copy_pathname(ae, ".");
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Add ".." entry which must be ignored. */ 
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 0);
	archive_entry_set_ctime(ae, 4, 0);
	archive_entry_set_mtime(ae, 5, 0);
	archive_entry_copy_pathname(ae, "..");
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Add "/" entry which must be ignored. */ 
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 0);
	archive_entry_set_ctime(ae, 4, 0);
	archive_entry_set_mtime(ae, 5, 0);
	archive_entry_copy_pathname(ae, "/");
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Add "../" entry which must be ignored. */ 
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 0);
	archive_entry_set_ctime(ae, 4, 0);
	archive_entry_set_mtime(ae, 5, 0);
	archive_entry_copy_pathname(ae, "../");
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Add "../../." entry which must be ignored. */ 
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 0);
	archive_entry_set_ctime(ae, 4, 0);
	archive_entry_set_mtime(ae, 5, 0);
	archive_entry_copy_pathname(ae, "../../.");
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Add "..//.././" entry which must be ignored. */ 
	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_atime(ae, 2, 0);
	archive_entry_set_ctime(ae, 4, 0);
	archive_entry_set_mtime(ae, 5, 0);
	archive_entry_copy_pathname(ae, "..//.././");
	archive_entry_set_mode(ae, S_IFDIR | 0755);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
	archive_entry_free(ae);

	/* Close out the archive without writing anything. */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_close(a));
	assertEqualInt(ARCHIVE_OK, archive_write_free(a));

	/* Verify the correct format for an empty Xar archive. */
	assertEqualInt(used, 0);
}
