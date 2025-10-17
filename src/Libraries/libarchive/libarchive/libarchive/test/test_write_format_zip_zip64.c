/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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

static void
verify_zip_filesize(uint64_t size, int expected)
{
	struct archive *a;
	struct archive_entry *ae;
	char buff[256];
	size_t used;

	/* Zip format: Create a new archive in memory. */
	assert((a = archive_write_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_write_set_format_zip(a));
	/* Disable Zip64 extensions. */
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_set_format_option(a, "zip", "zip64", NULL));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_write_open_memory(a, buff, sizeof(buff), &used));

	assert((ae = archive_entry_new()) != NULL);
	archive_entry_set_pathname(ae, "test");
	archive_entry_set_mode(ae, AE_IFREG | 0644);
	archive_entry_set_size(ae, size);
	assertEqualInt(expected, archive_write_header(a, ae));

	archive_entry_free(ae);

	/* Don't actually write 4GB! ;-) */
	assertEqualIntA(a, ARCHIVE_OK, archive_write_free(a));
}

DEFINE_TEST(test_write_format_zip_zip64_oversize)
{
	/* With Zip64 extensions disabled, we should be
	 * able to write a file with at most 4G-1 bytes. */

	/* Note:  Tar writer pads file to declared size when the file
	 * is closed.  If Zip writer is changed to behave the same
	 * way, it will be much harder to test the first case here. */
	verify_zip_filesize(0xffffffffLL, ARCHIVE_OK);

	verify_zip_filesize(0x100000000LL, ARCHIVE_FAILED);
}
