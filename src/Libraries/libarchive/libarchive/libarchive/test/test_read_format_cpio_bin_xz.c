/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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

static unsigned char archive[] = {
 0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, 0x00, 0x04,
 0xe6, 0xd6, 0xb4, 0x46, 0x02, 0x00, 0x21, 0x01,
 0x16, 0x00, 0x00, 0x00, 0x74, 0x2f, 0xe5, 0xa3,
 0xe0, 0x01, 0xff, 0x00, 0x33, 0x5d, 0x00, 0x63,
 0x9c, 0x3e, 0xa0, 0x43, 0x7c, 0xe6, 0x5d, 0xdc,
 0xeb, 0x76, 0x1d, 0x4b, 0x1b, 0xe2, 0x9e, 0x43,
 0x95, 0x97, 0x60, 0x16, 0x36, 0xc6, 0xd1, 0x3f,
 0x68, 0xd1, 0x94, 0xf9, 0xee, 0x47, 0xbb, 0xc9,
 0xf3, 0xa2, 0x01, 0x2a, 0x2f, 0x2b, 0xb2, 0x23,
 0x5a, 0x06, 0x9c, 0xd0, 0x4a, 0x6b, 0x5b, 0x14,
 0xb4, 0x00, 0x00, 0x00, 0x91, 0x62, 0x1e, 0x15,
 0x04, 0x46, 0x6b, 0x4d, 0x00, 0x01, 0x4f, 0x80,
 0x04, 0x00, 0x00, 0x00, 0xa1, 0x4b, 0xdf, 0x03,
 0xb1, 0xc4, 0x67, 0xfb, 0x02, 0x00, 0x00, 0x00,
 0x00, 0x04, 0x59, 0x5a      
};

DEFINE_TEST(test_read_format_cpio_bin_xz)
{
	struct archive_entry *ae;
	struct archive *a;
	int r;

	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
	r = archive_read_support_filter_xz(a);
	if (r == ARCHIVE_WARN) {
		skipping("xz reading not fully supported on this platform");
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_XZ);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_CPIO_BIN_LE);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}

