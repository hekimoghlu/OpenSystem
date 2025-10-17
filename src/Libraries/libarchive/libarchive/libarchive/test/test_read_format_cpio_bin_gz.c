/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
31,139,8,0,244,'M','p','C',0,3,';','^','(',202,178,177,242,173,227,11,230,
23,204,'L',12,12,12,5,206,'_','|','A','4',3,131,30,195,241,'B',6,'8','`',
132,210,220,'`','2','$',200,209,211,199,'5','H','Q','Q',145,'a',20,12,'i',
0,0,170,199,228,195,0,2,0,0};

DEFINE_TEST(test_read_format_cpio_bin_gz)
{
	struct archive_entry *ae;
	struct archive *a;
	int r;

	assert((a = archive_read_new()) != NULL);
	assertEqualInt(ARCHIVE_OK, archive_read_support_filter_all(a));
	r = archive_read_support_filter_gzip(a);
	if (r == ARCHIVE_WARN) {
		skipping("gzip reading not fully supported on this platform");
		assertEqualInt(ARCHIVE_OK, archive_read_free(a));
		return;
	}
	failure("archive_read_support_filter_gzip");
	assertEqualInt(ARCHIVE_OK, r);
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualInt(ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_filter_code(a, 0),
	    ARCHIVE_FILTER_GZIP);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_CPIO_BIN_LE);
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}


