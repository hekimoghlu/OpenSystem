/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
'B','Z','h','9','1','A','Y','&','S','Y',237,7,140,'W',0,0,27,251,144,208,
128,0,' ','@',1,'o',128,0,0,224,'"',30,0,0,'@',0,8,' ',0,'T','2',26,163,'&',
129,160,211,212,18,'I',169,234,13,168,26,6,150,'1',155,134,'p',8,173,3,183,
'J','S',26,20,'2',222,'b',240,160,'a','>',205,'f',29,170,227,'[',179,139,
'\'','L','o',211,':',178,'0',162,134,'*','>','8',24,153,230,147,'R','?',23,
'r','E','8','P',144,237,7,140,'W'};

DEFINE_TEST(test_read_format_tbz)
{
	struct archive_entry *ae;
	struct archive *a;
	int r;

	assert((a = archive_read_new()) != NULL);
	r = archive_read_support_filter_bzip2(a);
	if (r != ARCHIVE_OK) {
		skipping("Bzip2 support");
		archive_read_free(a);
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(1, archive_file_count(a));
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_BZIP2);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}


