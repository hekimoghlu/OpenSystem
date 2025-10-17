/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
31,157,144,'0','n',4,132,'!',3,6,140,26,'8','n',228,16,19,195,160,'A',26,
'1',202,144,'q','h','p','F',25,28,20,'a','X',196,152,145,' ',141,25,2,'k',
192,160,'A',163,163,201,135,29,'c',136,'<',201,'2','c','A',147,'.',0,12,20,
248,178,165,205,155,20,27,226,220,201,243,166,152,147,'T',164,4,'I',194,164,
136,148,16,'H',1,'(',']',202,180,169,211,167,'P',163,'J',157,'J',181,170,
213,171,'X',179,'j',221,202,181,171,215,175,'L',1};

DEFINE_TEST(test_read_format_cpio_svr4c_Z)
{
	struct archive_entry *ae;
	struct archive *a;
/*	printf("Archive address: start=%X, end=%X\n", archive, archive+sizeof(archive)); */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_filter_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_next_header(a, &ae));
	failure("archive_filter_name(a, 0)=\"%s\"",
	    archive_filter_name(a, 0));
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_COMPRESS);
	failure("archive_format_name(a)=\"%s\"", archive_format_name(a));
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_CPIO_SVR4_CRC);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}


