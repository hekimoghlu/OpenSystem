/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
31,139,8,0,236,'c',217,'D',0,3,'3','0','7','0','7','0','4','0','0',181,'0',
183,'L',2,210,6,6,'&',134,169,')',' ',218,192,'8',213,2,133,'6','0','0','2',
'1','6','7','0','5','0','N','6','@',5,'&',16,202,208,212,0,';','0',130,'1',
244,24,12,160,246,17,5,136,'U',135,14,146,'`',140,144,' ','G','O',31,215,
' ','E','E','E',134,'Q',128,21,0,0,'%',215,202,221,0,2,0,0};

DEFINE_TEST(test_read_format_cpio_svr4_gzip)
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
	assertEqualInt(ARCHIVE_OK, archive_read_support_format_all(a));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualInt(ARCHIVE_OK, archive_read_next_header(a, &ae));
	assertEqualInt(archive_filter_code(a, 0),
	    ARCHIVE_FILTER_GZIP);
	assertEqualInt(archive_format(a),
	    ARCHIVE_FORMAT_CPIO_SVR4_NOCRC);
	assertEqualInt(archive_entry_is_encrypted(ae), 0);
	assertEqualIntA(a, archive_read_has_encrypted_entries(a), ARCHIVE_READ_FORMAT_ENCRYPTION_UNSUPPORTED);
	assertEqualInt(ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}


